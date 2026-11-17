#!/usr/bin/env python
"""Check the exact top-8 candidate builder (dsad/exact_candidates.py) on a DSAD run.

    (a) crops    The share of pixels whose 8-set equals a torch brute force over all
                 sites, on crops of the first and last frame: window mode per radius,
                 exact mode (mismatches within fp32 rounding counted as near-ties)
                 and converged VPT. On full frames, exact mode against the CUDA
                 brute force.
    (b) PSNR     Every frame rendered with the candidates of exact mode (fp32 and
                 fp16 key), exact + k VPT passes and converged VPT, next to the PSNR
                 recorded in train.txt.
    (c) rescans  The cooperative rescan against the per-thread one, at caps and
                 tile-list capacities that force rescans and at tile sizes 16 and 8
                 (candidates and statistics must be identical), and GoPBuilder
                 against build_candidates.

usage:
    python tools/exact_candidates_check.py --ckpt <run_dir> [--gop <gop_dir>]

The run is decoded from its stored form, which is written first if needed. The exit
status is 1 if (c) fails.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dsad"))
import _paths  # noqa: E402

_paths.setup()
import decode  # noqa: E402
from exact_candidates import GoPBuilder, build_candidates, candidates_to_sets  # noqa: E402
from sad_appearance_v3 import vops  # noqa: E402


def load_run(ckpt: Path, gop: Path | None, dev):
    """(model, deform, frames, recorded PSNR) of a run, rebuilt from its stored form."""
    try:
        model, deform, info, frames = decode.load_run(ckpt, dev, gop)
    except (decode.RunError, ValueError) as e:
        raise SystemExit(str(e))
    if frames is None:
        raise SystemExit(f"{ckpt}: ground-truth GoP not found; pass --gop")
    return model, deform, frames, info.get("recorded_psnr") or float("nan")


@torch.no_grad()
def frame_sites(model, deform, i: int, T: int) -> torch.Tensor:
    """Packed [N, 10] sites of frame i (t = (i+1)/T, as in training)."""
    wh = torch.tensor([model.W, model.H], device=model._xyz.device, dtype=torch.float32)
    d_xyz, _, d_feat = deform.step(model.get_xyz, (i + 1) / T)
    pos = (torch.tanh(model._xyz + d_xyz) + 1.0) * 0.5 * wh
    return model._pack_sites_grad(pos, model._features_dc + d_feat).detach()


def reset_candidates(model) -> None:
    T = model.num_frames
    model._frame_cand0 = [None] * T
    model._frame_cand1 = [None] * T
    model._frame_init = [False] * T
    model._frame_step_idx = [0] * T
    model._frame_iter_count = [0] * T
    model._active_frame = None


@torch.no_grad()
def vpt_pass(model, sites, c0, c1, step_idx: int, jump: int):
    """One SAD update_candidates_compact pass with an explicit jump (as _vpt_step)."""
    model._ensure_packed_buffers(sites)
    vops.pack_candidate_sites(sites, model._packed, model.num_points)
    o0, o1 = torch.empty_like(c0), torch.empty_like(c1)
    vops.update_candidates_compact(
        c0, c1, o0, o1, model._packed, model._dummy_hilbert, model._dummy_hilbert,
        float(model.inv_sq), int(model.num_points), (int(jump) << 16) | (step_idx & 0xFFFF),
        step_idx >> 16, float(model.cand_radius_scale), int(model.cand_radius_probes),
        int(model.cand_inject_count), 0, 0, 1, int(model.W), int(model.H), int(model.W),
        int(model.H))
    return o0, o1


@torch.no_grad()
def vpt_converged(model, sites, i: int, passes: int):
    """JFA seed + flood + warm-up (_jfa_warmup), then `passes` _vpt_step passes."""
    model._activate_frame(i)
    model._jfa_warmup(sites, i)
    for _ in range(passes):
        model._vpt_step(sites, i)
    c0, c1 = model.cand_updater.get()
    return c0.clone(), c1.clone()


@torch.no_grad()
def render_psnr(model, sites, c0, c1, gt) -> float:
    out = vops.render_sad(c0, c1, sites, float(model.inv_sq), int(model.W), int(model.H))
    out = out[..., :3] if out.dim() == 3 and out.size(-1) == 4 else out
    mse = F.mse_loss(out.clamp(0, 1), gt.squeeze(0).permute(1, 2, 0)).item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def torch_keys(sites: torch.Tensor, xy: torch.Tensor, inv_sq: float, dtype=torch.float32):
    """SAD candidate keys [P, N] of all sites at the pixels xy [P, 2] (integer coordinates).

    Same operation order as SAD's voronoi_dmix2 (fp32 by default). torch uses no FMA,
    so a key may differ from the CUDA one in the last bit.
    """
    s = sites.to(dtype)
    px, py, lt, rad = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    ux, uy, la = s[:, 7], s[:, 8], s[:, 9]
    dx = xy[:, 0:1].to(dtype) - px[None]
    dy = xy[:, 1:2].to(dtype) - py[None]
    diff2 = dx * dx + dy * dy
    proj = ux[None] * dx + uy[None] * dy
    proj2 = proj * proj
    perp2 = (diff2 - proj2).clamp_min(0)
    l1 = torch.exp(la)
    l2 = 1.0 / l1
    d2n = (l1[None] * proj2 + l2[None] * perp2) * torch.tensor(inv_sq, dtype=dtype)
    inv_scale = torch.sqrt(torch.tensor(inv_sq, dtype=torch.float32)).to(dtype)
    tau = torch.exp(lt).clamp_min(1e-4)
    key = tau[None] * (d2n.clamp_min(1e-8).sqrt() - (rad * inv_scale)[None])
    key[:, ~(px >= 0)] = float("inf")      # SAD: position.x < 0 is an inactive site
    return key


@torch.no_grad()
def brute_crop(sites, inv_sq, x0, y0, w, h, chunk=512):
    """Exact top-8 (fp32 torch) for the crop's pixels, `chunk` pixels at a time."""
    dev = sites.device
    ys, xs = torch.meshgrid(torch.arange(y0, y0 + h, device=dev),
                            torch.arange(x0, x0 + w, device=dev), indexing="ij")
    xy = torch.stack([xs, ys], -1).reshape(-1, 2)
    idx = []
    for k in range(0, xy.shape[0], chunk):
        idx.append(torch_keys(sites, xy[k:k + chunk], inv_sq).topk(8, largest=False).indices)
    return xy, torch.cat(idx).sort(dim=1).values


@torch.no_grad()
def near_tie(sites, inv_sq, xy, mine, ref):
    """For mismatching pixels: (is_tie, relative fp64 gap) of the swapped sites.

    The fp32 key is ill-conditioned for thin, distant sites: d2 - proj^2 cancels, and
    the large perpendicular coefficient e^{-a} amplifies the error to about
    eps * tau s * e^{-3a/2} * |x - p|. CUDA (with FMA) and torch (without) round such
    keys differently, so two sites whose exact keys are that close can swap places. A
    mismatch is a tie when the swapped sites' fp64 keys differ by less than the sum of
    their fp32 error bounds (the model of the kernel's safety margin, kErrScale = 2e-6).
    """
    s64 = sites.double()
    W = 1.0 / math.sqrt(inv_sq)
    ties, gaps = [], []
    for p in range(xy.shape[0]):
        a, b = set(mine[p].tolist()), set(ref[p].tolist())
        swapped = sorted(a ^ b)
        k64 = torch_keys(sites, xy[p:p + 1], inv_sq, torch.float64)[0, swapped]
        s = s64[swapped]
        ts = s[:, 2].exp().clamp_min(1e-4) / W
        l1 = s[:, 9].exp()
        amp = (1.0 / l1) / torch.minimum(l1, 1.0 / l1).sqrt()      # B / sqrt(min(A, B))
        dist = (s[:, 0:2] - xy[p].double()).norm(dim=1)
        err = (ts * 2e-6 * amp * dist + 1e-6 * k64.abs().clamp_min(1.0)).sum()
        gap = (k64.max() - k64.min()).item()
        ties.append(gap <= err.item())
        gaps.append(gap / max(1.0, k64.abs().max().item()))
    return torch.tensor(ties), gaps


def part_a(model, deform, frames, radii, vpt_passes):
    T, W, H = len(frames), model.W, model.H
    s = 128
    crops = [(W // 2 - s // 2, H // 2 - s // 2), (W // 4 - s // 2, H // 4 - s // 2),
             (3 * W // 4 - s // 2, 3 * H // 4 - s // 2), (0, 0), (W - s, H - s)]
    rows, gaps = {}, []
    for i in sorted({0, T - 1}):
        sites = frame_sites(model, deform, i, T)
        reset_candidates(model)
        v0, v1 = vpt_converged(model, sites, i, vpt_passes)
        # The crops use the fp32 key; key_fp16 is checked on full frames below.
        e0, e1 = build_candidates(sites, W, H, inv_sq=model.inv_sq, key_fp16=False)
        cand = {"exact": candidates_to_sets(e0, e1), f"VPT-{vpt_passes}": candidates_to_sets(v0, v1)}
        for R in radii:
            w0, w1 = build_candidates(sites, W, H, inv_sq=model.inv_sq, mode="window",
                                      radius_px=R, key_fp16=False)
            cand[f"window R={R}"] = candidates_to_sets(w0, w1)
        for (x0, y0) in crops:
            xy, ref = brute_crop(sites, model.inv_sq, x0, y0, s, s)
            pix = (xy[:, 1] * W + xy[:, 0]).long()
            for name, sets in cand.items():
                same = (sets[pix] == ref).all(1)
                tie = 0
                if name == "exact" and not bool(same.all()):
                    bad = (~same).nonzero().flatten()
                    t, g = near_tie(sites, model.inv_sq, xy[bad], sets[pix][bad], ref[bad])
                    tie = int(t.sum())
                    gaps += g
                acc = rows.setdefault(name, [0, 0, 0])
                acc[0] += int(same.sum())
                acc[1] += tie
                acc[2] += same.numel()
    # Full frames, all of them: one batched exact build of the whole GoP vs the
    # CUDA brute force (same key code, every pixel scans every site) per frame,
    # for both key precisions.
    all_sites = torch.stack([frame_sites(model, deform, i, T) for i in range(T)])
    for kf in (False, True):
        e0, e1, infos = build_candidates(all_sites, W, H, inv_sq=model.inv_sq, key_fp16=kf,
                                         return_info=True)
        same = []
        for i in range(T):
            b0, b1 = build_candidates(all_sites[i], W, H, inv_sq=model.inv_sq, mode="brute",
                                      key_fp16=kf)
            same.append(((e0[i] == b0).all(1) & (e1[i] == b1).all(1)).float().mean().item())
        print(f"  key_fp16={int(kf)}: batched exact build of all {T} frames == CUDA brute force "
              f"(full frames, ordered lists): min over frames {100 * min(same):.4f}%  "
              f"[fallback px {sum(x['fallback_pixels'] for x in infos)}, "
              f"overflow tiles {sum(x['overflow_tiles'] for x in infos)}]", flush=True)
    print(f"  (a) crops ({len(crops)} x {s}x{s} px, frames {sorted({0, T - 1})}) vs torch brute force:")
    for name, (ok, tie, n) in rows.items():
        extra = ""
        if name == "exact":
            extra = (f"   + {100 * tie / n:.4f}% fp32 rounding ties "
                     f"(n={len(gaps)}, max rel fp64 gap {max(gaps, default=0):.1e})")
        print(f"      {name:14s} identical 8-sets {100 * ok / n:8.4f}%{extra}", flush=True)


def part_b(model, deform, frames, vpt_passes, extra_passes, recorded):
    T, W, H = len(frames), model.W, model.H
    res = {}
    agree = {"exact": 0.0, "exact-fp16": 0.0}
    for i in range(T):
        sites = frame_sites(model, deform, i, T)
        gt = frames[i]
        reset_candidates(model)
        v0, v1 = vpt_converged(model, sites, i, vpt_passes)
        res.setdefault(f"converged VPT ({vpt_passes} passes)", []).append(
            render_psnr(model, sites, v0, v1, gt))
        vset = candidates_to_sets(v0, v1)
        for name, kf in (("exact", False), ("exact-fp16", True)):
            e0, e1 = build_candidates(sites, W, H, inv_sq=model.inv_sq, key_fp16=kf)
            res.setdefault(name, []).append(render_psnr(model, sites, e0, e1, gt))
            agree[name] += (candidates_to_sets(e0, e1) == vset).all(1).float().mean().item() / T
            if not kf:
                c0, c1 = e0, e1
                done = 0
                jumps = (8, 4, 2, 1)
                for k in sorted(extra_passes):
                    while done < k:
                        c0, c1 = vpt_pass(model, sites, c0, c1, done, jumps[done % len(jumps)])
                        done += 1
                    res.setdefault(f"exact + {k} VPT", []).append(
                        render_psnr(model, sites, c0, c1, gt))
    order = ["exact", "exact-fp16"] + [f"exact + {k} VPT" for k in sorted(extra_passes)] + \
            [f"converged VPT ({vpt_passes} passes)"]
    print(f"  (b) mean PSNR over {T} frames (recorded by the run: {recorded:.4f}):")
    for name in order:
        v = res[name]
        print(f"      {name:28s} {sum(v) / len(v):.4f}   per-frame "
              + " ".join(f"{x:.3f}" for x in v), flush=True)
    for name, a in agree.items():
        print(f"      pixels whose {name} 8-set equals converged VPT's: {100 * a:.3f}%")
    return {k: sum(v) / len(v) for k, v in res.items()}


def part_c(model, deform, frames) -> bool:
    """(c) the cooperative rescan against the per-thread one; GoPBuilder against build_candidates."""
    T, W, H = len(frames), model.W, model.H
    cases = [(0.998, 256), (0.9, 256), (0.5, 256), (1e-6, 256), (0.998, 4), (0.998, 1)]
    n = bad = 0
    for i in sorted({0, T - 1}):
        sites = frame_sites(model, deform, i, T)
        for q, eps in cases:
            for tile in (16, 8):
                (a0, a1, ai), (b0, b1, bi) = (
                    build_candidates(sites, W, H, inv_sq=model.inv_sq, cap_quantile=q, entries_per_site=eps,
                                     tile_px=tile, coop_fallback=coop, return_info=True)
                    for coop in (True, False))
                same = torch.equal(a0, b0) and torch.equal(a1, b1) and ai == bi
                n += 1
                bad += not same
                if not same or (q, eps, tile) == (0.5, 256, 16):
                    print(f"      frame {i} cap {q:g} entries {eps} tile {tile}: rescanned px "
                          f"{ai['fallback_pixels']}, overflow tiles {ai['overflow_tiles']}: "
                          f"{'identical' if same else 'DIFFERENT'}", flush=True)
    gb, gsame = GoPBuilder(), True
    all_sites = torch.stack([frame_sites(model, deform, i, T) for i in range(T)])
    g0, g1 = gb(all_sites, W, H, inv_sq=model.inv_sq)
    for i in range(T):
        r0, r1 = build_candidates(all_sites[i], W, H, inv_sq=model.inv_sq)
        gsame &= torch.equal(g0[i], r0) and torch.equal(g1[i], r1)
    print(f"  (c) cooperative vs per-thread rescan: {n - bad}/{n} builds identical (candidates and "
          f"statistics); GoPBuilder (cap {gb.q:g}) == build_candidates on all {T} frames: {gsame}", flush=True)
    return bad == 0 and gsame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="run directory")
    ap.add_argument("--gop", type=Path, default=None,
                    help="ground-truth GoP (default: the one the run trained on)")
    ap.add_argument("--vpt_passes", type=int, default=256)
    ap.add_argument("--extra_passes", default="2,4,8",
                    help="VPT passes run on top of the exact candidates")
    ap.add_argument("--radii", default="8,16,32,64,128,256")
    ap.add_argument("--skip_a", action="store_true")
    ap.add_argument("--skip_b", action="store_true")
    ap.add_argument("--skip_c", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    model, deform, frames, recorded = load_run(args.ckpt, args.gop, dev)
    print(f"{args.ckpt}: N={model.num_points} T={len(frames)} {model.W}x{model.H}", flush=True)
    t0 = time.time()
    if not args.skip_a:
        part_a(model, deform, frames, [int(r) for r in args.radii.split(",")], args.vpt_passes)
    if not args.skip_b:
        part_b(model, deform, frames, args.vpt_passes,
               [int(k) for k in args.extra_passes.split(",") if k], recorded)
    ok = True
    if not args.skip_c:
        ok = part_c(model, deform, frames)
    print(f"  done in {time.time() - t0:.0f}s")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
