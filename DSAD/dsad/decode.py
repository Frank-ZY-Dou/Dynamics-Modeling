#!/usr/bin/env python
"""Decode a DSAD GoP from its stored form: frames, PSNR / MS-SSIM and decode rate.

The stored form (<run>/stored.npz, dsad/export.py) holds the sites, 9 numbers
each, and the anchor ROM. With --run it is written first if it is missing or
not from the run's current checkpoint. Frame i of T is decoded at
t_i = (i + 1) / T, as in training:

    axes      u = (cos theta, sin theta) of the stored angles, once per GoP
    ROM       Δz, Δcolor = Σ_j φ_j(x) a_j(t_i)
    pack      position tanh(z + Δz) in pixels, color + Δcolor: [N, 10] sites
    build     exact top-8 candidate buffers (dsad/exact_candidates.py)
    render    torch.ops.sad.render_sad, clamped to [0, 1]

The frames written and scored are those of the eager decoder (PyTorch ops). The
decode rate is by default that of the fused CUDA kernels (fast_step.FusedStep).
A pass over the GoP is timed from the stored parameters in GPU memory to the
rendered frames, and the rate is that of the median pass. It is a cold start:
every pass rebuilds the axes and every frame's candidates. --steady adds the
rate with prebuilt candidates (steady state). The fused sites can differ from
the eager ones in the last bits, so the fused decoder's PSNR and its largest
pixel difference from the eager frames are printed too.

usage:
    python dsad/decode.py --run checkpoints/<tag>/SAD3_<it>_<N>/<gop> [--gop <dir> | --data_root <dir>]
                          [--decoder fused|eager] [--batch frame|gop] [--steady] [--out <dir> | --no_write]
                          [--json <file>] [--axis angle|vector] [--size WxH]
    python dsad/decode.py --stored <file.npz> ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from . import _paths
except ImportError:
    import _paths
_paths.setup()

import export  # noqa: E402
from export import RunError, find_gop, read_run_info  # noqa: E402,F401  (used by the tools too)
from exact_candidates import GoPBuilder, build_candidates  # noqa: E402
try:
    from sad_appearance_v3 import vops  # noqa: E402
except ImportError as _e:   # SAD's CUDA extension is not built: --help still works; decoding stops in require_sad()
    vops, _SAD_ERROR = None, _e


def require_sad() -> None:
    """Raise ImportError with the reason if SAD's CUDA extension did not load."""
    if vops is None:
        raise ImportError(f"dsad/decode.py needs SAD's CUDA extension: {_SAD_ERROR}")


def load_frames(gop: Path, device) -> list[torch.Tensor]:
    """Every file of the GoP directory, in sorted order, as [1, 3, H, W] in [0, 1]."""
    import torchvision.transforms as transforms
    from PIL import Image
    return [transforms.ToTensor()(Image.open(gop / f)).unsqueeze(0).to(device)
            for f in sorted(os.listdir(gop))]


@torch.no_grad()
def ground_truth(model, meta: dict, device, gop: Path | None = None, data_root: Path | None = None):
    """(frames, directory) of the GoP to score a decode against, or (None, None) if it is
    not found: `gop` if given, else find_gop on the path recorded at training time (under
    `data_root` if given). A decode onto another grid is scored only against an explicit
    `gop`. Raises ValueError if the GoP does not have the decode's frame count and size or,
    at the fitting size, the pixel checksum recorded in the stored form."""
    T, H, W = meta["frames"], int(model.H), int(model.W)
    resized = (W, H) != (int(meta["width"]), int(meta["height"]))
    if gop is not None and not Path(gop).is_dir():
        raise ValueError(f"--gop {gop}: not a directory")
    if gop is None:
        if resized:    # the recorded GoP has the fitting size
            return None, None
        gop = find_gop(meta.get("gop"), data_root)
        if gop is None:
            return None, None
    gop = Path(gop)
    frames = load_frames(gop, device)
    if not frames:
        raise ValueError(f"{gop}: no frames")
    shape = (len(frames), frames[0].shape[2], frames[0].shape[3])
    if shape != (T, H, W):
        raise ValueError(f"the decode has {T} frames at {H}x{W}, but {gop} has {shape[0]} frames at "
                         f"{shape[1]}x{shape[2]}")
    want = meta.get("gop_sha256")
    if want and not resized and export.gop_pixel_hash(gop) != want:
        raise ValueError(f"{gop} holds other frames than the GoP the run was trained on (pixel "
                         f"checksum {want[:12]}... recorded at export); pass the right --gop / --data_root")
    return frames, gop


def _shown(gop_dir: Path | None) -> str | None:
    """The ground-truth directory as the outputs record it: relative to the DSAD directory
    when it lies inside it."""
    if gop_dir is None:
        return None
    p = Path(gop_dir)
    try:
        return str(p.resolve().relative_to(_paths.REPO.resolve())) if p.is_absolute() else str(p)
    except ValueError:
        return str(p)


def _info(meta: dict, model, **extra) -> dict:
    """The stored form's header with the decode's shape (T, H, W) and anchor mode."""
    return dict(meta, T=meta["frames"], H=int(model.H), W=int(model.W), mode=meta["anchor_mode"], **extra)


@torch.no_grad()
def load(stored: Path, device, gop: Path | None = None, data_root: Path | None = None,
         size: tuple[int, int] | None = None, rescale_radius: bool = True):
    """(model, deform, info, frames) from a stored.npz; frames as ground_truth() finds them
    (None if not found). `size` (W, H) decodes onto another grid (export.build_model)."""
    require_sad()
    model, deform, meta = export.load_stored(stored, device, size=size, rescale_radius=rescale_radius)
    frames, gop_dir = ground_truth(model, meta, device, gop, data_root)
    return model, deform, _info(meta, model, stored=str(stored), gop_dir=_shown(gop_dir)), frames


@torch.no_grad()
def load_run(run: Path, device, gop: Path | None = None, data_root: Path | None = None,
             axis: str = "angle", size: tuple[int, int] | None = None, rescale_radius: bool = True):
    """load() of <run>/stored.npz, exported first if it is missing or not from the current
    checkpoint. axis="vector" decodes the training checkpoint instead, with each axis as the
    trained 2-vector (export.load_checkpoint)."""
    require_sad()
    run = Path(run)
    run_info = read_run_info(run)
    if run_info.get("appearance") == "gaussian":
        raise RunError(f"{run}: a DSAD-GS (--appearance gaussian) run; it renders without "
                       "candidates, so there is nothing to decode here")
    if axis == "angle":
        stored = export.ensure_stored(run, device=device)
        model, deform, info, frames = load(stored, device, gop, data_root, size, rescale_radius)
    elif axis == "vector":
        model, deform, meta = export.load_checkpoint(run, device, size=size, rescale_radius=rescale_radius)
        frames, gop_dir = ground_truth(model, meta, device, gop, data_root)
        info = _info(meta, model, stored=None, gop_dir=_shown(gop_dir))
    else:
        raise ValueError(f"axis must be 'angle' or 'vector', not {axis!r}")
    info["recorded_psnr"] = run_info.get("recorded_psnr")
    return model, deform, info, frames


@torch.no_grad()
def frame_sites(model, deform, i: int, T: int) -> torch.Tensor:
    """Packed [N, 10] sites of frame i at t = (i + 1) / T, with the eager ops of training."""
    d_xyz, _, d_feat = deform.step(model.get_xyz, (i + 1) / T)
    pos = (torch.tanh(model._xyz + d_xyz) + 1.0) * 0.5 * model._wh()
    return model._pack_sites_grad(pos, model._features_dc + d_feat).detach()


def render(model, sites, c0, c1) -> torch.Tensor:
    """render_sad of the sites with candidates (c0, c1): [H, W, 3] clamped to [0, 1]."""
    out = vops.render_sad(c0, c1, sites, float(model.inv_sq), int(model.W), int(model.H))
    if out.dim() == 3 and out.size(-1) == 4:
        out = out[..., :3]
    return out.clamp_(0, 1)


@torch.no_grad()
def decode_gop(model, deform, T: int, builder=None) -> list[torch.Tensor]:
    """The eager decoder: all T frames, [H, W, 3] in [0, 1]. builder: an exact builder
    (default build_candidates; a GoPBuilder gives the same candidates)."""
    build = builder or build_candidates
    outs = []
    for i in range(T):
        sites = frame_sites(model, deform, i, T)
        c0, c1 = build(sites, model.W, model.H, inv_sq=model.inv_sq)
        outs.append(render(model, sites, c0, c1))
    return outs


class FusedDecoder:
    """The fused decoder (fast_step.FusedStep) on a model rebuilt from the stored form.

    Every pass rebuilds the axes from the stored angles and computes the anchor weights
    once. __call__ (cold start) and steady() decode frame by frame, as in the paper. A
    cold start builds every frame's exact candidates; steady state reuses those of
    prebuild(). gop() and steady_gop() decode the whole GoP at once
    (FusedStep.decode_gop) and give the same frames bit for bit."""

    def __init__(self, model, deform, angles: torch.Tensor):
        import fast_step
        core = deform.deform
        stacks = [core.coef_xyz] + ([core.coef_features] if core.enable_features else [])
        if any(st.K > 32 for st in stacks) or any(st.top_k is not None and st.top_k < st.K for st in stacks):
            raise RunError("the fused decoder covers at most 32 knots and no temporal top-K (use --decoder eager)")
        self.model, self.angles = model, angles
        self.fs = fast_step.FusedStep(SimpleNamespace(
            gaussian_model=model, deform=deform, args=SimpleNamespace(),
            lambda_move=0.0, lambda_temporal=0.0, lambda_arap=0.0))
        if not self.fs.E.rom_supported(self.fs.M, self.fs.K, self.fs.C):
            raise RunError(f"the fused decoder does not cover M={self.fs.M}, anchor top-K={self.fs.K}, "
                           f"C={self.fs.C}: it covers anchor top-K 8, 12 or 16 below M, M <= 1,024 and "
                           f"at most 8 coefficient columns (use --decoder eager)")
        self.T = model.num_frames
        self._prebuilt = False               # True once prebuild() ran

    @torch.no_grad()
    def rebuild_axes(self) -> None:
        self.model._aniso_dir.copy_(export.axis_from_angle(self.angles))

    @torch.no_grad()
    def _start_pass(self) -> None:
        self.rebuild_axes()                  # once per GoP pass (the axes do not depend on t)
        self.model.cand_method = "exact"     # eval_candidates builds the exact top-8
        self.fs.invalidate_basis()           # anchor weights and the builder's cap: once per pass
        if self.fs._dec is not None:
            self.fs._dec["topi"].zero_()     # no anchor selection carried over

    @torch.no_grad()
    def __call__(self) -> list[torch.Tensor]:
        """Cold start, frame by frame: each frame's ROM, packing, exact candidates and render."""
        self._start_pass()
        return [self.fs.decode(i, (i + 1) / self.T) for i in range(self.T)]

    @torch.no_grad()
    def gop(self) -> list[torch.Tensor]:
        """Cold start, GoP-batched: the ROM and packing of all frames in one launch, the exact
        candidates in batched builds of up to max_frames frames, one render per frame. The
        same frames as __call__."""
        self._start_pass()
        return list(self.fs.decode_gop(self.T).unbind(0))

    @torch.no_grad()
    def prebuild(self) -> None:
        """Build every frame's exact candidates from the fused sites and keep them as the
        frames' candidate buffers, for steady() and steady_gop()."""
        m = self.model
        self.rebuild_axes()
        m.cand_method = "exact"
        self.fs.invalidate_basis()
        for i in range(self.T):
            self.fs.decode(i, (i + 1) / self.T)          # leaves frame i's sites in _dec
            c0, c1 = build_candidates(self.fs._dec["sites"], m.W, m.H, inv_sq=m.inv_sq)
            m._frame_cand0[i], m._frame_cand1[i] = c0.clone(), c1.clone()
            m._frame_init[i] = True
        m._active_frame = None
        self._prebuilt = True

    @torch.no_grad()
    def steady(self) -> list[torch.Tensor]:
        """Steady state, frame by frame: fused ROM, packing and render with the candidates
        of prebuild()."""
        self.rebuild_axes()
        self.model.cand_method = "vpt"       # eval_candidates returns the frame's buffers
        self.fs.invalidate_basis()
        outs = [self.fs.decode(i, (i + 1) / self.T) for i in range(self.T)]
        self.model.cand_method = "exact"
        return outs

    @torch.no_grad()
    def steady_gop(self) -> list[torch.Tensor]:
        """Steady state, GoP-batched: the ROM and packing of all frames in one launch, one
        render per frame. The same frames as steady(). It renders with the frames' candidate
        buffers as they are: call prebuild() again after any change to the parameters."""
        m = self.model
        missing = [i for i in range(self.T) if m._frame_cand0[i] is None or not m._frame_init[i]]
        if not self._prebuilt or missing:
            raise RuntimeError("steady_gop: no prebuilt candidates; call prebuild() first")
        if m._active_frame is not None:   # the active frame's buffers are in the candidate updater
            m._frame_cand0[m._active_frame] = m.cand_updater.cand0
            m._frame_cand1[m._active_frame] = m.cand_updater.cand1
        self.rebuild_axes()
        self.fs.invalidate_basis()
        cands = [(m._frame_cand0[i], m._frame_cand1[i]) for i in range(self.T)]
        return list(self.fs.decode_gop(self.T, cands=cands).unbind(0))


def timed(fn, warmup: int, repeats: int) -> tuple[list[float], list]:
    """Seconds of each of `repeats` passes of fn() after `warmup` untimed ones."""
    out = None
    for _ in range(warmup):
        out = fn()
    secs = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t0)
    return secs, out


def rate(secs: list[float], T: int) -> dict:
    """Frames per second and ms per frame from the median pass time."""
    med = statistics.median(secs)
    return {"fps": T / med, "ms_per_frame": 1000 * med / T, "passes": len(secs)}


@torch.no_grad()
def scores(outs: list[torch.Tensor], frames: list[torch.Tensor]) -> dict:
    """Per-frame PSNR and MS-SSIM of decoded [H, W, 3] frames against [1, 3, H, W] ground truth."""
    from pytorch_msssim import ms_ssim
    psnr, msss = [], []
    for out, gt in zip(outs, frames):
        mse = F.mse_loss(out, gt.squeeze(0).permute(1, 2, 0)).item()
        psnr.append(10 * math.log10(1.0 / max(mse, 1e-12)))
        msss.append(ms_ssim(out.permute(2, 0, 1).unsqueeze(0).float(), gt.float(),
                            data_range=1, size_average=True).item())
    return {"psnr": sum(psnr) / len(psnr), "ms_ssim": sum(msss) / len(msss),
            "per_frame": psnr, "per_frame_ms_ssim": msss}


@torch.no_grad()
def psnr_only(outs, frames) -> float:
    per = [10 * math.log10(1.0 / max(F.mse_loss(o, g.squeeze(0).permute(1, 2, 0)).item(), 1e-12))
           for o, g in zip(outs, frames)]
    return sum(per) / len(per)


def write_frames(outs: list[torch.Tensor], out_dir: Path) -> None:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, o in enumerate(outs):
        arr = (o * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(arr).save(out_dir / f"frame_{i + 1:04d}.png")


def parse_size(text: str) -> tuple[int, int]:
    """Parse 'WxH', e.g. '1920x1080', into (W, H)."""
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--size takes WxH, e.g. 1920x1080, not {text!r}") from None
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"--size {text}: width and height must be positive")
    return w, h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path,
                     help="run directory (gaussian.pth, deform_sad/, train.txt); its stored.npz is "
                          "written first if missing or not from the current checkpoint")
    src.add_argument("--stored", type=Path, help="a stored.npz written by dsad/export.py")
    ap.add_argument("--gop", type=Path, default=None,
                    help="ground-truth GoP directory (default: the one the run was trained on)")
    ap.add_argument("--data_root", type=Path, default=None,
                    help="look the recorded GoP path up under this directory instead "
                         "(the longest trailing part of it that exists there)")
    ap.add_argument("--decoder", choices=["fused", "eager"], default="fused",
                    help="decoder to time (default fused, the paper's); the frames written and "
                         "scored are the eager decoder's")
    ap.add_argument("--warmup", type=int, default=5, help="untimed passes before timing")
    ap.add_argument("--repeats", type=int, default=30, help="timed passes over the GoP (median)")
    ap.add_argument("--batch", choices=["frame", "gop"], default="frame",
                    help="the fused decoder's timed pass: frame (default) decodes frame by frame, a batch of "
                         "one frame, the paper's protocol; gop decodes the whole GoP at once (the same frames, "
                         "faster)")
    ap.add_argument("--steady", action="store_true",
                    help="also time the fused decoder with the candidates prebuilt (not a cold start)")
    ap.add_argument("--axis", choices=["angle", "vector"], default="angle",
                    help="angle (default): decode the stored form, as the paper scores DSAD. vector: decode "
                         "the training checkpoint with each axis as the trained 2-vector (needs --run; not "
                         "timed)")
    ap.add_argument("--size", type=parse_size, default=None, metavar="WxH",
                    help="decode onto this grid instead of the fitting one, e.g. 1920x1080 for a 1280x720 "
                         "fit (the radius is rescaled); score against --gop at that size (not timed)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the decoded frames (default: decoded/ next to stored.npz; "
                         "decoded_vector/ or decoded_<W>x<H>/ for the other decodes)")
    ap.add_argument("--no_write", action="store_true", help="do not write PNG frames")
    ap.add_argument("--json", type=Path, default=None, help="also write the results here")
    args = ap.parse_args()
    if vops is None:
        raise SystemExit(f"cannot load SAD's CUDA extension: {_SAD_ERROR}")
    if args.axis == "vector" and args.run is None:
        ap.error("--axis vector decodes a run's checkpoint: pass --run")
    if args.decoder == "eager" and args.batch == "gop":
        ap.error("--batch gop is the fused decoder's GoP pass; the eager decoder decodes frame by frame")

    dev = torch.device("cuda:0")
    try:
        if args.run is not None:
            model, deform, info, frames = load_run(args.run, dev, args.gop, args.data_root, args.axis, args.size)
        else:
            model, deform, info, frames = load(args.stored, dev, args.gop, args.data_root, args.size)
    except (RunError, ValueError) as e:
        raise SystemExit(str(e))
    T = info["T"]
    resized = (info["W"], info["H"]) != (info["width"], info["height"])
    base = Path(info["stored"]).parent if info["stored"] else Path(args.run)
    print(f"{info['stored'] or args.run}: N={info['N']} M={info['M']} K={info['K']} top-K={info['anchor_top_k']} "
          f"mode={info['mode']} features={info['features']}  T={T} {info['W']}x{info['H']}"
          + (f" (fitted at {info['width']}x{info['height']})" if resized else "")
          + (f"  stored parameters: {info['stored_params']:,}" if args.axis == "angle" else
             f"  checkpoint numbers (axes as 2-vectors): {info['stored_params']:,}")
          + f" ({info['per_site']} per site + ROM {info['rom_params']:,})")
    result = {"stored": info["stored"], "axis": args.axis, "stored_params": info["stored_params"],
              "config": {k: info[k] for k in ("N", "M", "K", "anchor_top_k", "mode", "features",
                                              "opacity", "T", "H", "W", "gop", "gop_dir")}}

    # The reference decode (eager): the frames written and scored.
    eager = decode_gop(model, deform, T)
    decoder = outs = None
    if args.axis == "angle" and not resized:
        decoder, outs = time_decoders(model, deform, Path(info["stored"]), T, args, result, eager)
    else:
        print("not timed: decode rates are measured on the stored form at its fitting size")

    if frames is not None:
        s = scores(eager, frames)
        rec = info.get("recorded_psnr")
        print(f"decoded PSNR {s['psnr']:.4f} dB  MS-SSIM {s['ms_ssim']:.6f}  (eager decoder)"
              + (f"  (training-time PSNR in train.txt: {rec:.4f})" if rec is not None and not resized else ""))
        result.update(s, recorded_psnr=rec)
        if decoder == "fused":
            pf = psnr_only(outs, frames)
            result["fused_psnr"] = pf
            print(f"fused decoder: PSNR {pf:.4f} dB ({pf - s['psnr']:+.1e} vs eager), largest pixel "
                  f"difference {result['fused_max_abs_diff_vs_eager']:.3g}")
    elif resized:
        print(f"PSNR not computed: pass the ground truth at {info['W']}x{info['H']} with --gop")
    else:
        print(f"no ground-truth GoP found at {info.get('gop') or '(none recorded)'}"
              f"{f' under {args.data_root}' if args.data_root else ''}: PSNR not computed "
              "(pass --gop or --data_root)")
    if not args.no_write:
        name = "decoded" + ("_vector" if args.axis == "vector" else "") + (
            f"_{info['W']}x{info['H']}" if resized else "")
        out_dir = args.out or (base / name)
        write_frames(eager, out_dir)
        print(f"wrote {len(eager)} frames to {out_dir}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


def time_decoders(model, deform, stored: Path, T: int, args, result: dict, eager: list):
    """Time the cold-start decoder (--decoder) and, with --steady, the steady state; add the
    rates to `result`. Returns (the decoder timed, its frames of the last timed pass)."""
    dev = model._xyz.device
    angles = export.stored_angles(stored, dev)
    decoder, fused = args.decoder, None
    if decoder == "fused":
        try:
            fused = FusedDecoder(model, deform, angles)
        except RunError as e:
            print(f"fused decoder unavailable: {e}; timing the eager decoder")
            decoder = "eager"
    batch = args.batch if decoder == "fused" else "frame"   # the eager decoder runs frame by frame
    if args.batch != batch:
        print("timing the eager decoder, frame by frame (the GoP pass is the fused decoder's)")
    pass_name = "frame by frame" if batch == "frame" else "GoP-batched"
    if decoder == "fused":
        fn = fused if batch == "frame" else fused.gop
    else:
        gb = GoPBuilder()

        @torch.no_grad()
        def fn():
            model._aniso_dir.copy_(export.axis_from_angle(angles))   # once per GoP pass
            gb.reset()
            return decode_gop(model, deform, T, builder=gb)
    secs, outs = timed(fn, max(0, args.warmup), max(1, args.repeats))
    r = rate(secs, T)
    result.update(decoder=decoder, batch=batch, decode_fps=r["fps"], decode_ms_per_frame=r["ms_per_frame"],
                  timed_passes=r["passes"], gpu=torch.cuda.get_device_name(0))
    print(f"cold-start decode ({decoder}, {pass_name}, exact top-8 candidates): {r['ms_per_frame']:.3f} "
          f"ms/frame = {r['fps']:.1f} FPS  (median of {r['passes']} passes after {args.warmup} warm-up; "
          f"{torch.cuda.get_device_name(0)})")
    if decoder == "fused":
        result["fused_max_abs_diff_vs_eager"] = max(float((a - b).abs().max()) for a, b in zip(outs, eager))
    if args.steady:
        try:
            if fused is None:
                fused = FusedDecoder(model, deform, angles)
        except RunError as e:
            print(f"steady state not timed: {e}")
        else:
            fused.prebuild()
            s_secs, _ = timed(fused.steady if batch == "frame" else fused.steady_gop,
                              max(0, args.warmup), max(1, args.repeats))
            rs = rate(s_secs, T)
            result.update(steady_fps=rs["fps"], steady_ms_per_frame=rs["ms_per_frame"], steady_batch=batch)
            print(f"steady state (fused, {pass_name}, candidates prebuilt; not a cold start): "
                  f"{rs['ms_per_frame']:.3f} ms/frame = {rs['fps']:.1f} FPS")
    return decoder, outs


if __name__ == "__main__":
    main()
