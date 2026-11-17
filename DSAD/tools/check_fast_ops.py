#!/usr/bin/env python
"""Check the fused training step and decoder (--fast) against the eager ones.

After a few hundred eager iterations per stage (config: configs/dsad.yaml by
default), the same inputs go through both steps:

    coarse    site packing, clamp + MSE gradient, site gradients (bitwise)
    fine      top-K anchor sets (exact); anchor weights, residuals, sites, gradients and
              logged penalty sums (largest difference; gradients also against float64)
    noise     the same gradients from two eager render backwards (SAD's backward
              accumulates with atomics)
    optim     fused Adan and Adam against D2GV's Adan and torch.optim.Adam (bitwise)
    decode    FusedStep.decode against the eager decoder; its render against
              render_sad + clamp and decode_gop against decode() (bitwise)
    vpt       the fused VPT pass against SAD's update_candidates_compact (bitwise), and
              which of the two the model's VPT passes call
    render    synthetic: the fused render + MSE + backward and render_decode against
              SAD's render_sad and render_sad_backward (bitwise)
    romapply  synthetic: rom_apply and a warm-started rom_forward against a cold
              rom_forward (bitwise)
    exact     with --exact_cands: the above on exact top-8 candidates

At weight 0 (the paper's configuration) the fused step skips the temporal and ARAP
penalties, so their logged sums stay 0 and those lines show rel = 1.00. Pass a weight
to compare them, e.g. -- --lambda_arap 1e-3.

usage:
    python tools/check_fast_ops.py -d baselines/D2GV/dataset/bunny/bunny_1 --num_points 27047
        [--config configs/dsad.yaml] [--exact_cands] [--out <file.json>] [-- <trainer flags>]
    python tools/check_fast_ops.py --only vpt[,render,romapply,optim]
        the synthetic checks alone, without a dataset

The exit status is 1 if a check labeled bitwise fails. The other lines only report
differences.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "dsad"))
import _paths  # noqa: E402

_paths.setup()

RESULTS = {}
OPT_STEPS = 60


def report(name, a, b, bitwise=False):
    a = a.detach().float()
    b = b.detach().float()
    diff = (a - b).abs()
    mx = diff.max().item() if diff.numel() else 0.0
    scale = b.abs().max().item() if b.numel() else 0.0
    neq = (a != b).sum().item()
    rel = mx / scale if scale > 0 else 0.0
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item() if a.numel() > 1 else 1.0
    tag = ("BITWISE" if neq == 0 else f"{neq} elems differ") if bitwise else ""
    print(f"  {name:<34s} max|diff|={mx:.3e}  rel={rel:.2e}  cos={cos:.9f}  {tag}")
    RESULTS[name] = {"max_abs": mx, "rel": rel, "neq": neq, "n": a.numel(), "cos": cos}
    if bitwise:
        RESULTS[name]["bitwise"] = neq == 0
    return mx


def eager_dense_logits(anchors, x):
    """MotionSAD.basis up to the logits (same ops as the module)."""
    diff = x.unsqueeze(1) - anchors.mu.unsqueeze(0)
    mode = anchors.mode
    if mode == "voronoi":
        dist = (diff * diff).sum(-1).clamp_min(1e-12).sqrt()
        return -dist * torch.exp(-anchors.log_sigma).unsqueeze(0)
    if mode == "apollonius":
        dist = (diff * diff).sum(-1).clamp_min(1e-12).sqrt()
        return (anchors.radius.unsqueeze(0) - dist) * torch.exp(-anchors.log_sigma).unsqueeze(0)
    if mode == "power":
        sq = (diff * diff).sum(-1)
        return (anchors.weight.unsqueeze(0) - sq) * torch.exp(-2.0 * anchors.log_sigma).unsqueeze(0)
    u = anchors.aniso_dir / (anchors.aniso_dir.norm(dim=-1, keepdim=True) + 1e-8)
    proj = (diff * u.unsqueeze(0)).sum(-1)
    sq = (diff * diff).sum(-1)
    perp_sq = (sq - proj * proj).clamp_min(0.0)
    ea = anchors.log_aspect.exp().unsqueeze(0)
    sq_G = ea * proj * proj + (1.0 / ea) * perp_sq
    dist_G = sq_G.clamp_min(1e-12).sqrt()
    return (anchors.radius.unsqueeze(0) - dist_G) * torch.exp(-anchors.log_sigma).unsqueeze(0)


def site_param_names():
    return ["_xyz", "_features_dc", "_log_tau", "_radius", "_aniso_dir", "_log_aniso"]


def eager_fine_graph(tr, frame_idx, timestamp, gt, cands):
    """The eager fine-stage forward (as in train()), with fixed candidate buffers."""
    gm = tr.gaussian_model
    core = tr.deform.deform
    import losses_motion
    d_xyz, d_op, d_feat = tr.deform.step(gm.get_xyz, timestamp)
    xyz_def = torch.tanh(gm._xyz + d_xyz)
    feat_def = gm._features_dc + d_feat
    pos_px = (xyz_def + 1.0) * 0.5 * gm._wh()
    sites = gm._pack_sites_grad(pos_px, feat_def)
    rendered = gm.renderer(sites, cands[0], cands[1], gm.W, gm.H)
    rec = F.mse_loss(rendered.clamp(0, 1), gt)
    reg = 0.0
    if tr.lambda_move > 0:
        reg = reg + tr.lambda_move * losses_motion.movement_penalty(d_xyz)
    if tr.lambda_temporal > 0:
        reg = reg + tr.lambda_temporal * losses_motion.temporal_smoothness(core.coef_xyz)
    if tr.lambda_arap > 0:
        deformed = torch.tanh(gm._xyz + d_xyz)
        reg = reg + tr.lambda_arap * losses_motion.arap(gm.get_xyz.detach(), deformed, tr.knn_idx)
    return dict(d_xyz=d_xyz, d_feat=d_feat, sites=sites, rendered=rendered, rec=rec, reg=reg)


def exact_cands(gm, sites):
    """The decoder's exact top-8 candidate buffers for these sites (--cand_exact_tail)."""
    from sad_appearance_v3 import _build_exact
    c0, c1 = _build_exact(sites.detach().contiguous(), gm.W, gm.H, gm.inv_sq)
    return c0.clone(), c1.clone()


def report_cands(name, a, b):
    """Fraction of pixels whose candidate slots differ between two buffer pairs."""
    same = None
    for x, y in zip(a, b):
        eq = (x == y).reshape(-1, x.shape[-1]).all(-1)
        same = eq if same is None else same & eq
    frac = 1.0 - same.float().mean().item()
    print(f"  {name:<34s} pixels with a different top-8: {frac:.3e}")
    RESULTS[name] = {"diff_frac": frac, "identical": frac == 0.0}
    return frac


def exact_mode_path(gm, sites, frame_idx, ref):
    """gm._candidates in exact mode (the trainer's tail switch) against the builder."""
    gm.cand_method = "exact"
    gm._frame_init[frame_idx] = False
    got = tuple(c.clone() for c in gm._candidates(sites.detach().contiguous(), frame_idx))
    return report_cands("exact.gm_candidates_vs_builder", got, ref)


def all_params(tr):
    gm = tr.gaussian_model
    core = tr.deform.deform
    out = {f"site.{n}": getattr(gm, n) for n in site_param_names()}
    a = core.anchors
    for n in ["mu", "log_sigma", "radius", "log_aspect", "aniso_dir", "weight"]:
        if hasattr(a, n):
            out[f"anchor.{n}"] = getattr(a, n)
    stacks = [("coef_xyz", core.coef_xyz)]
    if core.enable_features:
        stacks.append(("coef_features", core.coef_features))
    for sn, s in stacks:
        for n in ["t", "log_r", "v"]:
            out[f"{sn}.{n}"] = getattr(s, n)
    return out


def fused_grads(fs):
    out = {}
    for n, g in zip(site_param_names(), fs.site_grads):
        out[f"site.{n}"] = g
    for n, p, g in zip(["mu", "log_sigma", "radius", "log_aspect", "aniso_dir", "weight"],
                       fs.anchor_params, fs.anchor_grads):
        if p.numel():
            out[f"anchor.{n}"] = g
    names = ["coef_xyz", "coef_features"]
    for sn, gs in zip(names, fs.stack_grads):
        for n, g in zip(["t", "log_r", "v"], gs):
            out[f"{sn}.{n}"] = g
    return out


def check_render_terms(device):
    """Fused render+MSE+backward vs SAD's render_sad / render_sad_backward with
    every site in exactly one (pixel, candidate) slot: each grad_sites entry is a
    single term, so the comparison is exact."""
    import fast_step
    E = fast_step.ext()
    vops = torch.ops.sad
    print("\n[render] synthetic image, one (pixel, candidate) per site")
    W, H = 64, 32
    P = W * H
    N = P * 8
    g = torch.Generator().manual_seed(0)
    pix = torch.arange(N) // 8
    sites = torch.zeros(N, 10)
    sites[:, 0] = (pix % W).float() + torch.randn(N, generator=g) * 3   # some x < 0: invalid sites
    sites[:, 1] = (pix // W).float() + torch.randn(N, generator=g) * 3
    sites[:, 2] = torch.randn(N, generator=g) + 4.0
    sites[:, 3] = torch.rand(N, generator=g) * 4.0
    sites[:, 4:7] = torch.rand(N, 3, generator=g) * 1.4 - 0.2            # exercises the clamp
    d = torch.randn(N, 2, generator=g)
    sites[:, 7:9] = d / d.norm(dim=-1, keepdim=True)
    sites[:, 9] = torch.randn(N, generator=g) * 0.5
    sites = sites.to(device).contiguous()
    cand = torch.arange(N, dtype=torch.int32).view(P, 8).clone()
    cand[torch.rand(P, 8, generator=g) < 0.05] = -1                      # missing candidates
    cand0 = cand[:, :4].contiguous().to(device)
    cand1 = cand[:, 4:].contiguous().to(device)
    gt = torch.rand(H, W, 3, generator=g).to(device).contiguous()
    inv_sq = 1.0 / max(W, H) ** 2
    ref_col = vops.render_sad(cand0, cand1, sites, inv_sq, W, H)
    rendered = ref_col.clone().requires_grad_(True)
    F.mse_loss(rendered.clamp(0, 1), gt).backward()
    ref_gs = vops.render_sad_backward(cand0, cand1, sites, rendered.grad.contiguous(), inv_sq, W, H)
    gs = torch.zeros(N, 10, device=device)
    stats = torch.zeros(4, device=device)
    col = torch.zeros(H, W, 3, device=device)
    E.render_mse_backward(cand0, cand1, sites, gt, 2.0 / gt.numel(), inv_sq, W, H, gs, stats, col)
    report("render.synthetic.color", col, ref_col, bitwise=True)
    report("render.synthetic.grad_sites (single terms)", gs, ref_gs, bitwise=True)

    # The decoder's render: render_sad + clamp_(0, 1) in one kernel, per frame and batched,
    # here also with NaN colors (kept by the clamp) and a site that is never valid.
    sites_n = sites.clone()
    sites_n[::97, 4] = float("nan")
    sites_n[5::89, 0] = -1.0
    out1 = torch.empty(H, W, 3, device=device)
    outF = torch.empty(2, H, W, 3, device=device)
    refs = [vops.render_sad(cand0, cand1, s, inv_sq, W, H).clamp_(0, 1) for s in (sites, sites_n)]
    E.render_decode(cand0, cand1, sites_n, inv_sq, W, H, out1)
    E.render_decode(torch.stack([cand0, cand0]), torch.stack([cand1, cand1]), torch.stack([sites, sites_n]),
                    inv_sq, W, H, outF)
    report("render.decode (render_sad + clamp)", out1.view(torch.int32), refs[1].view(torch.int32), bitwise=True)
    report("render.decode batched (2 frames)", outF.view(torch.int32), torch.stack(refs).view(torch.int32),
           bitwise=True)
    padded = torch.cat([sites_n, torch.zeros(N, 2, device=device)], 1).contiguous()   # the [N, 12] layout
    E.render_decode(cand0, cand1, padded, inv_sq, W, H, out1)
    report("render.decode (padded sites)", out1.view(torch.int32), refs[1].view(torch.int32), bitwise=True)

    # The warp aggregation: the lanes of a warp holding the same site in a slot are summed in
    # increasing lane order (from 0) and added with one atomic per value. Every site below is
    # held by one group of lanes of one warp, in one slot, so its gradient is exactly that sum of
    # the members' single terms, given by the same pixels with one site copy per (pixel, slot).
    print("[render] synthetic image, each site shared by a group of lanes in one warp and slot")
    x, y = torch.arange(P) % W, torch.arange(P) // W
    tid = (y % 16) * 16 + x % 16                                  # 16x16 blocks, warps of 32 threads
    warp = ((y // 16) * (W // 16) + x // 16) * 8 + tid // 32
    lane = (tid % 32).unsqueeze(1).expand(P, 8)
    label = torch.randint(0, 5, (P, 8), generator=g)
    label = torch.where(torch.rand(P, 8, generator=g) < 0.3, 8 + lane, label)   # some lone sites
    key = (warp.unsqueeze(1) * 8 + torch.arange(8)) * 64 + label             # (warp, slot, group)
    _, grp = torch.unique(key.flatten(), return_inverse=True)
    G = int(grp.max()) + 1
    gsites = torch.zeros(G, 10)
    gpix = torch.zeros(G, dtype=torch.long)
    gpix[grp] = torch.arange(P * 8) // 8                          # a pixel of the group, for the position
    gsites[:, 0] = (gpix % W).float() + torch.randn(G, generator=g) * 3
    gsites[:, 1] = (gpix // W).float() + torch.randn(G, generator=g) * 3
    gsites[:, 2] = torch.randn(G, generator=g) + 4.0
    gsites[:, 3] = torch.rand(G, generator=g) * 4.0
    gsites[:, 4:7] = torch.rand(G, 3, generator=g) * 1.4 - 0.2
    d = torch.randn(G, 2, generator=g)
    gsites[:, 7:9] = d / d.norm(dim=-1, keepdim=True)
    gsites[:, 9] = torch.randn(G, generator=g) * 0.5
    missing = torch.rand(P, 8, generator=g) < 0.05
    copies = gsites[grp].contiguous()                             # one copy per (pixel, slot)
    cand_a = torch.arange(N, dtype=torch.int32).view(P, 8).clone()
    cand_g = grp.view(P, 8).to(torch.int32).clone()
    cand_a[missing] = -1
    cand_g[missing] = -1
    gs_a = torch.zeros(N, 10, device=device)
    gs_g = torch.zeros(G, 10, device=device)
    for c, s, out in ((cand_a, copies, gs_a), (cand_g, gsites, gs_g)):
        E.render_mse_backward(c[:, :4].contiguous().to(device), c[:, 4:].contiguous().to(device),
                              s.to(device).contiguous(), gt, 2.0 / gt.numel(), inv_sq, W, H, out, stats,
                              torch.empty(0, device=device))
    # Expected: per group, 0 + t_1 + t_2 + ... in float32, members in lane order; a zero sum is not added.
    terms = gs_a.cpu()
    member = torch.arange(N)[~missing.flatten()]
    order = torch.argsort(grp[member] * 32 + lane.flatten()[member])
    member = member[order]
    mg = grp[member]
    rank = torch.arange(len(member)) - torch.searchsorted(mg, mg)
    expect = torch.zeros(G, 10)
    for r in range(int(rank.max()) + 1):
        sel = member[rank == r]
        expect[grp[sel]] = expect[grp[sel]] + terms[sel]
    groups = torch.bincount(mg, minlength=G)
    print(f"  {G} sites, {int((groups > 1).sum())} held by 2-{int(groups.max())} lanes")
    report("render.synthetic.grad_sites (warp groups)", gs_g, expect.to(device), bitwise=True)


def check_vpt(device, frames=()):
    """The fused VPT pass (vpt_prep + vpt_update) against SAD's pack_candidate_sites +
    update_candidates_compact, bitwise. The inputs are synthetic buffers with invalid,
    duplicate and out-of-range candidates, inactive and non-finite sites and exact key ties,
    and `frames`, (sites, cand0, cand1, W, H) of a trained model. Every jump distance of the
    cycle and step indices past 2^16 (step_high) are covered."""
    import fast_step
    from sad_appearance_v3 import _jump_for_step
    E = fast_step.ext()
    vops = torch.ops.sad
    print("\n[vpt] fused VPT pass vs SAD's update_candidates_compact (bitwise)")

    def synthetic():
        W, H, N = 200, 120, 3001
        g = torch.Generator().manual_seed(7)
        s = torch.zeros(N, 10)
        s[:, 0] = torch.rand(N, generator=g) * W
        s[:, 1] = torch.rand(N, generator=g) * H
        s[:, 2] = torch.randn(N, generator=g) * 2 + 2
        s[:, 3] = torch.randn(N, generator=g) * 3
        s[:, 4:7] = torch.rand(N, 3, generator=g)
        d = torch.randn(N, 2, generator=g)
        s[:, 7:9] = d / d.norm(dim=-1, keepdim=True)
        s[:, 9] = torch.randn(N, generator=g) * 1.5
        s[:40, 0] = -torch.rand(40, generator=g) - 1e-3         # inactive
        s[40, 0] = -0.0
        s[41:60, 1:3] = s[60:79, 1:3]                            # partial copies
        s[80:120] = s[120:160]                                   # exact duplicates: tied keys
        s[160, 2] = float("nan")                                 # non-finite parameters
        s[161, 9] = float("inf")
        s[162, 3] = float("inf")
        s[163, 3] = -float("inf")
        s[164, 0] = float("nan")
        s[165, 9] = -float("inf")
        s[166, 2], s[167, 2], s[168, 0] = 60.0, -60.0, 70000.0   # exp / half overflow
        cand = torch.randint(0, N, (W * H, 8), generator=g, dtype=torch.int32)
        cand[torch.rand(W * H, 8, generator=g) < 0.1] = -1                      # empty slots
        cand[torch.rand(W * H, 8, generator=g) < 0.02] = N + 5                  # out of range
        dup = torch.rand(W * H, generator=g) < 0.2
        cand[dup, 1] = cand[dup, 0]                                             # duplicates
        return s.to(device), cand[:, :4].contiguous().to(device), cand[:, 4:].contiguous().to(device), W, H

    cases = [("synthetic", *synthetic())] + [(f"frame{i}", *f) for i, f in enumerate(frames)]
    hil = torch.zeros(1, dtype=torch.int32, device=device)
    bad, n = [], 0
    for name, sites, c0, c1, W, H in cases:
        N = sites.shape[0]
        inv_sq = 1.0 / float(max(W, H) ** 2)
        packed = torch.zeros(N, 8, dtype=torch.float16, device=device)
        ph = torch.zeros(N, 2, dtype=torch.int32, device=device)
        pb = torch.zeros(N, 4, device=device)
        vops.pack_candidate_sites(sites, packed, N)
        E.vpt_prep(sites, inv_sq, ph, pb)
        for step_idx in list(range(12)) + [70001, 70002]:
            for inject in (4, 0, 9):
                jump = _jump_for_step(step_idx, W, H)
                step = (int(jump) << 16) | (step_idx & 0xFFFF)
                r0, r1, f0, f1 = (torch.full_like(c0, 7) for _ in range(4))
                vops.update_candidates_compact(c0, c1, r0, r1, packed, hil, hil, inv_sq, N, step, step_idx >> 16,
                                               64.0, 0, inject, 0, 0, 1, W, H, W, H)
                E.vpt_update(c0, c1, f0, f1, ph, pb, inv_sq, step, step_idx >> 16, inject, W, H)
                n += 1
                if not (torch.equal(r0, f0) and torch.equal(r1, f1)):
                    bad.append(f"{name} step={step_idx} inject={inject}")
    print(f"  {n} passes ({len(cases)} cases x 14 step indices x 3 inject counts): "
          + ("all bitwise identical" if not bad else f"{len(bad)} differ: {bad[:5]}"))
    RESULTS["vpt.passes"] = n
    RESULTS["vpt.bitwise"] = not bad


class _CallCounter:
    """Forwards every attribute of `obj`; counts the calls of its `name` into counts[key]."""

    def __init__(self, obj, name, key, counts):
        self._obj, self._name, self._key, self._counts = obj, name, key, counts

    def __getattr__(self, n):
        f = getattr(self._obj, n)
        if n != self._name:
            return f

        def counted(*a, **kw):
            self._counts[self._key] += 1
            return f(*a, **kw)
        return counted


@contextlib.contextmanager
def counting_vpt_ops(gm, fs, counts):
    """Count the VPT updates the model's passes run: SAD's update_candidates_compact ("sad", what
    the class's _vpt_step calls) and dsad_fast's vpt_update ("fused", FusedStep._vpt_step's)."""
    mod = sys.modules[type(gm).__module__]     # the module whose `vops` the class's _vpt_step calls
    saved = mod.vops, fs.E
    mod.vops = _CallCounter(saved[0], "update_candidates_compact", "sad", counts)
    fs.E = _CallCounter(saved[1], "vpt_update", "fused", counts)
    try:
        yield counts
    finally:
        mod.vops, fs.E = saved


def check_vpt_dispatch(tr, fs, warmup_calls):
    """Which op the trainer's model runs its VPT passes with. Expected: only SAD's
    update_candidates_compact in the eager warm-up (`warmup_calls`); for one pass, SAD's op
    with the fused VPT off (the eager step) and dsad_fast's vpt_update with it on (the fused
    step). Leaves the switch as it found it. Advances one frame's candidate state by two
    passes, so it runs last."""
    gm = tr.gaussian_model
    print("\n[vpt] which op the model's VPT passes call")
    f = next(i for i in range(gm.num_frames) if gm._frame_init[i])
    was_on = "_vpt_step" in gm.__dict__
    seen = {}
    try:
        with torch.no_grad():
            gm._activate_frame(f)
            sites = gm._pack_sites_grad((gm.get_xyz + 1.0) * 0.5 * gm._wh(), gm.get_features).detach()
            for on in (False, True):
                fs.set_fused_vpt(on)
                with counting_vpt_ops(gm, fs, {"sad": 0, "fused": 0}) as c:
                    gm._vpt_step(sites, f)
                    torch.cuda.synchronize()
                seen["fused" if on else "eager"] = dict(c)
    finally:
        fs.set_fused_vpt(was_on)
    warm_ok = warmup_calls["sad"] > 0 and warmup_calls["fused"] == 0
    ok = seen["eager"] == {"sad": 1, "fused": 0} and seen["fused"] == {"sad": 0, "fused": 1}
    print(f"  eager warm-up: {warmup_calls['sad']} passes with SAD's op, {warmup_calls['fused']} with "
          f"dsad_fast's   {'as intended' if warm_ok else 'WRONG OP'}")
    print(f"  one pass, fused VPT off: {seen['eager']}   on: {seen['fused']}   "
          + ("as intended" if ok else "WRONG OP"))
    RESULTS["vpt.warmup_sad_op"] = warm_ok
    RESULTS["vpt.dispatch"] = ok and warm_ok


def check_rom_apply(device):
    """rom_apply (the decoder's cached anchor basis) against rom_forward, bitwise, on synthetic inputs."""
    import math
    import fast_step
    E = fast_step.ext()
    print("\n[romapply] rom_apply with one rom_forward's basis vs a cold rom_forward (bitwise)")
    bits = lambda t: t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t.contiguous()  # noqa: E731
    beq = lambda a, b: a.shape == b.shape and bool((bits(a) == bits(b)).all())  # noqa: E731

    def sent(t):  # a NaN bit pattern no kernel writes: every output element must be written
        (t.view(torch.int32) if t.dtype == torch.float32 else t).fill_(0x7FC0BEEF if t.dtype == torch.float32 else -12345)
        return t
    W, H, N, n, bad = 1280.0, 720.0, 20011, 0, []
    empty = torch.empty(0, device=device)
    for mode in (0, 1, 2, 3):
        for K in (8, 12, 16):
            for C in (2, 5):
                for M in (K + 1, 256):
                    if not E.rom_supported(M, K, C):
                        continue
                    g = torch.Generator().manual_seed(1000 * mode + 100 * K + 10 * C + M)
                    rn = lambda *s_, sc=1.0, mu=0.0: (torch.randn(*s_, generator=g) * sc + mu).to(device)  # noqa: E731
                    P = rn(N, 2, sc=1.2)
                    P[:64] = rn(64, 2, sc=8.0)            # tanh saturation branch
                    P[64:80] = 0.0
                    P[80:84, 0] = float("nan")            # invalid sites: phi NaN, j = 0
                    P[84:86, 1] = float("inf")
                    P[86:88, 0] = -float("inf")
                    P[88:120] = rn(32, 2, sc=0.05)        # tanh polynomial branch
                    mu = (torch.rand(M, 2, generator=g) * 2 - 1).to(device)
                    an = [mu, rn(M, sc=0.3, mu=math.log(0.25)), rn(M, sc=0.05), rn(M, sc=0.3), rn(M, 2), rn(M, sc=0.05)]
                    for t_ in an:                         # anchor 1 duplicates anchor 0: tied logits
                        t_[1] = t_[0]
                    P[120:140] = torch.atanh(mu[0].clamp(-0.999, 0.999)).expand(20, 2)  # sites on the tie
                    site = [P, rn(N, 3, sc=0.3), rn(N), rn(N, sc=0.1), rn(N, 2), rn(N, sc=0.2)]
                    tables = [rn(M, C, sc=0.1) for _ in range(3)] + [torch.zeros(M, C, device=device), rn(M, C, sc=20.0)]
                    z = lambda *s_, **kw: torch.zeros(*s_, device=device, **kw)  # noqa: E731

                    def fwd(table, warm=None):
                        o = dict(d=sent(z(N, C)), phi=sent(z(N, K)), tab=sent(z(M, 8)), sites=sent(z(N, 10)),
                                 y=sent(z(N, 2)), topi=z(N, K, dtype=torch.int32) if warm is None else warm.clone())
                        E.rom_forward(P, mode, K, *an, table, warm is not None, o["d"], o["topi"], o["phi"], o["tab"],
                                      True, site, W, H, o["sites"], o["y"], empty, empty)
                        return o
                    with torch.no_grad():
                        base = fwd(tables[0])
                        for ti, tb in enumerate(tables):
                            ref, refw = fwd(tb), fwd(tb, warm=base["topi"])
                            d, sites, y = sent(z(N, C)), sent(z(N, 10)), sent(z(N, 2))
                            E.rom_apply(P, K, tb, base["topi"], base["phi"], d, True, site, W, H, sites, y)
                            ok = (beq(d, ref["d"]) and beq(sites, ref["sites"]) and beq(y, ref["y"])
                                  and beq(base["topi"], ref["topi"]) and beq(base["phi"], ref["phi"])
                                  and all(beq(refw[k], ref[k]) for k in ref))
                            if not ok:
                                bad.append(f"mode={mode} K={K} C={C} M={M} table={ti}")
                    n += 1
    print(f"  {n} (mode, K, C, M) combinations x 5 coefficient tables: "
          + ("all bitwise identical" if not bad else f"{len(bad)} differ: {bad[:5]}"))
    RESULTS["romapply.combos"] = n
    RESULTS["romapply.bitwise"] = not bad


def check_optimizers(device):
    print(f"\n[optim] fused Adan / Adam vs eager, {OPT_STEPS} steps each")
    sys.path.insert(0, os.path.join(ROOT, "baselines", "D2GV"))
    from optimizer import Adan
    import fast_step
    torch.manual_seed(0)
    shapes = [(22500, 2), (22500, 3), (22500,), (22500,), (22500, 2), (22500,)]
    p_e = [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]
    p_f = [torch.nn.Parameter(p.detach().clone()) for p in p_e]
    opt_e = Adan(p_e, lr=5e-3)
    opt_f = Adan(p_f, lr=5e-3)

    class Shim:  # the parts of FusedStep that _adan_step/_adam_step use
        E = fast_step.ext()
        _grad_of = {}
        _adam_checked = set()
    shim = Shim()
    ok = True
    for step in range(OPT_STEPS):
        if step % 20 == 19:
            for g in opt_e.param_groups + opt_f.param_groups:
                g["lr"] *= 0.5
        grads = [torch.randn(s, device=device) * (10.0 ** (step % 5 - 2)) for s in shapes]
        for p, g in zip(p_e, grads):
            p.grad = g.clone()
        opt_e.step()
        shim._grad_of = {p: g.clone() for p, g in zip(p_f, grads)}
        fast_step.FusedStep._adan_step(shim, opt_f)
        for i, (a, b) in enumerate(zip(p_f, p_e)):
            ok &= torch.equal(a, b)
            for k in ("exp_avg", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad"):
                ok &= torch.equal(opt_f.state[a][k], opt_e.state[b][k])
    print(f"  Adan params+states bitwise over {OPT_STEPS} steps: {ok}")
    RESULTS["optim.adan_bitwise"] = bool(ok)

    shapes = [(256, 2), (256,), (256,), (256,), (256, 2), (256, 16), (256, 16), (256, 16, 2)]
    q_e = [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]
    q_f = [torch.nn.Parameter(p.detach().clone()) for p in q_e]
    groups_e = [{"params": q_e[:5], "lr": 5e-3, "name": "anchors"},
                {"params": q_e[5:], "lr": 2e-3, "name": "coeffs"}]
    groups_f = [{"params": q_f[:5], "lr": 5e-3, "name": "anchors"},
                {"params": q_f[5:], "lr": 2e-3, "name": "coeffs"}]
    oe = torch.optim.Adam(groups_e, lr=0.0, eps=1e-15)
    of = torch.optim.Adam(groups_f, lr=0.0, eps=1e-15)
    ok = True
    for step in range(OPT_STEPS):
        for ge, gf in zip(oe.param_groups, of.param_groups):
            ge["lr"] = gf["lr"] = ge["lr"] * 0.97
        grads = [torch.randn(s, device=device) * (10.0 ** (step % 5 - 2)) for s in shapes]
        for p, g in zip(q_e, grads):
            p.grad = g.clone()
        oe.step()
        shim._grad_of = {p: g.clone() for p, g in zip(q_f, grads)}
        fast_step.FusedStep._adam_step(shim, of)
        for a, b in zip(q_f, q_e):
            ok &= torch.equal(a, b)
            for k in ("exp_avg", "exp_avg_sq"):
                ok &= torch.equal(of.state[a][k], oe.state[b][k])
            ok &= torch.equal(of.state[a]["step"], oe.state[b]["step"])
    print(f"  Adam params+states bitwise over {OPT_STEPS} steps: {ok}")
    RESULTS["optim.adam_bitwise"] = bool(ok)


def failed_checks() -> list[str]:
    """The names of the bitwise checks recorded in RESULTS that did not hold."""
    bad = []
    for k, v in RESULTS.items():
        if isinstance(v, dict) and (v.get("bitwise") is False or v.get("identical") is False):
            bad.append(k)
        elif k.endswith((".bitwise", "_bitwise", "_identical", ".dispatch")) and v is False:
            bad.append(k)
    return bad


QUICK = {"vpt": lambda dev: check_vpt(dev), "render": lambda dev: check_render_terms(dev),
         "romapply": lambda dev: check_rom_apply(dev), "optim": lambda dev: check_optimizers(dev)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", default=None, help="a GoP's frame directory, for the trained state")
    ap.add_argument("--num_points", type=int, default=None, help="sites of that state (e.g. 27047 for Bunny)")
    ap.add_argument("--only", default="",
                    help="run only these synthetic checks, comma-separated (vpt, render, romapply, "
                         "optim): no dataset, exit status 1 on a failure")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "dsad.yaml"))
    ap.add_argument("--warm", type=int, default=300, help="eager iterations per stage first")
    ap.add_argument("--output_dir", default="checkpoints",
                    help="where the warm-up run's log goes (data_name check_fast_ops)")
    ap.add_argument("--out", default="", help="write the comparison results here (JSON)")
    ap.add_argument("--exact_cands", action="store_true",
                    help="compare with the exact top-8 builder's candidates (the --cand_exact_tail "
                         "path), check builds from the fused and the eager sites against each other "
                         "and gm._candidates in exact mode against the builder; the fine warm-up "
                         "then trains in exact mode, as the tail does")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in QUICK]
        if unknown:
            ap.error(f"--only: unknown check(s) {unknown}; choose from {sorted(QUICK)}")
        dev = torch.device("cuda:0")
        import sad_appearance_v3  # noqa: F401  (registers torch.ops.sad)
        for n in names:
            QUICK[n](dev)
        bad = failed_checks()
        print("\n" + ("FAILED: " + ", ".join(bad) if bad else f"all passed ({', '.join(names)})"))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(RESULTS, fh, indent=1)
        raise SystemExit(1 if bad else 0)
    if args.dataset is None or args.num_points is None:
        ap.error("-d/--dataset and --num_points are required (or --only for the synthetic checks)")

    import train_sad_video as tsv
    import fast_step
    # --cand_exact_tail 0: this tool switches to the exact candidates itself (--exact_cands);
    # the config's tail, longer than the short warm-up, would never start and only warn.
    targs = tsv.parse_args(
        ["--config", args.config, "-d", args.dataset, "--num_points", str(args.num_points),
         "--data_name", "check_fast_ops", "--output_dir", args.output_dir,
         "--iterations", str(args.warm), "--coarse_iterations", str(args.warm),
         "--cand_exact_tail", "0", "--diag_every", "0", "--fast"] + extra)
    torch.manual_seed(targs.seed)
    torch.cuda.manual_seed(targs.seed)
    tr = tsv.SimpleTrainer(image_path=Path(targs.dataset), num_points=targs.num_points,
                           iterations=targs.iterations, args=targs)
    fs = tr._fast
    assert fs is not None, "fused step unavailable"
    tr._fast = None  # warm up with the eager step
    fs.set_fused_vpt(False)  # ... whose VPT passes are SAD's op (check_vpt_dispatch checks both)
    tr.test = lambda log_per_frame=False: (0.0, 0.0)
    tr.test_fps = lambda: 0.0
    gm = tr.gaussian_model
    E = fast_step.ext()
    dev = gm._xyz.device
    vops = torch.ops.sad
    W, H = gm.W, gm.H
    T = len(tr.original_image_list)

    # ------------------------------------------------------------------ coarse
    vpt_calls = {"sad": 0, "fused": 0}       # the warm-up's VPT passes, by op (check_vpt_dispatch)
    with counting_vpt_ops(gm, fs, vpt_calls):
        tr.train("coarse")
    f = 3
    ts = (f + 1) / T
    gt = tr._gt_hwc(f)
    gm._activate_frame(f)
    cands = tuple(c.clone() for c in gm.cand_updater.get())
    print("\n[coarse] after", args.warm, "eager iterations, frame", f)
    params = [getattr(gm, n) for n in site_param_names()]
    pos_px = (gm.get_xyz + 1.0) * 0.5 * gm._wh()
    sites_e = gm._pack_sites_grad(pos_px, gm.get_features)
    if args.exact_cands:
        cands = exact_cands(gm, sites_e)
    rendered_e = gm.renderer(sites_e, cands[0], cands[1], W, H)
    rec = F.mse_loss(rendered_e.clamp(0, 1), gt)
    (g_r,) = torch.autograd.grad(rec, rendered_e, retain_graph=True)
    inv_sq = float(gm.inv_sq)
    gs_e = vops.render_sad_backward(cands[0], cands[1], sites_e.detach().contiguous(),
                                    g_r.contiguous(), inv_sq, W, H)
    g_e = torch.autograd.grad(sites_e, params, grad_outputs=gs_e)
    fs.forward_backward("coarse", 1, f, ts, gt, cands=cands, gs=gs_e)
    report("coarse.sites", fs.sites, sites_e, bitwise=True)
    if args.exact_cands:
        report_cands("coarse.exact(fused) vs exact(eager)", exact_cands(gm, fs.sites), cands)
    buf = torch.zeros_like(g_r)
    stats = torch.zeros(4, device=dev)
    E.mse_fb(rendered_e.detach(), gt, fs.mse_alpha, buf, stats)
    report("coarse.dL/drender", buf, g_r, bitwise=True)
    report("coarse.rec_loss", stats[0:1] / gt.numel(), rec.detach().reshape(1))
    for n, a, b in zip(site_param_names(), fs.site_grads, g_e):
        report(f"coarse.grad.{n}", a, b, bitwise=True)
    # atomics noise of the eager backward itself
    gs_e2 = vops.render_sad_backward(cands[0], cands[1], sites_e.detach().contiguous(),
                                     g_r.contiguous(), inv_sq, W, H)
    report("noise.coarse.dL/dsites (2 eager runs)", gs_e2, gs_e)
    # fused render + MSE + backward on the same state
    gs_f = torch.zeros_like(gs_e)
    col = torch.zeros_like(gt)
    st_f = torch.zeros(4, device=dev)
    E.render_mse_backward(cands[0], cands[1], sites_e.detach().contiguous(), gt, fs.mse_alpha,
                          inv_sq, W, H, gs_f, st_f, col)
    report("coarse.render.color (fused vs render_sad)", col, rendered_e, bitwise=True)
    report("coarse.render.dL/dsites (fused vs SAD)", gs_f, gs_e)
    report("coarse.render.rec_loss (fused vs eager)", st_f[0:1] / gt.numel(), rec.detach().reshape(1))
    if args.exact_cands:
        # The trainer's tail switch; the fine warm-up below then runs in exact mode.
        exact_mode_path(gm, sites_e, f, cands)

    # ------------------------------------------------------------------ fine
    with counting_vpt_ops(gm, fs, vpt_calls):
        tr.train("fine")
    if tr.knn_idx is None:
        import losses_motion
        tr.knn_idx = losses_motion.build_knn_graph(gm.get_xyz.detach(), k=6)
    print("\n[fine] after", args.warm, "eager iterations, frame", f)
    gm._activate_frame(f)
    cands = tuple(c.clone() for c in gm.cand_updater.get())
    P = all_params(tr)
    names = list(P)
    for p in P.values():
        p.grad = None
    if args.exact_cands:
        cands = exact_cands(gm, eager_fine_graph(tr, f, ts, gt, cands)["sites"])
    G = eager_fine_graph(tr, f, ts, gt, cands)
    (g_r,) = torch.autograd.grad(G["rec"], G["rendered"], retain_graph=True)
    gs_e = vops.render_sad_backward(cands[0], cands[1], G["sites"].detach().contiguous(),
                                    g_r.contiguous(), inv_sq, W, H)
    surrogate = (G["sites"] * gs_e).sum() + G["reg"]
    g_e = dict(zip(names, torch.autograd.grad(surrogate, list(P.values()), retain_graph=True)))
    fs.stats.zero_()
    fs.forward_backward("fine", 7, f, ts, gt, cands=cands, gs=gs_e)
    if args.exact_cands:
        # Fine-stage sites differ from the eager ones in the last bits (sums over the K anchors).
        report_cands("fine.exact(fused) vs exact(eager)", exact_cands(gm, fs.sites), cands)
    # Logged loss terms (sums in fs.stats) against the eager values.
    import losses_motion as _lm
    with torch.no_grad():
        n_smooth = 30 * fs.M * 2
        report("fine.loss.move", fs.stats[1:2] / fs.N,
               _lm.movement_penalty(G["d_xyz"]).reshape(1))
        report("fine.loss.temporal", fs.stats[2:3] / n_smooth,
               _lm.temporal_smoothness(tr.deform.deform.coef_xyz).reshape(1))
        report("fine.loss.arap", fs.stats[3:4] / (fs.N * tr.knn_idx.shape[1]),
               _lm.arap(gm.get_xyz.detach(), torch.tanh(gm._xyz + G["d_xyz"]), tr.knn_idx).reshape(1))
    skipped = [n for k, (n, lam) in enumerate((("move", tr.lambda_move), ("temporal", tr.lambda_temporal),
                                                ("arap", tr.lambda_arap)), 1) if lam == 0 and float(fs.stats[k]) == 0.0]
    if skipped:
        print(f"  ({', '.join('fine.loss.' + n for n in skipped)}: weight 0, so the fused step skips the term and "
              "rel = 1.00. Pass a weight to compare it, e.g. -- --lambda_arap 1e-3.)")
    # The top-K scan is warm-started from the buffer's previous contents; that
    # must not change any output. Rerun warm, then with a random warm buffer.
    keep = ("topi", "phi", "d", "sites", "y")
    ref_out = {k: getattr(fs, k).clone() for k in keep}
    ref_g = [g.clone() for g in fs.site_grads]
    fs.forward_backward("fine", 7, f, ts, gt, cands=cands, gs=gs_e)
    same_prev = all(torch.equal(getattr(fs, k), ref_out[k]) for k in keep)
    fs.topi.copy_(torch.randint(-3, fs.M + 3, tuple(fs.topi.shape), device=dev, dtype=torch.int32))
    fs.forward_backward("fine", 7, f, ts, gt, cands=cands, gs=gs_e)
    same_rand = all(torch.equal(getattr(fs, k), ref_out[k]) for k in keep)
    # (position grads include ARAP atomics, so compare the other site grads)
    same_grad = all(torch.equal(a, b) for a, b in zip(fs.site_grads[1:], ref_g[1:]))
    print(f"  warm-started top-K: forward bitwise-identical to the first run with the previous "
          f"selection {same_prev}, with a random warm buffer {same_rand}; site grads {same_grad}")
    RESULTS["fine.warm_start_identical"] = bool(same_prev and same_rand)

    core = tr.deform.deform
    with torch.no_grad():
        logits = eager_dense_logits(core.anchors, gm.get_xyz)
        topv, topi = logits.topk(core.anchors.top_k, dim=-1)
        same = (torch.sort(topi, dim=-1).values == torch.sort(fs.topi.long(), dim=-1).values).all(dim=-1)
        n_bad = (~same).sum().item()
        print(f"  top-K anchor sets identical on {same.numel() - n_bad}/{same.numel()} sites")
        RESULTS["fine.topk_sets_identical"] = n_bad == 0
        phi_dense = core.anchors.basis(gm.get_xyz)
        report("fine.phi (sparse vs dense)", fs.phi, torch.gather(phi_dense, 1, fs.topi.long()))
        report("fine.d_xyz", fs.d[:, :2], G["d_xyz"])
        if fs.C >= 5:
            report("fine.d_feat", fs.d[:, 2:5], G["d_feat"])
        report("fine.sites", fs.sites, G["sites"])
    fg = fused_grads(fs)
    for n in names:
        report(f"fine.grad.{n}", fg[n], g_e[n])

    # Accuracy against a float64 evaluation of the same gradients (given the
    # same dL/dsites): the eager float32 and fused float32 errors side by side.
    print("\n  float64 reference, error = max|g32 - g64| / max|g64|:")
    import losses_motion
    with torch.enable_grad():
        core64 = copy.deepcopy(core).double()
        P64 = {k: getattr(gm, n).detach().double().requires_grad_(True)
               for k, n in zip(site_param_names(), site_param_names())}
        x64 = torch.tanh(P64["_xyz"])
        tsd = torch.tensor(float(torch.tensor(ts, dtype=torch.float32)), dtype=torch.float64, device=dev)
        phi64 = core64.anchors.basis(x64)
        d_xyz64 = phi64 @ core64.coef_xyz(tsd)
        d_feat64 = phi64 @ core64.coef_features(tsd)
        y64 = torch.tanh(P64["_xyz"] + d_xyz64)
        wh64 = gm._wh().double()
        pos64 = (y64 + 1.0) * 0.5 * wh64
        a = P64["_aniso_dir"]
        u64 = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        f64 = P64["_features_dc"] + d_feat64
        sites64 = torch.stack([pos64[:, 0], pos64[:, 1], P64["_log_tau"], P64["_radius"],
                               f64[:, 0], f64[:, 1], f64[:, 2], u64[:, 0], u64[:, 1],
                               P64["_log_aniso"]], dim=-1)
        reg64 = (tr.lambda_move * losses_motion.movement_penalty(d_xyz64)
                 + tr.lambda_temporal * losses_motion.temporal_smoothness(core64.coef_xyz)
                 + tr.lambda_arap * losses_motion.arap(x64.detach(), y64, tr.knn_idx))
        L64 = (sites64 * gs_e.double()).sum() + reg64
        names64 = [f"site.{n}" for n in site_param_names()]
        params64 = list(P64.values())
        a64 = core64.anchors
        for n in ["mu", "log_sigma", "radius", "log_aspect", "aniso_dir", "weight"]:
            if hasattr(a64, n):
                names64.append(f"anchor.{n}")
                params64.append(getattr(a64, n))
        for sn, s in [("coef_xyz", core64.coef_xyz), ("coef_features", core64.coef_features)]:
            for n in ["t", "log_r", "v"]:
                names64.append(f"{sn}.{n}")
                params64.append(getattr(s, n))
        g64 = dict(zip(names64, torch.autograd.grad(L64, params64)))
    for n in names:
        ref = g64[n]
        sc = ref.abs().max().item()
        e_eager = (g_e[n].double() - ref).abs().max().item() / sc if sc else 0.0
        e_fused = (fg[n].double() - ref).abs().max().item() / sc if sc else 0.0
        print(f"    {n:<28s} eager {e_eager:.2e}   fused {e_fused:.2e}")
        RESULTS[f"f64.{n}"] = {"eager": e_eager, "fused": e_fused}

    # Eager noise: the same gradients with a second render backward.
    gs_e2 = vops.render_sad_backward(cands[0], cands[1], G["sites"].detach().contiguous(),
                                     g_r.contiguous(), inv_sq, W, H)
    surrogate2 = (G["sites"] * gs_e2).sum() + G["reg"]
    g_e2 = dict(zip(names, torch.autograd.grad(surrogate2, list(P.values()))))
    for n in names:
        report(f"noise.fine.grad.{n}", g_e2[n], g_e[n])

    # ------------------------------------------------------------------ decode
    print("\n[decode] fused ROM forward vs DeformModelSAD.step at every frame time")
    with torch.no_grad():
        for i in range(T):
            tsi = (i + 1) / T
            d_xyz, _, d_feat = tr.deform.step(gm.get_xyz, tsi)
            E.temporal_forward(float(torch.tensor(tsi, dtype=torch.float32)), fs.stack_params[0],
                               fs.stack_params[1] if len(fs.stacks) > 1 else [], fs.lin, False,
                               fs.coef_table, fs.stats)
            E.rom_forward(gm._xyz, fs.mode, fs.K, *fs.anchor_params, fs.coef_table, False, fs.d, fs.topi,
                          fs.phi, fs.tab, False, [], 0.0, 0.0, torch.empty(0, device=dev),
                          torch.empty(0, device=dev), torch.empty(0, device=dev),
                          torch.empty(0, device=dev))
            dd = torch.cat([d_xyz, d_feat], 1) if fs.C >= 5 else d_xyz
            report(f"decode.d(t={tsi:.2f})", fs.d, dd)

    # Full decode (ROM + packing + render) vs the eager forward_dynamic, and timing.
    print("\n[decode] FusedStep.decode vs eager deform.step + forward_dynamic")
    import time as _time
    import math as _math
    with torch.no_grad():
        worst = 0.0
        for i in range(T):
            tsi = (i + 1) / T
            d_xyz, d_op, d_feat = tr.deform.step(gm.get_xyz, tsi)
            ref = gm.forward_dynamic(d_xyz, d_op, d_feat, frame_idx=i)["render"]
            out = fs.decode(i, tsi).permute(2, 0, 1).unsqueeze(0)
            img = tr.original_image_list[i]
            p_ref = 10 * _math.log10(1.0 / F.mse_loss(ref, img).item())
            p_out = 10 * _math.log10(1.0 / F.mse_loss(out, img).item())
            worst = max(worst, abs(p_ref - p_out))
            if i == 0:
                report("decode.frame0 image", out, ref)
                # the decoder's render against render_sad + clamp on its own sites and candidates
                sites_i = fs._dec["sites"]
                c0, c1 = gm.eval_candidates(sites_i, i)
                own = vops.render_sad(c0, c1, sites_i, inv_sq, W, H).clamp_(0, 1)
                report("decode.frame0 render (render_sad + clamp)", out.squeeze(0).permute(1, 2, 0).contiguous(),
                       own, bitwise=True)
        print(f"  max |PSNR(fused) - PSNR(eager)| over frames: {worst:.2e} dB")
        RESULTS["decode.max_psnr_diff_db"] = worst
        # A GoP pass at once (decode.py's fused decoder) against decode() frame by frame, in both
        # candidate modes: the training buffers, and the decoder's exact build (batched).
        mode0 = gm.cand_method
        for mode in ("vpt", "exact"):
            gm.cand_method = mode
            fs.invalidate_basis()
            per = torch.stack([fs.decode(i, (i + 1) / T) for i in range(T)])
            fs.invalidate_basis()
            cands = None if mode == "exact" else [gm.eval_candidates(None, i) for i in range(T)]
            gop = fs.decode_gop(T, cands=cands)
            report(f"decode.gop pass ({mode} candidates) vs per frame", gop.view(torch.int32),
                   per.view(torch.int32), bitwise=True)
        gm.cand_method = mode0

        def bench(fn, reps=5):
            fn()
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            return reps * T / (_time.perf_counter() - t0)

        def eager_all():
            for i in range(T):
                d_xyz, d_op, d_feat = tr.deform.step(gm.get_xyz, (i + 1) / T)
                gm.forward_dynamic(d_xyz, d_op, d_feat, frame_idx=i)

        def fused_all():
            for i in range(T):
                fs.decode(i, (i + 1) / T)
        fps_e, fps_f = bench(eager_all), bench(fused_all)
        print(f"  steady-state decode: eager {fps_e:.1f} FPS, fused {fps_f:.1f} FPS ({fps_f / fps_e:.2f}x)")
        RESULTS["decode.fps_eager"] = fps_e
        RESULTS["decode.fps_fused"] = fps_f

    # The VPT pass on every frame of the trained model: its sites at the frame time, its buffers.
    frames = []
    with torch.no_grad():
        for i in range(T):
            if not gm._frame_init[i]:
                continue
            gm._activate_frame(i)
            d_xyz, _, d_feat = tr.deform.step(gm.get_xyz, (i + 1) / T)
            pos = (torch.tanh(gm._xyz + d_xyz) + 1.0) * 0.5 * gm._wh()
            sites_i = gm._pack_sites_grad(pos, gm._features_dc + d_feat).detach()
            frames.append((sites_i, *(c.clone() for c in gm.cand_updater.get()), W, H))
    check_vpt(dev, frames)
    check_vpt_dispatch(tr, fs, vpt_calls)
    check_rom_apply(dev)
    check_render_terms(dev)
    check_optimizers(dev)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(RESULTS, fh, indent=1)
    bad = failed_checks()
    n = sum((isinstance(v, dict) and ("bitwise" in v or "identical" in v))                # as failed_checks() counts
            or (isinstance(v, bool) and k.endswith((".bitwise", "_bitwise", "_identical", ".dispatch")))
            for k, v in RESULTS.items())
    print("\nverdict: " + (f"FAILED bitwise checks: {', '.join(bad)}" if bad else
                           f"all {n} bitwise checks passed"))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
