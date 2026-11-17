"""Fused CUDA training step and decoder (--fast).

FusedStep.step runs one iteration of the eager step's model and loss in about
10 kernel launches (dsad/csrc/dsad_fast.cu), with hand-written gradients:

    temporal_forward      a_j(t) of every coefficient stack (+ temporal smoothness)
    rom_forward           x = tanh(z), top-K anchors, softmax, d = Σ φ a, packed sites
                          (+ movement penalty)
    arap_fb               ARAP loss and its gradient
    (candidates)          SAD's JFA and VPT ops; with fused_vpt, vpt_prep + vpt_update
    render_mse_backward   render, clamp + MSE and render backward in one kernel
    rom_backward          site and ROM gradients
    anchor_backward       anchor and knot gradients
    adan_step, adam_step  the two optimizer updates
    stats_step            loss values for logging, on the GPU

FusedStep.decode is the matching decoder. The anchor weights do not depend on
t, so it runs rom_forward once and then only the product d = Σ φ a(t) and the
packing, until the parameters change (step() or invalidate_basis()).

Covered: SAD appearance, every anchor mode, anchor top-K 8, 12 or 16, at most
1,024 anchors and 32 knots, no temporal top-K, lambda_spatial = 0.
unsupported_reason() tells the trainer when to fall back to the eager step.
The header of dsad_fast.cu states which results are bit-identical to the eager
step, and tools/check_fast_ops.py checks them.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

try:
    from . import _paths
except ImportError:
    import _paths

_EXT = None
MODES = {"voronoi": 0, "apollonius": 1, "power": 2, "apollonius_aniso": 3}


def ext():
    """Build (once, cached by torch under TORCH_EXTENSIONS_DIR) and load the kernels."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        # Repository-local extension cache, and the interpreter's ninja on PATH.
        _paths.setup_env()
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "dsad_fast.cu")
        # Build for the architecture of the current GPU. DSAD_FAST_ARCH, e.g. "7.5",
        # overrides it, which also allows building without a visible GPU.
        arch = os.environ.get("DSAD_FAST_ARCH")
        if not arch:
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}.{minor}"
        sm = arch.replace(".", "")
        _EXT = load(name="dsad_fast", sources=[src],
                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{sm},code=sm_{sm}"],
                    verbose=False)
    return _EXT


def _f32(x: float) -> float:
    """Round a Python float to float32 (what a CUDA kernel receives as a scalar)."""
    return float(np.float32(x))


def _mean_scale(lam: float, count: int) -> float:
    """float32 value of `lam * (1 / count)`: MeanBackward divides the incoming
    float32 gradient `lam` by `count`, which torch does as a multiply by the
    float32 reciprocal."""
    return float(np.float32(lam) * (np.float32(1.0) / np.float32(count)))


def unsupported_reason(trainer) -> str | None:
    """Why the fused step cannot reproduce this trainer's eager step (None if it can)."""
    args = trainer.args
    if not trainer._is_sad_appearance:
        return "--appearance gaussian"
    if trainer.lambda_spatial > 0:
        return "--lambda_spatial > 0"
    core = trainer.deform.deform
    a = core.anchors
    stacks = [core.coef_xyz] + ([core.coef_features] if core.enable_features else [])
    if any(s.top_k is not None and s.top_k < s.K for s in stacks):
        return "temporal top-K"
    if any(s.K > 32 for s in stacks):
        return "more than 32 knots"
    C = sum(s.out_dim for s in stacks)
    if a.M > 1024:
        return f"more than 1,024 anchors (M={a.M})"
    if a.top_k is None or not ext().rom_supported(a.M, a.top_k, C):
        return (f"anchor top-K {a.top_k} with M={a.M} (supported: top-K 8, 12 or 16, below M, "
                f"M <= 1,024, at most 8 coefficient columns)")
    gopt = trainer.gaussian_model.optimizer
    if type(gopt).__name__ == "Adan":
        d = gopt.defaults
        if d["weight_decay"] != 0 or d["max_grad_norm"] != 0 or d["fused"] or not d["foreach"]:
            return "Adan settings"
    elif isinstance(gopt, torch.optim.Adam):
        if not _adam_ok(gopt):
            return "site Adam settings"
    else:
        return f"site optimizer {type(gopt).__name__}"
    if not _adam_ok(trainer.deform.optimizer):
        return "ROM Adam settings"
    return None


def _adam_ok(opt) -> bool:
    for g in opt.param_groups:
        if (g.get("amsgrad") or g.get("weight_decay", 0) != 0 or g.get("maximize")
                or g.get("capturable") or g.get("differentiable") or g.get("fused")
                or g.get("foreach") is False or g.get("decoupled_weight_decay")):
            return False
    return isinstance(opt, torch.optim.Adam)


class FusedStep:
    """Training step and decoder of a SimpleTrainer on the fused kernels."""

    def __init__(self, trainer, fused_vpt: bool = False):
        """fused_vpt: run the model's VPT candidate passes with the fused kernel (see
        set_fused_vpt), as the trainer does with --fast. Off, the model keeps SAD's op."""
        self.tr = trainer
        self.E = ext()
        gm = trainer.gaussian_model
        core = trainer.deform.deform
        dev = gm._xyz.device
        self.dev = dev
        N, H, W = gm.num_points, gm.H, gm.W
        self.N, self.H, self.W = N, H, W
        self.site_params = [gm._xyz, gm._features_dc, gm._log_tau, gm._radius,
                            gm._aniso_dir, gm._log_aniso]
        self.site_grads = [torch.zeros_like(p) for p in self.site_params]
        self.sites = torch.zeros(N, 10, device=dev)
        self.y = torch.zeros(N, 2, device=dev)
        self.gy_arap = torch.zeros(N, 2, device=dev)
        self.grad_render = torch.zeros(H, W, 3, device=dev)
        self.gs = torch.zeros(N, 10, device=dev)       # dL/dsites
        self._empty = torch.empty(0, device=dev)
        self.mse_alpha = 2.0 / (H * W * 3)            # MseLossBackward, mean reduction
        # One kernel for render + MSE + render backward (default), or SAD's
        # render_sad / render_sad_backward ops (--fast_sad_render).
        self.fused_render = not getattr(trainer.args, "fast_sad_render", False)

        # ---- ROM ----
        a = core.anchors
        self.mode = MODES[a.mode]
        self.K = a.top_k
        self.M = a.M
        self.stacks = [core.coef_xyz] + ([core.coef_features] if core.enable_features else [])
        self.C = sum(s.out_dim for s in self.stacks)
        empty = torch.empty(0, device=dev)
        self.anchor_params = [
            a.mu, a.log_sigma,
            getattr(a, "radius", empty),
            getattr(a, "log_aspect", empty),
            getattr(a, "aniso_dir", empty),
            getattr(a, "weight", empty),
        ]
        self.anchor_grads = [torch.zeros_like(p) for p in self.anchor_params]
        self.stack_params = [[s.t, s.log_r, s.v] for s in self.stacks]
        self.stack_grads = [[torch.zeros_like(p) for p in sp] for sp in self.stack_params]
        self.coef_table = torch.zeros(self.M, self.C, device=dev)
        self.d = torch.zeros(N, self.C, device=dev)
        self.topi = torch.zeros(N, self.K, dtype=torch.int32, device=dev)
        self.phi = torch.zeros(N, self.K, device=dev)
        self.tab = torch.zeros(self.M, 8, device=dev)
        self.gacc = torch.zeros(self.M, 8, device=dev)
        self.gcoef = torch.zeros(self.M, self.C, device=dev)
        # Same grid as losses_motion.temporal_smoothness.
        self.lin = torch.linspace(0.0, 1.0, 32, device=dev)

        # Map every parameter to its gradient buffer (for the optimizers).
        self._grad_of = {}
        for p, g in zip(self.site_params, self.site_grads):
            self._grad_of[p] = g
        for p, g in zip(self.anchor_params, self.anchor_grads):
            if p.numel():
                self._grad_of[p] = g
        for ps, gs in zip(self.stack_params, self.stack_grads):
            for p, g in zip(ps, gs):
                self._grad_of[p] = g

        # ---- loss bookkeeping (stays on the GPU) ----
        lam_m, lam_t, lam_a = trainer.lambda_move, trainer.lambda_temporal, trainer.lambda_arap
        n_smooth = 30 * self.M * 2   # (32 - 2) second differences of a [32, M, 2] curve
        self.stats = torch.zeros(4, device=dev)         # sums: rec, move, temporal, arap
        self.scale = torch.tensor([1.0 / (H * W * 3), 1.0 / N, 1.0 / n_smooth, 1.0 / (N * 6)],
                                  device=dev)
        self.lam = torch.tensor([1.0, lam_m, lam_t, lam_a], device=dev)
        self.cur = torch.zeros(5, device=dev)            # rec, move, temporal, arap, total
        self.running = torch.zeros(5, dtype=torch.float64, device=dev)
        # Gradient scales, as autograd forms them in float32:
        #   move:     MeanBackward  lam / N         (then x2 via d*d)
        #   temporal: MeanBackward  lam / n_smooth  (then x2 via d2*d2)
        #   arap:     MseLossBackward (2 / (N k)) * lam
        # (tensor / scalar multiplies by the float32 reciprocal).
        self.move_coef = _mean_scale(lam_m, N) if lam_m > 0 else 0.0
        self.smooth_coef = _mean_scale(lam_t, n_smooth) if lam_t > 0 else 0.0
        self.lam_arap = lam_a
        self._adam_checked = set()
        self._dec = None
        self._dec_basis = False   # decode(): (topi, phi) valid for the current parameters
        self._gop = None          # decode_gop(): the GoP's tables and sites

        self.vpt_ph = self.vpt_pb = None
        if fused_vpt:              # off: leave the model's VPT op as it is (SAD's, unless set)
            self.set_fused_vpt(True)

    def step(self, stage: str, it: int, frame_idx: int, timestamp: float, gt_hwc: torch.Tensor):
        """One training iteration: forward, backward and both optimizer updates."""
        self._dec_basis = False   # the parameters change below
        self.forward_backward(stage, it, frame_idx, timestamp, gt_hwc)
        self._site_optimizer_step(self.tr.gaussian_model.optimizer)
        if stage == "fine":
            self._adam_step(self.tr.deform.optimizer)
        self.E.stats_step(self.stats, self.scale, self.lam, self.cur, self.running)

    def forward_backward(self, stage, it, frame_idx, timestamp, gt_hwc, cands=None, gs=None):
        """Loss and gradients into the gradient buffers. `cands` (fixed candidate
        buffers, skipping the candidate update) and `gs` (dL/dsites instead of the
        render backward) are used by tools/check_fast_ops.py."""
        E = self.E
        gm = self.tr.gaussian_model
        vops = torch.ops.sad
        fine = stage == "fine"
        self.rendered = None
        if fine:
            t = float(np.float32(timestamp))
            self.t = t
            E.temporal_forward(t, self.stack_params[0],
                               self.stack_params[1] if len(self.stacks) > 1 else [],
                               self.lin, self.smooth_coef != 0.0, self.coef_table, self.stats)
            E.rom_forward(gm._xyz, self.mode, self.K, *self.anchor_params, self.coef_table, True,
                          self.d, self.topi, self.phi, self.tab, True, self.site_params,
                          float(self.W), float(self.H), self.sites, self.y, self.gy_arap,
                          self.stats)
            if self.lam_arap > 0:
                tr = self.tr
                if (tr.knn_idx is None
                        or (tr.knn_rebuild_every and it % tr.knn_rebuild_every == 0)):
                    import losses_motion
                    tr.knn_idx = losses_motion.build_knn_graph(gm.get_xyz.detach(), k=6)
                kn = tr.knn_idx.shape[1]
                coef = _f32(2.0 / (self.N * kn)) * _f32(self.lam_arap)
                E.arap_fb(gm._xyz, self.y, tr.knn_idx.contiguous(), coef, self.gy_arap, self.stats)
        else:
            E.sites_forward(self.site_params, float(self.W), float(self.H), self.sites, self.y)

        if cands is None:
            cands = gm._candidates(self.sites, frame_idx)
        cand0, cand1 = cands
        inv_sq = float(gm.inv_sq)
        if gs is None and self.fused_render:
            # render + clamp/MSE + render backward in one kernel (SAD's terms,
            # warp-aggregated accumulation).
            E.render_mse_backward(cand0, cand1, self.sites, gt_hwc, self.mse_alpha, inv_sq,
                                  self.W, self.H, self.gs, self.stats, self._empty)
            gs = self.gs
        elif gs is None:
            rendered = vops.render_sad(cand0, cand1, self.sites, inv_sq, self.W, self.H)
            E.mse_fb(rendered, gt_hwc, self.mse_alpha, self.grad_render, self.stats)
            gs = vops.render_sad_backward(cand0, cand1, self.sites, self.grad_render,
                                          inv_sq, self.W, self.H)
            self.rendered = rendered
        if fine:
            E.rom_backward(self.site_params, float(self.W), float(self.H), gs, self.site_grads,
                           self.mode, self.K, self.coef_table, self.tab, self.d, self.topi,
                           self.phi, self.y,
                           self.gy_arap if self.lam_arap > 0 else torch.empty(0, device=self.dev),
                           self.move_coef, self.gacc, self.gcoef)
            E.anchor_backward(self.t, self.stack_params[0], self.stack_grads[0],
                              self.stack_params[1] if len(self.stacks) > 1 else [],
                              self.stack_grads[1] if len(self.stacks) > 1 else [],
                              self.lin, self.smooth_coef, self.mode, *self.anchor_params,
                              self.anchor_grads, self.tab, self.gacc, self.gcoef)
        else:
            E.sites_backward(self.site_params, float(self.W), float(self.H), gs, self.site_grads)

    def set_fused_vpt(self, on: bool) -> bool:
        """Run the model's VPT passes (SADAppearance_V3._candidates) with the fused kernel (on) or
        with SAD's update_candidates_compact (off). The fused pass gives the same buffers bit for
        bit with nvcc 12.4 on sm_75; check another toolchain with
        `python tools/check_fast_ops.py --only vpt`. The override is set on this model instance
        only, so the eager step keeps SAD's op. A model with radius probes also keeps it.
        Returns whether the fused pass is on."""
        gm = self.tr.gaussian_model
        if not on or getattr(gm, "cand_radius_probes", 0) != 0 or not hasattr(gm, "_vpt_step"):
            if "_vpt_step" in gm.__dict__:
                del gm._vpt_step                 # the class's method: SAD's op
            return False
        if self.vpt_ph is None:
            self.vpt_ph = torch.zeros(self.N, 2, dtype=torch.int32, device=self.dev)
            self.vpt_pb = torch.zeros(self.N, 4, device=self.dev)
        gm._vpt_step = self._vpt_step
        return True

    @torch.no_grad()
    def _vpt_step(self, sites_det: torch.Tensor, frame_idx: int) -> None:
        """SADAppearance_V3._vpt_step with the fused kernels: the same pass (jump distance and step
        index of the frame's counter, inject probes, ping-pong buffers), the same buffers."""
        try:
            from .sad_appearance_v3 import _jump_for_step
        except ImportError:
            from sad_appearance_v3 import _jump_for_step
        gm = self.tr.gaussian_model
        gm._ensure_packed_buffers(sites_det)
        self.E.vpt_prep(sites_det, float(gm.inv_sq), self.vpt_ph, self.vpt_pb)
        step_idx = gm._frame_step_idx[frame_idx]
        jump = _jump_for_step(step_idx, gm.W, gm.H)
        cand0_in, cand1_in = gm.cand_updater.cand0, gm.cand_updater.cand1
        cand0_out, cand1_out = gm._cand0_b, gm._cand1_b
        self.E.vpt_update(cand0_in, cand1_in, cand0_out, cand1_out, self.vpt_ph, self.vpt_pb,
                          float(gm.inv_sq), (int(jump) << 16) | (step_idx & 0xFFFF), step_idx >> 16,
                          int(gm.cand_inject_count), int(gm.W), int(gm.H))
        gm.cand_updater.cand0, gm.cand_updater.cand1 = cand0_out, cand1_out
        gm._cand0_b, gm._cand1_b = cand0_in, cand1_in
        gm._frame_step_idx[frame_idx] = step_idx + 1

    # Decoding: fused ROM and site packing, then render_decode (render_sad's color and the
    # clamp to [0, 1] in one kernel) with the frame's candidates.
    def invalidate_basis(self) -> None:
        """Make the next decode() recompute the anchor weights, and start a new GoP pass of
        the exact builder (exact_candidates.GoPBuilder). Call it after changing parameters
        outside step()."""
        self._dec_basis = False
        eb = getattr(self.tr.gaussian_model, "exact_builder", None)
        if eb is not None:
            eb.reset()

    def _dec_buffers(self) -> dict:
        """decode()'s buffers: the anchor basis (topi, phi) and the frame's coefficient table and
        sites (also padded to 12 numbers per site, the layout render_decode loads fastest)."""
        if self._dec is None:
            dev, N = self.dev, self.N
            self._dec = dict(
                table=torch.zeros(self.M, self.C, device=dev), d=torch.zeros(N, self.C, device=dev),
                topi=torch.zeros(N, self.K, dtype=torch.int32, device=dev),
                phi=torch.zeros(N, self.K, device=dev), tab=torch.zeros(self.M, 8, device=dev),
                sites=torch.zeros(N, 10, device=dev), sites12=torch.zeros(N, 12, device=dev))
        return self._dec

    def _dec_basis_once(self, table: torch.Tensor) -> None:
        """Compute the anchor basis (topi, phi) once for the current parameters: rom_forward's
        selection and weights, which do not depend on t (the table only sets its d)."""
        if not self._dec_basis:
            b = self._dec
            self.E.rom_forward(self.tr.gaussian_model._xyz, self.mode, self.K, *self.anchor_params, table, True,
                               b["d"], b["topi"], b["phi"], b["tab"], False, [], 0.0, 0.0, self._empty,
                               self._empty, self._empty, self._empty)
            self._dec_basis = True

    @torch.no_grad()
    def decode(self, frame_idx: int, timestamp: float) -> torch.Tensor:
        """Frame `frame_idx` at time `timestamp`, [H, W, 3] clamped to [0, 1]."""
        E = self.E
        gm = self.tr.gaussian_model
        b = self._dec_buffers()
        E.temporal_forward(float(np.float32(timestamp)), self.stack_params[0],
                           self.stack_params[1] if len(self.stacks) > 1 else [],
                           self.lin, False, b["table"], self._empty)
        self._dec_basis_once(b["table"])
        # The frame's sites from the basis (bit-identical to rom_forward's packing), once in SAD's
        # layout for the candidates and once padded for the render.
        E.rom_apply_gop(gm._xyz, self.K, b["table"].unsqueeze(0), b["topi"], b["phi"], self.site_params,
                        float(self.W), float(self.H), b["sites"].unsqueeze(0), b["sites12"].unsqueeze(0))
        cand0, cand1 = gm.eval_candidates(b["sites"], frame_idx)
        out = torch.empty(self.H, self.W, 3, device=self.dev)
        E.render_decode(cand0, cand1, b["sites12"], float(gm.inv_sq), self.W, self.H, out)
        return out

    @torch.no_grad()
    def decode_gop(self, T: int, cands=None) -> torch.Tensor:
        """The T frames of a GoP pass, [T, H, W, 3], bit-identical to decode(i, (i + 1) / T) with
        exact candidates.

        The coefficient tables and the packed sites of all frames take one launch each. The
        exact candidates come from gm.exact_builder (whatever gm.cand_method is) and are built
        and rendered max_frames frames at a time, so that only those frames' buffers are alive.
        cands: a list of T per-frame (cand0, cand1) to render with instead; nothing is built."""
        E = self.E
        gm = self.tr.gaussian_model
        if cands is not None and len(cands) != T:
            raise ValueError(f"decode_gop: {len(cands)} candidate pairs for {T} frames")
        b = self._dec_buffers()
        g = self._gop
        if g is None or g["tables"].shape[0] != T:
            g = self._gop = dict(tables=torch.zeros(T, self.M, self.C, device=self.dev),
                                 sites=torch.zeros(T, self.N, 10, device=self.dev),
                                 sites12=torch.zeros(T, self.N, 12, device=self.dev))
        E.temporal_gop(self.stack_params[0], self.stack_params[1] if len(self.stacks) > 1 else [], g["tables"])
        self._dec_basis_once(g["tables"][0])
        E.rom_apply_gop(gm._xyz, self.K, g["tables"], b["topi"], b["phi"], self.site_params,
                        float(self.W), float(self.H), g["sites"], g["sites12"])
        inv_sq = float(gm.inv_sq)
        if cands is not None:
            out = torch.empty(T, self.H, self.W, 3, device=self.dev)
            for i, (c0, c1) in enumerate(cands):    # the same sites, padded: faster loads
                E.render_decode(c0, c1, g["sites12"][i], inv_sq, self.W, self.H, out[i])
            return out
        eb = gm.exact_builder
        step = getattr(eb, "max_frames", T)
        out = None
        for a in range(0, T, step):
            c0s, c1s = eb(g["sites"][a:a + step], self.W, self.H, inv_sq=gm.inv_sq)
            if out is None:     # after the first build, whose workspace is free again
                out = torch.empty(T, self.H, self.W, 3, device=self.dev)
            for i, (c0, c1) in enumerate(zip(c0s, c1s)):
                E.render_decode(c0, c1, g["sites12"][a + i], inv_sq, self.W, self.H, out[a + i])
            del c0s, c1s
        return out

    def deltas(self):
        """(Δxyz, Δopacity, Δfeatures) of the last fine step, for the diagnostic line."""
        d = self.d
        d_xyz = d[:, :2]
        d_feat = d[:, 2:5] if self.C >= 5 else torch.zeros(self.N, 3, device=self.dev)
        return d_xyz, torch.zeros(self.N, 1, device=self.dev), d_feat

    def _site_optimizer_step(self, opt):
        if isinstance(opt, torch.optim.Adam):
            self._adam_step(opt)
        else:
            self._adan_step(opt)
        opt._opt_called = True   # the LR scheduler checks that step() ran

    @torch.no_grad()
    def _adan_step(self, opt):
        """Adan.step() of baselines/D2GV/optimizer.py (multi-tensor path, no gradient clipping,
        no weight decay). It follows that file's arithmetic (Adan, Apache-2.0, THIRD_PARTY_NOTICES.md)."""
        P, G, M_, V, DIF, NPG, scal = [], [], [], [], [], [], []
        init = None
        for group in opt.param_groups:
            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            b1, b2, b3 = group["betas"]
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            bc3 = 1.0 - b3 ** step
            lr = group["lr"]
            step_size_diff = lr * b2 / bc2
            step_size = lr / bc1
            for p in group["params"]:
                g = self._grad_of.get(p)
                if g is None:
                    continue
                st = opt.state[p]
                if len(st) == 0:
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["exp_avg_diff"] = torch.zeros_like(p)
                first = "neg_pre_grad" not in st or step == 1
                if "neg_pre_grad" not in st:
                    st["neg_pre_grad"] = torch.zeros_like(p)
                if init is None:
                    init = first
                assert init == first, "Adan groups out of step"
                P.append(p.data); G.append(g); M_.append(st["exp_avg"])
                V.append(st["exp_avg_sq"]); DIF.append(st["exp_avg_diff"])
                NPG.append(st["neg_pre_grad"])
                # _foreach_div_(denom, bias_correction3_sqrt) multiplies by the
                # reciprocal, computed in double and rounded to float32.
                inv_bc3s = float(np.float32(1.0 / math.sqrt(bc3)))
                scal += [b1, b2, b3, 1 - b1, 1 - b2, 1 - b3, inv_bc3s, group["eps"],
                         -step_size, -step_size_diff]
        if P:
            self.E.adan_step(P, G, M_, V, DIF, NPG, scal, bool(init))

    @torch.no_grad()
    def _adam_step(self, opt):
        """torch.optim.Adam.step() (foreach path, no amsgrad / weight decay). It follows PyTorch's
        arithmetic (BSD-3-Clause, THIRD_PARTY_NOTICES.md)."""
        P, G, M_, V, S, plist = [], [], [], [], [], []
        for group in opt.param_groups:
            for p in group["params"]:
                g = self._grad_of.get(p)
                if g is None:
                    continue
                st = opt.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0, dtype=torch.float32)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                P.append(p.data); G.append(g); M_.append(st["exp_avg"]); V.append(st["exp_avg_sq"])
                S.append(st["step"])
                plist.append(group)
        if not P:
            return
        # Steps are float32 CPU tensors, incremented as torch does (+1.0 each).
        torch._foreach_add_(S, 1.0)
        step = S[0].item()
        if id(opt) not in self._adam_checked:
            if any(s.item() != step for s in S):
                raise RuntimeError("fused Adam needs equal step counts for all parameters")
            self._adam_checked.add(id(opt))
        scal = []
        cache = {}
        for group in plist:
            key = id(group)
            if key not in cache:
                b1, b2 = group["betas"]
                bc1 = 1 - b1 ** step
                bc2 = 1 - b2 ** step
                cache[key] = [1 - b1, b2, 1 - b2, bc2 ** 0.5, group["eps"], (group["lr"] / bc1) * -1]
            scal += cache[key]
        self.E.adam_step(P, G, M_, V, scal)
