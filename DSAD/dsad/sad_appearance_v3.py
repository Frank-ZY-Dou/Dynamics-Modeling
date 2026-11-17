"""SADAppearance_V3: SAD appearance (--appearance sad) on the SAD repository's PyTorch backend.

From references/SAD/backends/pytorch it uses torch.ops.sad.* (render_sad,
render_sad_backward, the JFA seed and flood, the VPT candidate update) and
modules.SADRenderer and CandidateUpdater.

Interface (that of D2GV's GaussianImage_Cholesky, plus a frame index):
    forward_mse_loss(gt, frame_idx)                              -> scalar loss
    forward_dynamic_mse_loss(d_xyz, d_op, d_feat, gt, frame_idx) -> scalar loss
    forward(frame_idx)                                           -> {"render": [1, 3, H, W]}
    forward_dynamic(d_xyz, d_op, d_feat, frame_idx)              -> {"render": [1, 3, H, W]}
    get_xyz, get_features, get_opacity                           properties
    optimizer, scheduler                                         Adan, StepLR

Candidates. Each pixel blends its top-8 sites, read from two int32 [H*W, 4]
candidate buffers. They are built with SAD's jump flooding (JFA) and refined
with its Voronoi particle tracking (VPT). Every frame of the GoP keeps its own
buffers: a JFA seed, a full flood and VPT passes on first use, then a VPT pass
every cand_freq calls and a full re-flood every cand_flood_every calls. With cand_method =
"exact" the buffers are instead rebuilt by exact_candidates.GoPBuilder every
cand_freq calls, and evaluation renders with freshly built exact candidates.
The trainer's --cand_exact_tail N switches to it for the last N fine
iterations (N = 6000 in the paper). The buffers are not saved with the
checkpoint: a decoder rebuilds them from the sites.
"""
from __future__ import annotations
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import _paths
except ImportError:
    import _paths

# SAD's PyTorch extension, found from this file's location.
if not (_paths.SAD_BACKEND / "torch_ext").is_dir():
    raise ImportError(
        f"SAD backend not found at {_paths.SAD_BACKEND}.\n"
        "The SAD appearance decoder needs the CUDA extension from the SAD repository "
        "(MIT, https://github.com/LuckyIYI/SAD). From the DSAD directory run:\n"
        "    bash setup.sh\n"
        "Training with --appearance gaussian (the DSAD-GS ablation) does not need it.")
if str(_paths.SAD_BACKEND) not in sys.path:
    sys.path.insert(0, str(_paths.SAD_BACKEND))
try:
    import torch_ext.sad_ops  # noqa: F401,E402  -- registers torch.ops.sad.*
except ImportError as _e:   # Python reports a missing build as a "circular import": say what is wrong
    _ext = _paths.SAD_BACKEND / "torch_ext" / "sad_ops"
    raise ImportError(
        (f"SAD's CUDA extension is not built (no _sad_ext* in {_ext})" if not any(_ext.glob("_sad_ext*"))
         else f"SAD's CUDA extension did not load ({_e})")
        + ". From the DSAD directory run:\n    bash setup.sh") from _e
from modules import SADRenderer, CandidateUpdater  # noqa: E402

# Adan (D2GV's optimizer for canonical primitives)
if str(_paths.D2GV_DIR) not in sys.path:
    sys.path.insert(0, str(_paths.D2GV_DIR))
try:
    from optimizer import Adan as AdanOpt
except Exception as _e:  # reported when a model is built (the sites are optimized with Adan)
    AdanOpt = None
    _adan_import_err = _e

vops = torch.ops.sad


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _jump_for_step(step_idx: int, W: int, H: int) -> int:
    """Jump distance of VPT pass step_idx, cycling through max_dim/2, max_dim/4, ..., 1.
    Adapted from SAD's _jump_step_for_index (backends/pytorch/train.py; MIT License,
    THIRD_PARTY_NOTICES.md), which holds the last distance instead of cycling."""
    pow2 = _next_pow2(max(W, H))
    if pow2 <= 1:
        return 1
    stages = 0
    tmp = pow2
    while tmp > 1:
        tmp >>= 1
        stages += 1
    stage = step_idx % max(1, stages)
    jump = pow2 >> (stage + 1)
    return max(jump, 1)


def _build_exact(sites_det: torch.Tensor, W: int, H: int, inv_sq: float):
    """Exact top-8 candidate buffers for one frame (exact_candidates.py, default
    settings). The model's own builds use its GoPBuilder (same candidates)."""
    try:
        from .exact_candidates import build_candidates
    except ImportError:
        from exact_candidates import build_candidates
    return build_candidates(sites_det, W, H, inv_sq=inv_sq)


def _gop_builder():
    """A new per-GoP exact builder (exact_candidates.GoPBuilder with its defaults)."""
    try:
        from .exact_candidates import GoPBuilder
    except ImportError:
        from exact_candidates import GoPBuilder
    return GoPBuilder()


class SADAppearance_V3(nn.Module):
    """The canonical SAD sites of one GoP, with a candidate buffer pair per frame.

    Candidate settings (the trainer passes its --cand_* flags; the paper uses
    cand_freq=2, cand_warmup_passes=16, cand_flood_every=1000, cand_inject_count=4):
        cand_freq               update a frame's candidates every N of its calls.
        cand_passes_per_update  VPT passes per update.
        cand_warmup_passes      VPT passes after each JFA flood.
        cand_inject_count       random sites injected per VPT pass.
        cand_radius_scale, cand_radius_probes   radius probes of SAD's VPT update (0: off).
        cand_flood_every        JFA re-flood every N calls of a frame (<= 0: never).
        cand_flood_levels       unused: every flood runs the full pyramid.
        cand_method             "vpt": VPT passes with periodic floods. "exact": a rebuild
                                with the exact builder (exact_candidates.py) every
                                cand_freq calls; evaluation then renders with freshly
                                built exact candidates too.

    The six site attributes are optimized with Adan (D2GV's optimizer.py), with optional
    per-attribute LR multipliers `lr_mults` (e.g. {"color": 2.0}), and StepLR halves the
    rates every `step_size` steps, counted over both stages.
    """

    def __init__(self, loss_type="L2", opt_type="adan",
                 num_points=25000, H=720, W=1280,
                 BLOCK_H=16, BLOCK_W=16, device="cuda",
                 lr=5e-3, quantize=False,
                 cand_freq=1,
                 cand_passes_per_update=1,
                 cand_warmup_passes=12,
                 cand_inject_count=4,
                 cand_radius_scale=64.0,
                 cand_radius_probes=0,
                 cand_flood_every=200,
                 cand_flood_levels=0,
                 num_frames=1,
                 lr_mults=None,
                 init_image=None,
                 step_size=20000,
                 cand_method="vpt",
                 **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.H, self.W = int(H), int(W)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.BLOCK_H, self.BLOCK_W = BLOCK_H, BLOCK_W
        self.cand_freq = int(cand_freq)
        self.cand_passes_per_update = int(cand_passes_per_update)
        self.cand_warmup_passes = int(cand_warmup_passes)
        self.cand_inject_count = int(cand_inject_count)
        self.cand_radius_scale = float(cand_radius_scale)
        self.cand_radius_probes = int(cand_radius_probes)
        self.cand_flood_every = int(cand_flood_every)
        self.cand_flood_levels = int(cand_flood_levels)
        if cand_method not in ("vpt", "exact"):
            raise ValueError(f"cand_method must be 'vpt' or 'exact', not {cand_method!r}")
        self.cand_method = cand_method
        # The builder of the exact candidates (cand_method "exact", evaluation). Its cap is chosen
        # once per GoP pass: FusedStep.invalidate_basis() and the trainer, at every stage start,
        # reset it.
        self.exact_builder = _gop_builder()
        self.num_frames = max(1, int(num_frames))
        # One candidate buffer pair per frame (the paper's App. E). A single pair shared by the T
        # deformed configurations of the sites would be stale at every frame; a frame's own pair
        # only tracks the slow drift of the canonical sites.
        self._frame_cand0 = [None] * self.num_frames
        self._frame_cand1 = [None] * self.num_frames
        self._frame_init = [False] * self.num_frames
        self._frame_step_idx = [0] * self.num_frames
        self._frame_iter_count = [0] * self.num_frames
        self._active_frame = None
        self.inv_sq = 1.0 / float(max(self.W, self.H) ** 2)

        # Site initialization: gradient-weighted on init_image, or uniformly random.
        if init_image is not None:
            # init_image is [H, W, 3] in [0, 1], e.g. the GoP's temporal mean. Half the
            # probability mass is uniform and half proportional to the luminance-gradient
            # magnitude, so flat regions still get sites. Colors are read from the image.
            img = init_image.to(self.device, torch.float32)
            lum = img.mean(-1)
            gx = torch.zeros_like(lum)
            gy = torch.zeros_like(lum)
            gx[:, 1:-1] = 0.5 * (lum[:, 2:] - lum[:, :-2])
            gy[1:-1] = 0.5 * (lum[2:] - lum[:-2])
            mag = torch.sqrt(gx * gx + gy * gy).reshape(-1)
            prob = 0.5 / mag.numel() + 0.5 * mag / mag.sum().clamp_min(1e-12)
            idx = torch.multinomial(prob, num_points, replacement=True)
            sp0 = torch.zeros(num_points, 10, device=self.device, dtype=torch.float32)
            sp0[:, 0] = (idx % self.W).float() + torch.rand(num_points, device=self.device)
            sp0[:, 1] = (idx // self.W).float() + torch.rand(num_points, device=self.device)
            sp0[:, 2] = 5.0
            sp0[:, 3] = 4.0
            sp0[:, 4:7] = img.reshape(-1, 3)[idx]
            sp0[:, 7] = 1.0
            sp0[:, 8] = 0.0
            sp0[:, 9] = 0.0
        else:
            sp0 = torch.zeros(num_points, 10, device=self.device, dtype=torch.float32)
            sp0[:, 0] = torch.rand(num_points, device=self.device) * self.W
            sp0[:, 1] = torch.rand(num_points, device=self.device) * self.H
            sp0[:, 2] = 5.0
            sp0[:, 3] = 4.0
            sp0[:, 4:7] = torch.rand(num_points, 3, device=self.device)
            sp0[:, 7] = 1.0
            sp0[:, 8] = 0.0
            sp0[:, 9] = 0.0
        self.num_points = num_points

        # Positions are stored before the tanh, the space in which the ROM's Δxyz is added.
        pos_norm = (2.0 * sp0[:, 0:2] / torch.tensor([self.W, self.H], device=self.device)
                    - 1.0).clamp(-0.999, 0.999)
        self._xyz         = nn.Parameter(torch.atanh(pos_norm))
        self._features_dc = nn.Parameter(sp0[:, 4:7].clone())
        self._log_tau     = nn.Parameter(sp0[:, 2].clone())
        self._radius      = nn.Parameter(sp0[:, 3].clone())
        self._aniso_dir   = nn.Parameter(sp0[:, 7:9].clone())
        self._log_aniso   = nn.Parameter(sp0[:, 9].clone())
        self.register_buffer('_opacity', torch.ones((self.num_points, 1)))

        # Optimizer and StepLR. lr_mults scales `lr` per attribute, e.g. {"xyz": 0.1}; without it
        # every site attribute shares `lr`.
        params = self.parameters()
        if lr_mults:
            attrs = {"xyz": self._xyz, "color": self._features_dc,
                     "tau": self._log_tau, "radius": self._radius,
                     "dir": self._aniso_dir, "aniso": self._log_aniso}
            unknown = set(lr_mults) - set(attrs)
            if unknown:
                raise ValueError(f"unknown lr_mults keys {sorted(unknown)}; "
                                 f"valid: {sorted(attrs)}")
            params = [{"params": [p], "lr": lr * float(lr_mults.get(k, 1.0)), "name": k}
                      for k, p in attrs.items()]
        if opt_type != "adan":
            raise ValueError(f"opt_type must be 'adan' (the sites are optimized with Adan), not {opt_type!r}")
        if AdanOpt is None:
            raise ImportError(f"Adan (baselines/D2GV/optimizer.py) could not be imported: {_adan_import_err}; "
                              "run setup.sh to fetch the D2GV submodule")
        self.optimizer = AdanOpt(params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(step_size), gamma=0.5)

        # SAD's CandidateUpdater and SADRenderer.
        self.cand_updater = CandidateUpdater(
            width=self.W, height=self.H,
            site_count=self.num_points,
            seed=0,
            device=self.device,
        )
        self.renderer = SADRenderer(inv_scale_sq=self.inv_sq)

    def _wh(self) -> torch.Tensor:
        """[W, H] as a float32 tensor on the sites' device. Cached: building it
        with torch.tensor() copies from the host and synchronizes the stream on
        every call. Not a registered buffer, so state_dict does not hold it."""
        dev = self._xyz.device
        wh = getattr(self, "_wh_cache", None)
        if wh is None or wh.device != dev:
            wh = torch.tensor([self.W, self.H], device=dev, dtype=torch.float32)
            self._wh_cache = wh
        return wh

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self._opacity

    def _pack_sites_grad(self, pos_px: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Packed [N, 10] sites, differentiable in every site parameter. Columns, as the
        render kernel reads them: 0:2 position (pixels), 2 log_tau, 3 radius, 4:7 color,
        7:9 unit anisotropy direction, 9 log_aniso."""
        u = self._aniso_dir / (self._aniso_dir.norm(dim=-1, keepdim=True) + 1e-8)
        sites = torch.stack([
            pos_px[:, 0],
            pos_px[:, 1],
            self._log_tau,
            self._radius,
            features[:, 0],
            features[:, 1],
            features[:, 2],
            u[:, 0],
            u[:, 1],
            self._log_aniso,
        ], dim=-1).contiguous()  # [N, 10]
        return sites

    # Per-frame candidate state. cand_updater.cand0 / cand1 hold the active frame's buffers;
    # those of the other frames are kept in _frame_cand0 / _frame_cand1, and _activate_frame
    # swaps them in before any read or update.
    @torch.no_grad()
    def _activate_frame(self, frame_idx: int) -> None:
        """Make frame_idx's candidate buffers the active ones in cand_updater, allocating
        them (zeros) on first use."""
        if frame_idx == self._active_frame:
            return
        if self._active_frame is not None:
            self._frame_cand0[self._active_frame] = self.cand_updater.cand0
            self._frame_cand1[self._active_frame] = self.cand_updater.cand1
        if self._frame_cand0[frame_idx] is None:
            self._frame_cand0[frame_idx] = torch.zeros_like(self.cand_updater.cand0)
            self._frame_cand1[frame_idx] = torch.zeros_like(self.cand_updater.cand1)
        self.cand_updater.cand0 = self._frame_cand0[frame_idx]
        self.cand_updater.cand1 = self._frame_cand1[frame_idx]
        self._active_frame = frame_idx

    @torch.no_grad()
    def _jfa_warmup(self, sites_det: torch.Tensor, frame_idx: int) -> None:
        """JFA seed, full flood and cand_warmup_passes VPT passes for frame_idx, which must
        be the active frame."""
        W, H, M = self.W, self.H, self.num_points
        cand0 = self.cand_updater.cand0
        # Seed: every site enters cand0 at its own pixel.
        vops.jfa_seed(cand0, sites_det, M, 1, W, H)
        # Flood with the step halving from next_pow2(max(W, H)) / 2 down to 1.
        cand0_b = torch.zeros_like(cand0)
        step = _next_pow2(max(W, H)) // 2
        rd, wr = cand0, cand0_b
        while step >= 1:
            vops.jfa_flood(rd, wr, sites_det, float(self.inv_sq), M, int(step), 1, W, H)
            rd, wr = wr, rd
            step //= 2
        self.cand_updater.cand0 = rd
        for _ in range(self.cand_warmup_passes):
            self._vpt_step(sites_det, frame_idx)
        self._frame_init[frame_idx] = True

    @torch.no_grad()
    def _ensure_packed_buffers(self, sites_det: torch.Tensor) -> None:
        """Allocate, on first use, the half-precision packed sites and the ping-pong
        candidate buffers of update_candidates_compact. All frames share them."""
        if getattr(self, "_packed", None) is None:
            self._packed = torch.zeros((self.num_points, 8),
                                        device=sites_det.device,
                                        dtype=torch.float16)
            self._cand0_b = torch.zeros_like(self.cand_updater.cand0)
            self._cand1_b = torch.zeros_like(self.cand_updater.cand1)
            self._dummy_hilbert = torch.zeros((1,),
                                               device=sites_det.device,
                                               dtype=torch.int32)

    @torch.no_grad()
    def _vpt_step(self, sites_det: torch.Tensor, frame_idx: int) -> None:
        """One update_candidates_compact pass on the active frame. The jump distance
        cycles with the frame's step index, which also seeds the injected sites, so
        every pass injects different ones."""
        self._ensure_packed_buffers(sites_det)
        # The sites change at every iteration, so they are packed at every pass.
        vops.pack_candidate_sites(sites_det, self._packed, self.num_points)
        step_idx = self._frame_step_idx[frame_idx]
        jump = _jump_for_step(step_idx, self.W, self.H)
        # step: the jump in the high 16 bits, the step index in the low 16. step_high
        # carries the bits of a step index above 0xFFFF.
        step = (int(jump) << 16) | (step_idx & 0xFFFF)
        step_high = step_idx >> 16
        cand0_in = self.cand_updater.cand0
        cand1_in = self.cand_updater.cand1
        cand0_out = self._cand0_b
        cand1_out = self._cand1_b
        vops.update_candidates_compact(
            cand0_in, cand1_in, cand0_out, cand1_out,
            self._packed,
            self._dummy_hilbert, self._dummy_hilbert,
            float(self.inv_sq),
            int(self.num_points),
            int(step),
            int(step_high),
            float(self.cand_radius_scale),
            int(self.cand_radius_probes),
            int(self.cand_inject_count),
            0,    # hilbert_probes (unused)
            0,    # hilbert_window
            1,    # cand_downscale
            int(self.W), int(self.H),  # target dims
            int(self.W), int(self.H),  # cand dims
        )
        # Ping-pong: the buffers just written are the next pass's inputs.
        self.cand_updater.cand0 = cand0_out
        self.cand_updater.cand1 = cand1_out
        self._cand0_b = cand0_in
        self._cand1_b = cand1_in
        self._frame_step_idx[frame_idx] = step_idx + 1

    @torch.no_grad()
    def _periodic_flood(self, sites_det: torch.Tensor, frame_idx: int) -> None:
        """Rebuild the active frame's candidates from the current sites, as _jfa_warmup does:
        JFA seed, full flood and cand_warmup_passes VPT passes. Run every cand_flood_every
        calls, so that the candidates cannot drift indefinitely."""
        W, H, M = self.W, self.H, self.num_points
        cand0 = self.cand_updater.cand0
        vops.jfa_seed(cand0, sites_det, M, 1, W, H)
        cand0_b = torch.zeros_like(cand0)
        step = _next_pow2(max(W, H)) // 2
        rd, wr = cand0, cand0_b
        while step >= 1:
            vops.jfa_flood(rd, wr, sites_det, float(self.inv_sq), M, int(step), 1, W, H)
            rd, wr = wr, rd
            step //= 2
        self.cand_updater.cand0 = rd
        for _ in range(self.cand_warmup_passes):
            self._vpt_step(sites_det, frame_idx)

    @torch.no_grad()
    def _candidates(self, sites_det: torch.Tensor, frame_idx: int):
        """Advance `frame_idx`'s candidate state for one training render and
        return its (cand0, cand1): JFA warm-up on the frame's first call, then a
        VPT pass every `cand_freq` calls and a full re-flood every
        `cand_flood_every` calls (per-frame counters)."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} out of range for num_frames={self.num_frames}")
        self._activate_frame(frame_idx)

        if self.cand_method == "exact":
            # Rebuild the exact top-8 on the first call and every cand_freq calls; the
            # frame keeps it in between.
            if not self._frame_init[frame_idx] or self._frame_iter_count[frame_idx] % self.cand_freq == 0:
                c0, c1 = self.exact_builder(sites_det, self.W, self.H, inv_sq=self.inv_sq)
                self.cand_updater.cand0, self.cand_updater.cand1 = c0, c1
                self._frame_init[frame_idx] = True
            self._frame_iter_count[frame_idx] += 1
            return self.cand_updater.get()

        if not self._frame_init[frame_idx]:
            self._jfa_warmup(sites_det, frame_idx)
        else:
            it_f = self._frame_iter_count[frame_idx]
            if (self.cand_flood_every > 0 and it_f > 0
                    and it_f % self.cand_flood_every == 0):
                self._periodic_flood(sites_det, frame_idx)
            if it_f % self.cand_freq == 0:
                for _ in range(self.cand_passes_per_update):
                    self._vpt_step(sites_det, frame_idx)
        self._frame_iter_count[frame_idx] += 1
        return self.cand_updater.get()

    @torch.no_grad()
    def eval_candidates(self, sites_det: torch.Tensor, frame_idx: int):
        """Candidates for an evaluation render of `frame_idx`. With cand_method "exact":
        freshly built exact top-8, as the decoder uses. Otherwise the frame's training
        buffers as they are (JFA warm-up on first use); the call does not count toward
        cand_freq or cand_flood_every."""
        if self.cand_method == "exact":
            return self.exact_builder(sites_det, self.W, self.H, inv_sq=self.inv_sq)
        self._activate_frame(frame_idx)
        if not self._frame_init[frame_idx]:
            self._jfa_warmup(sites_det, frame_idx)
        return self.cand_updater.get()

    def _mse_loss(self, pos_px: torch.Tensor, features: torch.Tensor,
                  gt: torch.Tensor, frame_idx: int = 0) -> torch.Tensor:
        """MSE between the render clamped to [0, 1] and gt ([1, 3, H, W] or [H, W, 3]).
        Advances the frame's candidate state (_candidates)."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} out of range for num_frames={self.num_frames}")
        sites_grad = self._pack_sites_grad(pos_px, features)
        cand0, cand1 = self._candidates(sites_grad.detach(), frame_idx)
        rendered = self.renderer(sites_grad, cand0, cand1, self.W, self.H)
        # Strip alpha if the kernel returned RGBA.
        if rendered.dim() == 3 and rendered.size(-1) == 4:
            rendered = rendered[..., :3]
        rendered = rendered.clamp(0, 1)

        if gt.dim() == 4:
            gt_hwc = gt.squeeze(0).permute(1, 2, 0).contiguous()
        else:
            gt_hwc = gt.contiguous()
        return F.mse_loss(rendered, gt_hwc)

    def forward_mse_loss(self, gt: torch.Tensor, frame_idx: int = 0) -> torch.Tensor:
        """MSE of the canonical render (no deformation) against gt."""
        wh = self._wh()
        pos_px = (self.get_xyz + 1.0) * 0.5 * wh
        return self._mse_loss(pos_px, self.get_features, gt, frame_idx=frame_idx)

    def forward_dynamic_mse_loss(self, deformed_xyz, deformed_opacity,
                                  deformed_features, gt, frame_idx: int = 0):
        """MSE of the deformed render against gt. The arguments are the residuals: the
        positions are tanh(_xyz + Δxyz), the colors _features_dc + Δfeatures. Δopacity
        is ignored."""
        xyz_def = torch.tanh(self._xyz + deformed_xyz)
        feat_def = self._features_dc + deformed_features
        wh = self._wh()
        pos_px = (xyz_def + 1.0) * 0.5 * wh
        return self._mse_loss(pos_px, feat_def, gt, frame_idx=frame_idx)

    @torch.no_grad()
    def _render_detached(self, pos_px: torch.Tensor,
                         features: torch.Tensor,
                         frame_idx: int = 0) -> torch.Tensor:
        """Render with eval_candidates, without gradients: [H, W, 3] clamped to [0, 1]."""
        sites = self._pack_sites_grad(pos_px, features).detach()
        cand0, cand1 = self.eval_candidates(sites, frame_idx)
        out = vops.render_sad(cand0, cand1, sites, float(self.inv_sq),
                              int(self.W), int(self.H))
        if out.dim() == 3 and out.size(-1) == 4:
            out = out[..., :3]
        return out.clamp(0, 1)

    @torch.no_grad()
    def forward(self, frame_idx: int = 0):
        """The canonical render: {"render": [1, 3, H, W]}."""
        wh = self._wh()
        pos_px = (self.get_xyz + 1.0) * 0.5 * wh
        rendered = self._render_detached(pos_px, self.get_features, frame_idx=frame_idx)
        return {"render": rendered.unsqueeze(0).permute(0, 3, 1, 2).contiguous()}

    @torch.no_grad()
    def forward_dynamic(self, deformed_xyz, deformed_opacity, deformed_features,
                        frame_idx: int = 0):
        """The deformed render, {"render": [1, 3, H, W]}; residuals as in
        forward_dynamic_mse_loss."""
        xyz_def = torch.tanh(self._xyz + deformed_xyz)
        feat_def = self._features_dc + deformed_features
        wh = self._wh()
        pos_px = (xyz_def + 1.0) * 0.5 * wh
        rendered = self._render_detached(pos_px, feat_def, frame_idx=frame_idx)
        return {"render": rendered.unsqueeze(0).permute(0, 3, 1, 2).contiguous()}
