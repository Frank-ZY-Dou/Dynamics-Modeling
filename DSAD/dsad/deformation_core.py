"""The anchor ROM (the paper's Sec. 3.3): Δ(x, t) = Σ_j φ_j(x) a_j(t) over M anchors.

φ_j are the anchor weights (motion_sad.MotionSAD) and a_j(t) the knot-based
temporal coefficients (temporal_1d_sad.Temporal1DSAD), one stack per residual
channel: Δxyz, and optionally Δfeatures and Δopacity. DeformModelSAD is a
drop-in replacement for D2GV's DeformModel:

    step(xyz, timestamp) -> (Δxyz, Δopacity, Δfeatures)
    train_setting(), update_learning_rate(it)
    save_weights(path, it), load_weights(path, it)

Parameters (Eq. 8): M (3 + ν) for the anchors, with ν = 0 for voronoi, 1 for
apollonius and power, 4 for apollonius_aniso; and M K (2 + d) for a stack of
K knots with d channels. The paper's ROM (apollonius_aniso, Δxyz and
Δfeatures) has 7 M + 9 M K parameters: 20,224 at M = 256, K = 8.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

try:
    from .motion_sad import MotionSAD
    from .temporal_1d_sad import Temporal1DSAD
except ImportError:
    from motion_sad import MotionSAD
    from temporal_1d_sad import Temporal1DSAD


def _expon_lr(lr_init, lr_final, max_steps, lr_delay_mult=0.01, lr_delay_steps=0):
    """Exponential LR schedule with optional warm-up delay.

    Re-implements, with `math` in place of numpy, the decay D2GV uses
    (`utils.get_expon_lr_func`), which comes from Plenoxels (BSD 2-Clause) and
    JaxNeRF (Apache 2.0); see THIRD_PARTY_NOTICES.md."""
    import math
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                0.5 * math.pi * max(min(step / lr_delay_steps, 1.0), 0.0)
            )
        else:
            delay = 1.0
        t = max(min(step / max_steps, 1.0), 0.0)
        log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
        return delay * log_lerp
    return helper


class SADDeformCore(nn.Module):
    """Anchor weights times temporal coefficients.

    One anchor field (`anchors`) is shared by the coefficient stacks of the
    enabled channels: coef_xyz (always), coef_opacity and coef_features.
    """

    def __init__(
        self,
        num_anchors: int = 128,
        num_knots: int = 8,
        anchor_init_sigma: float = 0.25,
        time_init_radius: float = 0.15,
        anchor_top_k: int | None = 8,
        time_top_k: int | None = None,
        enable_opacity: bool = False,
        enable_features: bool = False,
        anchor_mode: str = "voronoi",
    ):
        super().__init__()
        self.anchors = MotionSAD(
            num_anchors=num_anchors,
            init_sigma=anchor_init_sigma,
            top_k=anchor_top_k,
            mode=anchor_mode,
        )
        self.coef_xyz = Temporal1DSAD(
            num_anchors=num_anchors, num_knots=num_knots, out_dim=2,
            init_radius=time_init_radius, top_k=time_top_k,
        )
        self.enable_opacity = enable_opacity
        self.enable_features = enable_features
        if enable_opacity:
            self.coef_opacity = Temporal1DSAD(
                num_anchors=num_anchors, num_knots=num_knots, out_dim=1,
                init_radius=time_init_radius, top_k=time_top_k,
            )
        if enable_features:
            self.coef_features = Temporal1DSAD(
                num_anchors=num_anchors, num_knots=num_knots, out_dim=3,
                init_radius=time_init_radius, top_k=time_top_k,
            )

    def forward(self, xyz: torch.Tensor, timestamp: torch.Tensor):
        """(Δxyz [N, 2], Δopacity [N, 1], Δfeatures [N, 3]) at the canonical positions
        xyz [N, 2] and the 0-dim time `timestamp`. A disabled channel is zero."""
        phi = self.anchors.basis(xyz)                     # [N, M]
        a_xyz = self.coef_xyz(timestamp)                   # [M, 2]
        d_xyz = phi @ a_xyz                                # [N, 2]

        if self.enable_opacity:
            d_op = phi @ self.coef_opacity(timestamp)
        else:
            d_op = torch.zeros(xyz.size(0), 1, device=xyz.device, dtype=xyz.dtype)
        if self.enable_features:
            d_feat = phi @ self.coef_features(timestamp)
        else:
            d_feat = torch.zeros(xyz.size(0), 3, device=xyz.device, dtype=xyz.dtype)
        return d_xyz, d_op, d_feat


class DeformModelSAD:
    """SADDeformCore with its Adam optimizer and learning-rate schedule, behind the
    interface of D2GV's DeformModel."""

    def __init__(
        self,
        num_anchors: int = 128,
        num_knots: int = 8,
        anchor_init_sigma: float = 0.25,
        time_init_radius: float = 0.15,
        anchor_top_k: int | None = 8,
        time_top_k: int | None = None,
        enable_opacity: bool = False,
        enable_features: bool = False,
        anchor_mode: str = "voronoi",
        lr_anchor: float = 5e-3,
        lr_coef: float = 5e-3,
        max_steps: int = 50000,
        device: str = "cuda",
    ):
        self.deform = SADDeformCore(
            num_anchors=num_anchors,
            num_knots=num_knots,
            anchor_init_sigma=anchor_init_sigma,
            time_init_radius=time_init_radius,
            anchor_top_k=anchor_top_k,
            time_top_k=time_top_k,
            enable_opacity=enable_opacity,
            enable_features=enable_features,
            anchor_mode=anchor_mode,
        ).to(device)
        self.optimizer = None
        self.lr_anchor = lr_anchor
        self.lr_coef = lr_coef
        self.max_steps = max_steps
        self._ts_cache = {}

    def timestamp_tensor(self, timestamp: float, device, dtype) -> torch.Tensor:
        """0-dim tensor holding `timestamp`, cached per value. Building it with
        torch.tensor() on every call would copy from the host and synchronize the
        stream; a GoP only uses its T frame times."""
        key = (float(timestamp), torch.device(device), dtype)
        t = self._ts_cache.get(key)
        if t is None:
            t = torch.tensor(key[0], device=device, dtype=dtype)
            if len(self._ts_cache) < 4096:
                self._ts_cache[key] = t
        return t

    def step(self, xyz: torch.Tensor, timestamp):
        """(Δxyz, Δopacity, Δfeatures) at the canonical positions xyz [N, 2] and time
        `timestamp` (a float or a tensor)."""
        if not isinstance(timestamp, torch.Tensor):
            timestamp = self.timestamp_tensor(timestamp, xyz.device, xyz.dtype)
        timestamp = timestamp.to(xyz.device).to(xyz.dtype).reshape(())
        d_xyz, d_op, d_feat = self.deform(xyz, timestamp)
        return d_xyz, d_op, d_feat

    def train_setting(self):
        """Build the Adam optimizer with the parameter groups "anchors" (lr_anchor) and
        "coeffs" (lr_coef). Each rate decays exponentially to 1% over max_steps."""
        a = self.deform.anchors
        anchor_params = [a.mu, a.log_sigma]
        if a.mode == "power":
            anchor_params.append(a.weight)
        elif a.mode == "apollonius":
            anchor_params.append(a.radius)
        elif a.mode == "apollonius_aniso":
            anchor_params += [a.radius, a.log_aspect, a.aniso_dir]

        coef_modules = [self.deform.coef_xyz]
        if self.deform.enable_opacity:
            coef_modules.append(self.deform.coef_opacity)
        if self.deform.enable_features:
            coef_modules.append(self.deform.coef_features)
        import itertools
        coef_params = list(itertools.chain(*[m.parameters() for m in coef_modules]))

        param_groups = [
            {"params": anchor_params, "lr": self.lr_anchor, "name": "anchors"},
            {"params": coef_params,   "lr": self.lr_coef,   "name": "coeffs"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self._sched_anchor = _expon_lr(self.lr_anchor, self.lr_anchor * 0.01,
                                       max_steps=self.max_steps, lr_delay_mult=0.01)
        self._sched_coef = _expon_lr(self.lr_coef, self.lr_coef * 0.01,
                                     max_steps=self.max_steps, lr_delay_mult=0.01)

    def update_learning_rate(self, iteration: int):
        for pg in self.optimizer.param_groups:
            if pg["name"] == "anchors":
                pg["lr"] = self._sched_anchor(iteration)
            elif pg["name"] == "coeffs":
                pg["lr"] = self._sched_coef(iteration)
        return self.optimizer.param_groups[0]["lr"]

    def save_weights(self, model_path: str, iteration: int):
        out = os.path.join(model_path, f"deform_sad/iteration_{iteration}")
        os.makedirs(out, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out, "deform_sad.pth"))

    def load_weights(self, model_path: str, iteration: int):
        path = os.path.join(model_path, f"deform_sad/iteration_{iteration}/deform_sad.pth")
        self.deform.load_state_dict(torch.load(path))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.deform.parameters())
