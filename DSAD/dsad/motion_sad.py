"""Anchor tessellation: the spatial basis φ_j(x) of the ROM (the paper's Sec. 3.1).

Each of the M anchors has a position p_j and a temperature τ_j. `mode` selects
its score d_j(x) at a point x in canonical coordinates [-1, 1]^2 (Eq. 3):

    voronoi           ‖x - p_j‖
    apollonius        ‖x - p_j‖ - r_j
    power             ‖x - p_j‖² - w_j
    apollonius_aniso  ‖x - p_j‖_G - r_j,  G = e^α û û^T + e^(-α) (I - û û^T),  det G = 1

apollonius_aniso is the SAD score, the one the paper uses. The anchor weights
are a softmax of the logits -d_j(x) / τ_j^β (β = 2 for power, 1 otherwise)
over the top_k anchors with the largest logits, and zero for the others: a
sparse partition of unity (Eq. 4).

Anchors start on a sunflower lattice with r_j = w_j = α_j = 0, so every mode
starts as the same Voronoi partition.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_MODES = ("voronoi", "apollonius", "power", "apollonius_aniso")


class MotionSAD(nn.Module):
    """Anchor weights φ_j(x) of M anchors.

    Parameters: mu [M, 2] (p_j) and log_sigma [M] (log τ_j) in every mode;
    weight [M] (w_j) for power; radius [M] (r_j) for apollonius and
    apollonius_aniso; log_aspect [M] (α_j) and aniso_dir [M, 2] (û_j,
    normalized on use) for apollonius_aniso.
    """

    def __init__(
        self,
        num_anchors: int = 128,
        init_sigma: float = 0.25,
        top_k: int | None = None,
        coord_range: float = 1.0,
        mode: str = "voronoi",
    ):
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.M = num_anchors
        self.top_k = top_k
        self.coord_range = coord_range
        self.mode = mode

        anchor_init = _sunflower_grid(num_anchors, coord_range)
        self.mu = nn.Parameter(anchor_init)                                     # [M, 2]
        self.log_sigma = nn.Parameter(
            torch.full((num_anchors,), math.log(init_sigma)))                   # [M]

        # weight, radius and log_aspect start at zero: every mode starts as plain Voronoi.
        if mode == "power":
            self.weight = nn.Parameter(torch.zeros(num_anchors))                # [M]
        elif mode == "apollonius":
            self.radius = nn.Parameter(torch.zeros(num_anchors))                # [M]
        elif mode == "apollonius_aniso":
            self.radius = nn.Parameter(torch.zeros(num_anchors))                # [M]
            self.log_aspect = nn.Parameter(torch.zeros(num_anchors))            # [M]
            phi0 = torch.rand(num_anchors) * 2 * math.pi
            self.aniso_dir = nn.Parameter(
                torch.stack([torch.cos(phi0), torch.sin(phi0)], dim=-1))        # [M, 2]

    @property
    def num_anchors(self) -> int:
        return self.M

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """Anchor weights at the points x [N, 2] in [-1, 1]^2: [N, M], each row summing
        to 1, with at most top_k nonzeros if top_k is set."""
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)                            # [N, M, 2]
        mode = self.mode

        if mode == "voronoi":
            # logit = -‖x - p_j‖ / τ_j
            dist = (diff * diff).sum(-1).clamp_min(1e-12).sqrt()
            inv_s = torch.exp(-self.log_sigma).unsqueeze(0)
            logits = -dist * inv_s

        elif mode == "apollonius":
            # logit = (r_j - ‖x - p_j‖) / τ_j
            dist = (diff * diff).sum(-1).clamp_min(1e-12).sqrt()
            inv_s = torch.exp(-self.log_sigma).unsqueeze(0)
            logits = (self.radius.unsqueeze(0) - dist) * inv_s

        elif mode == "power":
            # logit = (w_j - ‖x - p_j‖²) / τ_j²
            sq = (diff * diff).sum(-1)
            inv2s2 = torch.exp(-2.0 * self.log_sigma).unsqueeze(0)
            logits = (self.weight.unsqueeze(0) - sq) * inv2s2

        elif mode == "apollonius_aniso":
            # logit = (r_j - ‖x - p_j‖_G) / τ_j,  ‖v‖_G² = e^α (û·v)² + e^(-α) (‖v‖² - (û·v)²)
            u = self.aniso_dir / (self.aniso_dir.norm(dim=-1, keepdim=True) + 1e-8)
            proj = (diff * u.unsqueeze(0)).sum(-1)                              # [N, M]
            sq = (diff * diff).sum(-1)
            perp_sq = (sq - proj * proj).clamp_min(0.0)
            ea = self.log_aspect.exp().unsqueeze(0)
            sq_G = ea * proj * proj + (1.0 / ea) * perp_sq
            dist_G = sq_G.clamp_min(1e-12).sqrt()
            inv_s = torch.exp(-self.log_sigma).unsqueeze(0)
            logits = (self.radius.unsqueeze(0) - dist_G) * inv_s

        else:
            raise RuntimeError(f"unreachable: mode={self.mode}")

        if self.top_k is not None and self.top_k < self.M:
            topv, topi = logits.topk(self.top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(1, topi, topv)
            logits = masked
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, temporal_coeffs: torch.Tensor) -> torch.Tensor:
        """Σ_j φ_j(x) a_j at x [N, 2] for coefficients temporal_coeffs [M, D]: [N, D]."""
        phi = self.basis(x)
        return phi @ temporal_coeffs


def _sunflower_grid(n: int, r: float) -> torch.Tensor:
    """[n, 2] points of a sunflower (Fibonacci) lattice on the disk of radius r."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    theta = 2.0 * math.pi * i * phi
    rr = r * torch.sqrt(i / n)
    return torch.stack([rr * torch.cos(theta), rr * torch.sin(theta)], dim=-1)
