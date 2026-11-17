"""Optional motion regularizers of the ROM.

    movement_penalty      mean over the sites of ‖Δs_i(t)‖²
    temporal_smoothness   mean squared second difference of a_j(t) on a uniform 32-point time grid
    arap                  mean of (‖s_i(t) - s_l(t)‖ - ‖s_i - s_l‖)² over the canonical 6-NN graph
    spatial_smoothness    mean squared difference of Σ_j φ_j(x) a_j between nearby points

The first three are the penalties of the paper's Eq. 11. The paper's
configuration sets every weight to 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def movement_penalty(delta_xyz: torch.Tensor) -> torch.Tensor:
    """Mean over the sites of the squared displacement ‖Δ_i‖². delta_xyz: [N, 2]."""
    return (delta_xyz * delta_xyz).sum(-1).mean()


def temporal_smoothness(coef_module, order: int = 2, num_samples: int = 32) -> torch.Tensor:
    """Mean squared finite difference (first or second order) of a_j(t), sampled at
    `num_samples` uniform times in [0, 1]."""
    device = coef_module.t.device
    ts = torch.linspace(0.0, 1.0, num_samples, device=device)
    curves = coef_module(ts)           # [num_samples, M, out_dim]
    if order == 1:
        d = curves[1:] - curves[:-1]
    else:
        d = curves[2:] - 2 * curves[1:-1] + curves[:-2]
    return (d * d).mean()


def spatial_smoothness(anchor_module, temporal_coeffs: torch.Tensor,
                       num_samples: int = 2048) -> torch.Tensor:
    """Mean squared difference of u(x) = Σ_j φ_j(x) a_j between random x in [-1, 1]^2 and
    x + ε, ε ~ N(0, 0.01²). temporal_coeffs: [M, D]."""
    device = anchor_module.mu.device
    x = (torch.rand(num_samples, 2, device=device) * 2.0 - 1.0)
    eps = 1e-2 * torch.randn_like(x)
    u1 = anchor_module.basis(x) @ temporal_coeffs
    u2 = anchor_module.basis(x + eps) @ temporal_coeffs
    return ((u1 - u2) ** 2).mean()


def build_knn_graph(xyz: torch.Tensor, k: int = 6) -> torch.Tensor:
    """[N, k] indices of each site's k nearest neighbors among xyz [N, 2] (itself excluded)."""
    with torch.no_grad():
        d = torch.cdist(xyz, xyz)
        d.fill_diagonal_(float("inf"))
        _, idx = d.topk(k, dim=-1, largest=False)
    return idx


def arap(xyz_canon: torch.Tensor, xyz_deformed: torch.Tensor,
         knn_idx: torch.Tensor) -> torch.Tensor:
    """As-rigid-as-possible penalty: mean squared change of the distance between each
    site and its canonical neighbors knn_idx [N, k]."""
    nb_c = xyz_canon[knn_idx]                              # [N, k, 2]
    nb_d = xyz_deformed[knn_idx]
    dc = (xyz_canon.unsqueeze(1) - nb_c).norm(dim=-1)      # canonical distances
    dd = (xyz_deformed.unsqueeze(1) - nb_d).norm(dim=-1)   # deformed distances
    return F.mse_loss(dd, dc)
