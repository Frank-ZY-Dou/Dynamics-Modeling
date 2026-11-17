"""Temporal coefficients a_j(t) of the ROM (the paper's Eq. 6).

Each anchor j has K knots with times t_jk, bandwidths ρ_jk and values v_jk in R^d:

    a_j(t) = Σ_k ω_jk(t) v_jk,    ω_jk(t) = softmax_k( -(t - t_jk)² / (2 ρ_jk²) )

Knot times start uniform on [0, 1] and values at zero, so a_j(t) = 0 at
initialization (the identity deformation).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal1DSAD(nn.Module):
    """Knot-based coefficient curves a_j(t) of M anchors.

    Parameters: t [M, K] knot times, log_r [M, K] log bandwidths, v [M, K, out_dim]
    knot values. top_k, if set, restricts the softmax to the top_k knots with the
    largest logits (default: all K, as in the paper).
    """

    def __init__(
        self,
        num_anchors: int,
        num_knots: int = 8,
        out_dim: int = 2,
        init_radius: float = 0.15,
        top_k: int | None = None,
    ):
        super().__init__()
        self.M = num_anchors            # anchors (ROM rank)
        self.K = num_knots              # knots per anchor
        self.out_dim = out_dim
        self.top_k = top_k

        # Knot times start uniform on [0, 1].
        knot_times = torch.linspace(0.0, 1.0, num_knots).unsqueeze(0).repeat(num_anchors, 1)
        self.t = nn.Parameter(knot_times.clone())                           # [M, K]
        self.log_r = nn.Parameter(torch.full((num_anchors, num_knots),
                                             math.log(init_radius)))         # [M, K]
        # Zero values: the deformation starts as the identity.
        self.v = nn.Parameter(torch.zeros(num_anchors, num_knots, out_dim))  # [M, K, d]

    @property
    def num_anchors(self) -> int:
        return self.M

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """a(t) at `time` in [0, 1]: [M, out_dim] for a 0-dim tensor, [B, M, out_dim] for a
        [B] tensor."""
        scalar = time.dim() == 0
        t_q = time.view(-1, 1, 1)                         # [B, 1, 1]
        diff = t_q - self.t.unsqueeze(0)                   # [B, M, K]
        inv2r2 = torch.exp(-2.0 * self.log_r).unsqueeze(0)
        logits = -0.5 * (diff * diff) * inv2r2             # -(t - t_k)² / (2 ρ_k²)

        if self.top_k is not None and self.top_k < self.K:
            topv, topi = logits.topk(self.top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topi, topv)
            logits = masked

        w = F.softmax(logits, dim=-1)                      # [B, M, K]
        a = (w.unsqueeze(-1) * self.v.unsqueeze(0)).sum(dim=-2)  # [B, M, out_dim]
        return a[0] if scalar else a
