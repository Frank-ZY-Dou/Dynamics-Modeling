"""Training diagnostics: parameter counts and ROM statistics for the log, as Python numbers."""
from __future__ import annotations

from typing import Dict, Any

import math
import torch


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def rom_module_breakdown(sad_deform_core) -> Dict[str, int]:
    """Parameter count of every tensor of a SADDeformCore, and their sum as "_total"."""
    out = {}
    out["anchors.mu"]       = sad_deform_core.anchors.mu.numel()
    out["anchors.log_sigma"] = sad_deform_core.anchors.log_sigma.numel()
    # Mode-specific anchor parameters (power: weight; apollonius: radius;
    # apollonius_aniso: radius, log_aspect, aniso_dir).
    for name in ("weight", "radius", "log_aspect", "aniso_dir"):
        p = getattr(sad_deform_core.anchors, name, None)
        if isinstance(p, torch.nn.Parameter):
            out[f"anchors.{name}"] = p.numel()
    out["coef_xyz.t"]       = sad_deform_core.coef_xyz.t.numel()
    out["coef_xyz.log_r"]   = sad_deform_core.coef_xyz.log_r.numel()
    out["coef_xyz.v"]       = sad_deform_core.coef_xyz.v.numel()
    if sad_deform_core.enable_opacity:
        out["coef_opacity.t"]     = sad_deform_core.coef_opacity.t.numel()
        out["coef_opacity.log_r"] = sad_deform_core.coef_opacity.log_r.numel()
        out["coef_opacity.v"]     = sad_deform_core.coef_opacity.v.numel()
    if sad_deform_core.enable_features:
        out["coef_features.t"]     = sad_deform_core.coef_features.t.numel()
        out["coef_features.log_r"] = sad_deform_core.coef_features.log_r.numel()
        out["coef_features.v"]     = sad_deform_core.coef_features.v.numel()
    out["_total"] = sum(v for k, v in out.items() if not k.startswith("_"))
    return out


@torch.no_grad()
def anchor_stats(anchors) -> Dict[str, float]:
    """Statistics of the anchor positions (mu) and temperatures (sigma)."""
    mu = anchors.mu.detach()
    sig = anchors.log_sigma.detach().exp()
    return {
        "anchors.mu.mean_x":  float(mu[:, 0].mean()),
        "anchors.mu.mean_y":  float(mu[:, 1].mean()),
        "anchors.mu.std":     float(mu.std()),
        "anchors.mu.abs_max": float(mu.abs().max()),
        "anchors.sigma.mean": float(sig.mean()),
        "anchors.sigma.std":  float(sig.std()),
        "anchors.sigma.min":  float(sig.min()),
        "anchors.sigma.max":  float(sig.max()),
    }


@torch.no_grad()
def coef_stats(coef_module, name: str) -> Dict[str, float]:
    """Statistics of a Temporal1DSAD: its knot values and bandwidths, and the curves
    a_j(t) sampled at 64 uniform times (magnitude, per-anchor energy, second difference)."""
    v = coef_module.v.detach()
    r = coef_module.log_r.detach().exp()
    ts = torch.linspace(0.0, 1.0, 64, device=v.device)
    a = coef_module(ts)            # [64, M, d]
    energy = a.pow(2).sum(dim=(0, 2)).sqrt()     # [M]: L2 norm of a_j over time and channels
    d2 = a[2:] - 2 * a[1:-1] + a[:-2]            # second difference in time
    return {
        f"{name}.v.abs_max":   float(v.abs().max()),
        f"{name}.v.rms":       float(v.pow(2).mean().sqrt()),
        f"{name}.r.mean":      float(r.mean()),
        f"{name}.r.min":       float(r.min()),
        f"{name}.a_grid.rms":  float(a.pow(2).mean().sqrt()),
        f"{name}.a_grid.abs_max":  float(a.abs().max()),
        f"{name}.energy.mean": float(energy.mean()),
        f"{name}.energy.max":  float(energy.max()),
        f"{name}.active_frac": float((energy > 1e-4).float().mean()),  # fraction of anchors that move
        f"{name}.d2.rms":      float(d2.pow(2).mean().sqrt()),
    }


@torch.no_grad()
def delta_stats(d_xyz: torch.Tensor, d_op: torch.Tensor, d_feat: torch.Tensor,
                enable_opacity: bool, enable_features: bool) -> Dict[str, float]:
    """RMS and largest magnitude of one frame's residuals."""
    out = {
        "delta.xyz.rms":     float(d_xyz.pow(2).mean().sqrt()),
        "delta.xyz.abs_max": float(d_xyz.abs().max()),
    }
    if enable_opacity:
        out["delta.op.rms"]     = float(d_op.pow(2).mean().sqrt())
        out["delta.op.abs_max"] = float(d_op.abs().max())
    if enable_features:
        out["delta.feat.rms"]     = float(d_feat.pow(2).mean().sqrt())
        out["delta.feat.abs_max"] = float(d_feat.abs().max())
    return out


@torch.no_grad()
def basis_stats(anchors, xyz: torch.Tensor) -> Dict[str, float]:
    """Statistics of the anchor weights at xyz [N, 2], averaged over the rows: the row
    sum (1 for a partition of unity), the largest weight (1: hard assignment, 1/M:
    uniform) and the effective support exp(entropy), at most the anchor top-K."""
    phi = anchors.basis(xyz)                  # [N, M]
    phi_sum = float(phi.sum(dim=-1).mean())
    phi_max = float(phi.max(dim=-1).values.mean())
    # The clamp avoids log(0).
    entropy = -(phi.clamp_min(1e-12) * phi.clamp_min(1e-12).log()).sum(dim=-1)
    eff_k = float(entropy.exp().mean())
    return {
        "basis.phi.sum":    phi_sum,
        "basis.phi.max":    phi_max,
        "basis.eff_k":      eff_k,
    }


def format_dict(d: Dict[str, Any], fmt: str = "{:.4g}") -> str:
    """'key=value' pairs on one line, floats formatted with `fmt`."""
    return "  ".join(f"{k}={fmt.format(v) if isinstance(v, float) else v}"
                     for k, v in d.items())
