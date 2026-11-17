"""Exact top-8 candidate buffers for the SAD renderer, built from the sites.

For every pixel x the builder finds the 8 sites with the smallest candidate key

    key_i(x) = max(exp(log_tau_i), 1e-4)
               * (sqrt(max(d2a_i(x) * inv_sq, 1e-8)) - radius_i * sqrt(inv_sq)),

the quantity that SAD's VPT update minimizes (jfa.cu, candidatePackedDistance).
d2a_i is the squared anisotropic distance to site i, and x = (px, py) in
integer pixel coordinates, with no +0.5, as in SAD's kernels. Ties go to the
smaller site index.

Modes:
    exact   (default) bins each site into the tiles where it can beat a bound on the
            tile's 8th-smallest key, and rescans against all sites the pixels that
            the bound does not certify. Equal to brute, without host synchronization.
    window  scans only the cells within radius_px of each tile. Not exact: thin,
            low-tau sites reach hundreds of pixels along their axis.
    brute   every pixel scans every site. For validation.

The output has SAD's layout: two contiguous int32 [H*W, 4] tensors, row-major
pixels, slots sorted by ascending key, unused slots -1.

    from dsad.exact_candidates import build_candidates
    cand0, cand1 = build_candidates(sites, W, H)          # sites: packed [N, 10] fp32, CUDA
    img = torch.ops.sad.render_sad(cand0, cand1, sites, inv_sq, W, H)
    c0, c1 = build_candidates(gop_sites, W, H)            # [T, N, 10] gives [T, H*W, 4] each

The decoders use GoPBuilder. The kernels are in csrc/exact_candidates_cuda.cu.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import torch

try:
    from . import _paths
except ImportError:
    import _paths

_EXT = None
_CSRC = Path(__file__).resolve().parent / "csrc"
_MODES = {"exact": 0, "window": 1, "brute": 1}


def _ext():
    """JIT-compile (once per TORCH_EXTENSIONS_DIR) and load the CUDA extension."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        # Repository-local extension cache, and the interpreter's ninja on PATH.
        _paths.setup_env()
        _EXT = load(
            name="dsad_exact_candidates",
            sources=[str(_CSRC / "exact_candidates.cpp"),
                     str(_CSRC / "exact_candidates_cuda.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],   # no fast-math: keys must match SAD's IEEE expf/sqrtf
            verbose=bool(int(os.environ.get("DSAD_EXT_VERBOSE", "0"))),
        )
    return _EXT


def _ordered_to_float(bits: int) -> float:
    """Inverse of the kernel's order-preserving map from float to uint32."""
    u = bits & 0xFFFFFFFF
    b = (u & 0x7FFFFFFF) if (u & 0x80000000) else (~u & 0xFFFFFFFF)
    return struct.unpack("<f", struct.pack("<I", b))[0]


def build_candidates(sites: torch.Tensor, W: int, H: int, *, inv_sq: float | None = None,
                     mode: str = "exact", radius_px: int | None = None, cell_px: int = 16,
                     tile_px: int = 16, key_fp16: bool = True, cap_quantile: float = 0.998,
                     entries_per_site: int = 256, coop_fallback: bool = True,
                     return_info: bool = False):
    """Top-8 candidate buffers of one frame, or of F frames in one batched build.

    Args:
        sites:     packed SAD sites [N, 10] float32 on CUDA (the layout of
                   SADAppearance_V3._pack_sites_grad), or [F, N, 10] for F frames.
        W, H:      frame size in pixels.
        inv_sq:    1 / max(W, H)^2 by default (SADAppearance_V3.inv_sq).
        mode:      "exact", "window" or "brute" (module docstring).
        radius_px: "window": search radius in pixels (required).
                   "exact": radius of the window pass that bounds the keys
                   (default cell_px).
        cell_px:   cell size of the site grid of the window scans.
        tile_px:   tile size, 8 or 16 (one CUDA block per tile).
        key_fp16:  rank with half-precision site parameters, as VPT does after
                   pack_candidate_sites (default); False ranks with the fp32 key.
                   Rendering always uses the fp32 sites.
        cap_quantile, entries_per_site, coop_fallback: "exact" only, and speed
                   only: the candidates do not depend on them. Tile bounds above
                   the cap_quantile quantile are capped. entries_per_site is the
                   preallocated tile-list capacity per site (at 16-px tiles).
                   Pixels left uncertified by the cap, and tiles whose list did
                   not fit, are rescanned against all sites. coop_fallback=True
                   rescans by whole warps in tiles with few such pixels; False
                   uses one thread per pixel everywhere.
        return_info: also return the build statistics, a dict (a list of dicts
                   for batched input). Synchronizes with the host.

    Returns:
        (cand0, cand1), int32 [H*W, 4] each ([F, H*W, 4] for batched input), or
        (cand0, cand1, info).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    if inv_sq is None:
        inv_sq = 1.0 / float(max(W, H) ** 2)
    if mode == "brute":
        radius_px = int(max(W, H))
    elif radius_px is None:
        if mode == "window":
            raise ValueError("mode='window' needs radius_px")
        radius_px = int(cell_px)
    single = sites.dim() == 2
    batch = sites.detach().float().contiguous()
    if single:
        batch = batch.unsqueeze(0)
    cand0, cand1, stats = _ext().build_candidates(
        batch, int(W), int(H), float(inv_sq), _MODES[mode], int(radius_px), int(cell_px),
        int(tile_px), bool(key_fp16), float(cap_quantile), int(entries_per_site), bool(coop_fallback))
    # The build runs on PyTorch's current stream, but SAD's ops always use the default
    # stream: make it wait, so that a build on a side stream is complete before they read.
    current = torch.cuda.current_stream(batch.device)
    default = torch.cuda.default_stream(batch.device)
    if current != default:
        default.wait_stream(current)
    if single:
        cand0, cand1 = cand0[0], cand1[0]
    if return_info:
        infos = [{} for _ in range(batch.shape[0])]
        if _MODES[mode] == 0:
            infos = [{"list_entries": s[0],                      # (site, tile) pairs
                      "max_tile_bound": _ordered_to_float(s[1]),  # largest tile bound
                      "cap": _ordered_to_float(s[2]),
                      "capped_tiles": s[3],
                      "fallback_pixels": s[4],                   # uncertified pixels, rescanned
                      "overflow_tiles": s[5]}                    # tiles over capacity, rescanned
                     for s in stats.cpu().tolist()]
        return cand0, cand1, (infos[0] if single else infos)
    return cand0, cand1


# GoPBuilder's defaults: the two cap quantiles, and the number of rescanned pixels above which a
# pass uses the higher one. Speed only: every setting gives the same candidates.
GOP_Q_LOW, GOP_Q_HIGH, GOP_LIMIT = 0.98, 0.998, 64
# Frames per launch of a batched build. The workspace and the candidate buffers of a launch grow
# with its frames, so this bounds the memory. A frame's candidates do not depend on it.
GOP_MAX_FRAMES = 8


class GoPBuilder:
    """The exact builder with its cap quantile chosen once per GoP pass. Speed only: every
    cap gives the same candidates.

    The first build after reset() runs at q_low and reads how many pixels it rescanned
    against all sites (one host synchronization per pass). If that exceeds `limit`, the rest
    of the pass uses q_high, else q_low. A batched build ([F, N, 10]) probes frame 0 alone,
    builds the other frames in launches of at most `max_frames` frames, and returns two lists
    of F per-frame [H*W, 4] tensors. With q_low == q_high the cap is fixed: no probe and no
    host synchronization.

    Call reset() at the start of every pass over a GoP (FusedStep.invalidate_basis() does).
    """

    def __init__(self, q_low: float | None = None, q_high: float | None = None, limit: int | None = None,
                 entries_per_site: int = 256, coop_fallback: bool = True, max_frames: int | None = None):
        self.q_low = GOP_Q_LOW if q_low is None else float(q_low)
        self.q_high = GOP_Q_HIGH if q_high is None else float(q_high)
        self.limit = GOP_LIMIT if limit is None else int(limit)
        self.entries_per_site = int(entries_per_site)
        self.coop_fallback = bool(coop_fallback)
        self.max_frames = GOP_MAX_FRAMES if max_frames is None else int(max_frames)
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.q = None            # the pass's cap, once decided
        self.probe_fallback = None   # pixels the pass's probe rescanned against all sites
        self.probes = 0          # passes probed since construction

    def reset(self) -> None:
        self.q = None
        self.probe_fallback = None

    def _batched(self, sites: torch.Tensor, W: int, H: int, kw: dict):
        """Frames [F, N, 10] at the pass's cap, at most max_frames per launch: two lists of F tensors."""
        c0s, c1s = [], []
        for a in range(0, sites.shape[0], self.max_frames):
            c0, c1 = build_candidates(sites[a:a + self.max_frames], W, H, cap_quantile=self.q, **kw)
            c0s += c0.unbind(0)
            c1s += c1.unbind(0)
        return c0s, c1s

    def __call__(self, sites: torch.Tensor, W: int, H: int, *, inv_sq: float | None = None, **kw):
        kw = dict(kw, inv_sq=inv_sq, entries_per_site=self.entries_per_site, coop_fallback=self.coop_fallback)
        if self.q is None and self.q_low == self.q_high:   # a fixed cap: nothing to probe
            self.q = self.q_low
        if self.q is not None:
            if sites.dim() == 3:
                return self._batched(sites, W, H, kw)
            return build_candidates(sites, W, H, cap_quantile=self.q, **kw)
        self.probes += 1
        first = sites if sites.dim() == 2 else sites[0]
        c0, c1, info = build_candidates(first, W, H, cap_quantile=self.q_low, return_info=True, **kw)
        self.probe_fallback = int(info["fallback_pixels"])
        self.q = self.q_high if self.probe_fallback > self.limit else self.q_low
        if sites.dim() == 2:
            return c0, c1
        r0, r1 = self._batched(sites[1:], W, H, kw)
        return [c0, *r0], [c1, *r1]


def candidates_to_sets(cand0: torch.Tensor, cand1: torch.Tensor) -> torch.Tensor:
    """[..., H*W, 8] int64 candidate indices, sorted per pixel (for set comparisons)."""
    return torch.cat([cand0, cand1], dim=-1).long().sort(dim=-1).values


__all__ = ["build_candidates", "GoPBuilder", "candidates_to_sets"]
