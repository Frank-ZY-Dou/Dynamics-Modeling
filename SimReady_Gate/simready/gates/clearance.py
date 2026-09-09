"""G2 clearance: the program's `min_gap` compared pair by pair, with explicit coverage.

`verify_scene` answers "which pairs penetrate"; it keeps no per-pair distance for separated
pairs, so a positive clearance threshold needs its own pass. Every pair of bodies that is not
fixed-fixed is either measured by the mesh evaluator (when its bounding boxes come within the
threshold) or certified distant by the axis-aligned bound, which is a lower bound on the true
distance. A body and its declared support rest at zero gap by design, so such pairs are exempt
from the clearance requirement only; their penetration is still counted by `verify_scene`.
"""
from __future__ import annotations

import math

import numpy as np

from ..errors import GeometryQueryError
from .verify import pair_signed_distances


def check_clearance(scene, min_gap: float, support_of: dict | None = None) -> dict:
    if isinstance(min_gap, bool) or not isinstance(min_gap, (int, float)) or not math.isfinite(min_gap) or min_gap < 0:
        raise ValueError(f"min_gap must be a finite non-negative number, got {min_gap!r}")
    bodies = scene.bodies
    names = [b.name for b in bodies]
    exempt = {frozenset((a, s)) for a, s in (support_of or {}).items() if a != s}
    slack = 1e-3                                  # measure a little beyond the threshold
    measured = {}
    for i, j, s, *_ in pair_signed_distances(bodies, prefilter=min_gap + slack):
        if not math.isfinite(s):
            raise GeometryQueryError(f"non-finite distance for {names[i]}, {names[j]}")
        measured[(i, j)] = float(s)
    lo = [b.world_aabb()[0] for b in bodies]
    hi = [b.world_aabb()[1] for b in bodies]
    checked = n_measured = n_distant = n_exempt = 0
    violations = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if bodies[i].fixed and bodies[j].fixed:
                continue
            if frozenset((names[i], names[j])) in exempt:
                n_exempt += 1
                continue
            checked += 1
            if (i, j) in measured:
                n_measured += 1
                s = measured[(i, j)]
                if s < min_gap - 1e-9:
                    violations.append({"a": names[i], "b": names[j], "gap_m": s})
            else:
                # not a candidate: some axis separates the boxes by more than min_gap + slack
                axis_gap = float(np.maximum(lo[i] - hi[j], lo[j] - hi[i]).max())
                if axis_gap <= min_gap:
                    raise GeometryQueryError(f"pair {names[i]}, {names[j]} neither measured nor certified distant")
                n_distant += 1
    return {"pass": not violations, "min_gap_m": float(min_gap), "pairs_checked": checked,
            "pairs_measured": n_measured, "pairs_certified_distant": n_distant, "pairs_exempt_support": n_exempt,
            "violations": violations}
