"""G2: mesh-level verification with the S4R evaluator convention (piecewise FCL score).

Penetration is the count of body pairs whose score is negative: the `distance()` gap when the
surfaces are separated, minus the largest `collide()` contact depth when they intersect. Only
the sign of the score is exact; the magnitude of a negative score is FCL's triangle-clip
length, not a penetration depth, and is reported as `max_pen` for continuity only.

Two cases FCL cannot see are handled here: (1) a body wholly inside another (surfaces do not
intersect, `distance()` is positive) is detected with a point-in-mesh test on nested AABBs and
counted as penetrating; (2) the contact normal of a penetrating pair is oriented by probing
(which translation of body j increases the score), never by the centre-to-centre heuristic
that fails next to concave fixed geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.model import Scene, Body
from ..errors import GeometryQueryError

PEN_EPS = 1e-6     # |signed| below this is floating-point touching, not penetration
TOUCH_EPS = 5e-5   # a pair that separates under a 0.05 mm nudge is touching, not penetrating (FCL coplanar-face artifact)


def _fcl_model(verts: np.ndarray, faces: np.ndarray):
    import fcl
    m = fcl.BVHModel()
    m.beginModel(len(verts), len(faces))
    m.addSubModel(verts.astype(np.float64), faces.astype(np.int32))
    m.endModel()
    return fcl.CollisionObject(m, fcl.Transform())


def _score(oi, oj):
    """(signed, normal_or_None, wi, wj): FCL gap when separated, minus the deepest contact depth
    when intersecting; the returned normal is FCL's raw direction (unoriented)."""
    import fcl
    req = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
    res = fcl.DistanceResult()
    d = float(fcl.distance(oi, oj, req, res))
    if d > 0.0:
        wi = np.asarray(res.nearest_points[0], dtype=np.float64)
        wj = np.asarray(res.nearest_points[1], dtype=np.float64)
        raw = wj - wi
        nr = float(np.linalg.norm(raw))
        if nr < 1e-9:                      # numerically touching: no direction from the witnesses
            return 0.0, None, None, None
        return d, raw / nr, wi, wj
    creq = fcl.CollisionRequest(num_max_contacts=32, enable_contact=True)
    cres = fcl.CollisionResult()
    fcl.collide(oi, oj, creq, cres)
    if not cres.is_collision or not cres.contacts:
        return 0.0, None, None, None
    c = max(cres.contacts, key=lambda c: c.penetration_depth)
    nrm = np.asarray(c.normal, dtype=np.float64)
    nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
    w = np.asarray(c.pos, dtype=np.float64)
    return -float(c.penetration_depth), nrm, w, w.copy()


def _touch_normal(oi, oj, ci, cj, up, eps=1e-3):
    """Direction for a touching pair that FCL reports without contact points: if lifting body j
    along `up` by eps opens a gap of about eps, the contact is a resting contact (normal along
    up, signed by which body is higher); otherwise the centre line."""
    t0 = np.asarray(oj.getTranslation(), dtype=np.float64)
    oj.setTranslation(t0 + eps * up); d_up = _score(oi, oj)[0]
    oj.setTranslation(t0 - eps * up); d_dn = _score(oi, oj)[0]
    oj.setTranslation(t0)
    if d_up > 0.5 * eps and d_dn <= 0.0:
        return up.copy()
    if d_dn > 0.5 * eps and d_up <= 0.0:
        return -up.copy()
    d = cj - ci
    return d / max(np.linalg.norm(d), 1e-12)


def _contained(inner_verts, outer_mesh):
    """True when the inner body's lowest vertex lies inside the outer mesh (the surfaces are
    apart, so one surface point decides; the mean of a non-convex body's vertices may fall in
    a hole and is not used)."""
    pts = inner_verts[[np.argmin(inner_verts[:, 2])]]
    try:
        return bool(np.all(outer_mesh.contains(pts)))
    except Exception as exc:  # a failed query must not count as absence
        raise GeometryQueryError(f"point-in-mesh query failed: {type(exc).__name__}: {exc}") from exc


def _contained_touching(inner_verts, outer_mesh):
    """Containment of a body whose surface touches the other body's surface: points slightly
    inside the inner body (towards its box centre) must lie inside the outer solid. A body
    resting in a cavity fails this (those points are in the cavity), a body flush against the
    inside of a solid passes."""
    c = 0.5 * (inner_verts.min(axis=0) + inner_verts.max(axis=0))
    step = max(1, len(inner_verts) // 32)
    pts = c + (inner_verts[::step] - c) * (1.0 - 1e-3)
    try:
        return bool(np.all(outer_mesh.contains(pts)))
    except Exception as exc:
        raise GeometryQueryError(f"point-in-mesh query failed: {type(exc).__name__}: {exc}") from exc


class PairScore(tuple):
    """(i, j, signed, normal, wit_i, wit_j) with a `contained` attribute: True when the score
    comes from the point-in-mesh test of a body wholly inside another."""
    contained = False


def pair_signed_distances(bodies: list, scale: float = 1.0, prefilter: float | None = None,
                          skip_fixed_pairs: bool = True, with_witness: bool = False,
                          containment: bool = True, probe_eps: float = 1e-3):
    """Return a list of (i, j, signed, normal, wit_i, wit_j) over candidate pairs.

    signed > 0: FCL boundary gap; signed < 0: minus FCL's deepest contact depth (sign exact,
    magnitude not a depth), or minus (gap + inner extent) for a body wholly inside another.
    normal points from body i toward body j in the sense that moving j along +normal separates
    the pair (checked by a probe for penetrating pairs).
    """
    import fcl
    n = len(bodies)
    verts = [b.world_vertices(scale) for b in bodies]
    lo = [v.min(axis=0) for v in verts]
    hi = [v.max(axis=0) for v in verts]
    objs = [_fcl_model(verts[k], bodies[k].faces) for k in range(n)]
    tmesh = {}

    def mesh_of(k):
        if k not in tmesh:
            import trimesh
            tmesh[k] = trimesh.Trimesh(verts[k], bodies[k].faces, process=False)
        return tmesh[k]

    pieces = {}

    def pieces_of(k):
        """Vertex index arrays of body k's connected pieces (one array for a connected mesh)."""
        if k not in pieces:
            import trimesh
            f = np.asarray(bodies[k].faces)
            if len(f) == 0:
                pieces[k] = []
            else:
                edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
                labels = trimesh.graph.connected_component_labels(edges, node_count=len(verts[k]))
                used = np.zeros(len(verts[k]), dtype=bool); used[f.ravel()] = True
                pieces[k] = [np.nonzero((labels == lab) & used)[0] for lab in np.unique(labels[used])]
        return pieces[k]

    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if skip_fixed_pairs and bodies[i].fixed and bodies[j].fixed:
                continue
            gap = np.maximum(lo[i] - hi[j], lo[j] - hi[i])
            thr = prefilter if prefilter is not None else 0.0
            if float(gap.max()) > thr:
                continue
            ci, cj = bodies[i].center, bodies[j].center
            signed, nrm, wi, wj = _score(objs[i], objs[j])
            inside = False
            if signed > 0.0:
                # separated surfaces: FCL's witness direction is exact; but one body may be
                # wholly inside the other (nested AABBs), which FCL reports as a positive gap
                if containment:
                    inner = outer = None
                    if np.all(lo[i] >= lo[j] - 1e-9) and np.all(hi[i] <= hi[j] + 1e-9):
                        inner, outer = i, j
                    elif np.all(lo[j] >= lo[i] - 1e-9) and np.all(hi[j] <= hi[i] + 1e-9):
                        inner, outer = j, i
                    if inner is not None and _contained(verts[inner], mesh_of(outer)):
                        d = cj - ci
                        nrm = d / max(np.linalg.norm(d), 1e-12)
                        ext = bodies[inner].extent_along(nrm if inner == j else -nrm, scale)
                        signed = -(signed + ext)
                        wi = wj = verts[inner].mean(0)
                        inside = True
                    else:
                        # a body made of several closed pieces can have one piece inside the other
                        # body while the bodies' AABBs do not nest: test the pieces one by one
                        for a, b in ((j, i), (i, j)):
                            if inside or len(pieces_of(a)) < 2:
                                continue
                            for idx in pieces_of(a):
                                cv = verts[a][idx]
                                if (np.all(cv.min(axis=0) >= lo[b] - 1e-9) and np.all(cv.max(axis=0) <= hi[b] + 1e-9)
                                        and _contained(cv, mesh_of(b))):
                                    cen = cv.mean(0)
                                    d = (cen - ci) if a == j else (cj - cen)
                                    nrm = d / max(np.linalg.norm(d), 1e-12)
                                    proj = cv @ nrm
                                    signed = -(signed + float(proj.max() - proj.min()))
                                    wi = wj = cen
                                    inner = a
                                    inside = True
                                    break
            elif nrm is None:
                # touching without contact points: probe whether it is a resting contact
                wi = 0.5 * (ci + cj); wj = wi.copy()
                nrm = _touch_normal(objs[i], objs[j], ci, cj, np.array([0.0, 0.0, 1.0]), eps=probe_eps)
                inner = None
            else:
                # intersecting surfaces: FCL's contact normal has no reliable orientation;
                # keep the sign for which moving body j along +normal separates the pair
                t0 = np.asarray(objs[j].getTranslation(), dtype=np.float64)
                objs[j].setTranslation(t0 + probe_eps * nrm); s_plus = _score(objs[i], objs[j])[0]
                objs[j].setTranslation(t0 - probe_eps * nrm); s_minus = _score(objs[i], objs[j])[0]
                objs[j].setTranslation(t0)
                if s_minus > s_plus:
                    nrm = -nrm
                elif s_minus == s_plus and np.dot(nrm, cj - ci) < 0.0:
                    nrm = -nrm
                # coplanar faces that merely touch make FCL report a clip-length "depth": if a
                # TOUCH_EPS nudge in some direction already clears every contact, the pair is touching
                for probe in (nrm, [0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]):
                    objs[j].setTranslation(t0 + TOUCH_EPS * np.asarray(probe, dtype=np.float64))
                    if _score(objs[i], objs[j])[0] >= 0.0:
                        signed = 0.0
                        break
                objs[j].setTranslation(t0)
                inner = None
            if containment and signed == 0.0 and not inside:
                # touching surfaces with nested boxes: a body flush against the inside of a solid
                # (a face of the inner coincides with a face of the outer) is inside it
                inner = outer = None
                if np.all(lo[i] >= lo[j] - 1e-9) and np.all(hi[i] <= hi[j] + 1e-9):
                    inner, outer = i, j
                elif np.all(lo[j] >= lo[i] - 1e-9) and np.all(hi[j] <= hi[i] + 1e-9):
                    inner, outer = j, i
                if inner is not None and _contained_touching(verts[inner], mesh_of(outer)):
                    d = cj - ci
                    nrm = d / max(np.linalg.norm(d), 1e-12)
                    signed = -bodies[inner].extent_along(nrm if inner == j else -nrm, scale)
                    wi = wj = verts[inner].mean(0)
                    inside = True
                else:
                    inner = None
            pair = PairScore((i, j, signed, nrm, wi, wj))
            pair.contained = inside
            pair.inner_index = inner if inside else None
            out.append(pair)
    return out


@dataclass
class VerifyReport:
    pen_pairs: int
    max_pen: float                                  # magnitude of the most negative score (not a depth)
    min_signed: float
    pairs: list = field(default_factory=list)      # (name_i, name_j, signed)
    support: dict = field(default_factory=dict)    # name -> gap to declared support (m)
    floating: list = field(default_factory=list)
    sunk: list = field(default_factory=list)
    contained: list = field(default_factory=list)  # (inner, outer) pairs counted as penetrating

    @property
    def ok(self) -> bool:
        return self.pen_pairs == 0 and not self.floating and not self.sunk

    def summary(self) -> str:
        s = f"pen={self.pen_pairs} maxScore={self.max_pen:.4g} minGap={self.min_signed:.4g}"
        if self.contained: s += f" contained={self.contained}"
        if self.floating: s += f" floating={self.floating}"
        if self.sunk: s += f" sunk={self.sunk}"
        return s


def verify_scene(scene: Scene, supports: dict | None = None, gap_tol: float = 2e-3,
                 sink_tol: float = 1e-3) -> VerifyReport:
    """G2 over all pairs (fixed-fixed skipped). `supports` maps a body name to the
    height its lowest point should rest at."""
    # keep touching pairs: a strict AABB gap > 0 test drops bodies resting exactly on a surface
    pairs = pair_signed_distances(scene.bodies, prefilter=max(gap_tol, 1e-3))
    pen = [(scene.bodies[i].name, scene.bodies[j].name, s) for i, j, s, *_ in pairs if s < -PEN_EPS]
    max_pen = max((-s for _, _, s in pen), default=0.0)
    min_signed = min((s for _, _, s, *_ in pairs), default=float("inf"))
    rep = VerifyReport(pen_pairs=len(pen), max_pen=max_pen, min_signed=min_signed, pairs=pen)
    # containment is reported separately as well (it is inside `pen` already)
    for p in pairs:
        if getattr(p, "contained", False):
            i, j = p[0], p[1]
            inner = getattr(p, "inner_index", None)
            if inner is None:
                bi, bj = scene.bodies[i], scene.bodies[j]
                vi, vj = bi.world_aabb(), bj.world_aabb()
                inner = i if np.all(vi[0] >= vj[0] - 1e-9) and np.all(vi[1] <= vj[1] + 1e-9) else j
            outer = j if inner == i else i
            rep.contained.append((scene.bodies[inner].name, scene.bodies[outer].name))
    # A free body is supported when some contact holds it from below: a pair within gap_tol whose
    # normal has a vertical component pointing up into the body (a wall contact does not count).
    up = scene.up
    held = {b.name: False for b in scene.free()}
    touch = {b.name: float("inf") for b in scene.free()}
    for i, j, s_, nrm, *_ in pairs:
        for k, sign in ((i, -1.0), (j, +1.0)):
            nm = scene.bodies[k].name
            if nm not in touch:
                continue
            touch[nm] = min(touch[nm], s_)
            if s_ <= gap_tol and float(np.dot(sign * nrm, up)) > 0.5:
                held[nm] = True
    for name, g in touch.items():
        rep.support[name] = g
        if not held[name]:
            rep.floating.append(name)
    for name, h0 in (supports or {}).items():
        b = scene[name]
        low = float((b.world_vertices() @ scene.up).min())
        if low - h0 < -sink_tol:
            rep.sunk.append(name)
    return rep
