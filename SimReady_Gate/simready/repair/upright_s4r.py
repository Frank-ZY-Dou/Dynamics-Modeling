"""G3: S4R repair on a support plane (in-plane translation + yaw per body).

Library port of the S4R upright-on-plane variant: free bodies are scaled
about their reference centres from s_min to 1, kept resting on a declared
support height, roll/pitch driven to zero by a smooth tilt homotopy for
bodies tagged upright, and every scale increment solves one minimum-norm QP
whose rows are (a) frozen-witness contact rows from exact mesh detection and
(b) extra linear rows supplied by the DSL compiler. Fixed bodies are
obstacles with no variables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp

from ..scene.model import Body, Scene
from ..gates.verify import pair_signed_distances


def rotz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def roty(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def decompose_zyx(R: np.ndarray):
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        return math.atan2(-R[0, 1], R[1, 1]), 0.0, pitch
    roll = math.atan2(R[2, 1] / cp, R[2, 2] / cp)
    yaw = math.atan2(R[1, 0] / cp, R[0, 0] / cp)
    return yaw, roll, pitch


def tilt_schedule(s: float, s_min: float) -> float:
    x = min(max((s - s_min) / max(1.0 - s_min, 1e-9), 0.0), 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)


@dataclass
class RepairSpec:
    supports: dict                       # body name -> rest height of its lowest point
    support_of: dict = field(default_factory=dict)   # body name -> name of its support body
    upright: set = field(default_factory=set)   # bodies whose roll/pitch go to 0
    rows_fn: object = None               # callable(state) -> list[(coeffs, lo, hi)]
    prefer: dict = field(default_factory=dict)  # name -> (target_xy, weight)
    place: dict = field(default_factory=dict)   # name -> (x, y, yaw_deg|None): the drag in the shrunken scale-space
    yaw_weight: dict = field(default_factory=dict)  # name -> weight (default size^2)
    d_hat: float = 0.01
    ds_max: float = 0.05
    s_min: float = 0.05
    max_yaw_step: float = 0.08
    max_xy_step: float = 0.03
    tail_iters: int = 20


@dataclass
class State:
    """Mutable per-step state handed to the DSL rows callback."""
    scene: Scene
    free: list                 # free body indices into scene.bodies
    var: dict                  # body index -> column offset (x, y, yaw)
    xy: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    scale: float
    tilt: float

    def body_scale(self, k: int) -> float:
        return 1.0 if self.scene.bodies[k].fixed else self.scale


@dataclass
class RepairResult:
    scene: Scene
    pen_before: int
    pen_after: int
    steps: int
    displacement: np.ndarray     # per free body xy displacement
    rmsd: float
    trace: list = field(default_factory=list)


def _is_support_pair(scene: Scene, i: int, j: int, spec: RepairSpec) -> bool:
    """(free body, its declared support)."""
    bi, bj = scene.bodies[i], scene.bodies[j]
    return spec.support_of.get(bi.name) == bj.name or spec.support_of.get(bj.name) == bi.name


VERTICAL_COS = 0.866   # |n . up| above cos 30 deg: the resting contact carried by the support equality


def _rows_satisfied(st: State, spec: RepairSpec, tol: float = 1e-4) -> bool:
    if spec.rows_fn is None:
        return True
    for coeffs, lo, hi in spec.rows_fn(st):
        if lo > tol or hi < -tol:
            return False
    return True


def _pose_bodies(st: State, spec: RepairSpec):
    """Write the current continuation poses into the scene bodies (in place)."""
    sc = st.scene
    for k, b in enumerate(sc.bodies):
        if b.fixed:
            continue
        i = st.free.index(k)
        tf = st.tilt if b.name in spec.upright else 1.0
        R = rotz(st.yaw[i]) @ roty(tf * st.pitch[i]) @ rotx(tf * st.roll[i])
        b.rotation = R
        h0 = spec.supports.get(b.name)
        z = h0 + b.support_offset(sc.up, st.scale) if h0 is not None else b.center[2]
        b.center = np.array([st.xy[i, 0], st.xy[i, 1], z])


def _inplane_separation(scene: Scene, i: int, j: int, nrm: np.ndarray) -> np.ndarray:
    """Horizontal direction that separates two overlapping footprints the fastest: the axis of
    least footprint overlap, pointing from body i to body j (falls back to the centre line)."""
    bi, bj = scene.bodies[i], scene.bodies[j]
    (li, hi_), (lj, hj) = bi.world_aabb(), bj.world_aabb()
    best, best_ov = None, np.inf
    for ax in (0, 1):
        ov = min(hi_[ax], hj[ax]) - max(li[ax], lj[ax])
        if ov < best_ov:
            best_ov, best = ov, ax
    d = np.zeros(3)
    ci, cj = bi.center, bj.center
    if best is None or abs(cj[best] - ci[best]) < 1e-9:
        d[:2] = (cj - ci)[:2]
    else:
        d[best] = np.sign(cj[best] - ci[best])
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else nrm


def _footprint_overlap(scene: Scene, i: int, j: int, nrm: np.ndarray) -> float:
    """Length of the overlap of the two bodies' projections on the horizontal direction nrm."""
    pi = scene.bodies[i].world_vertices() @ nrm; pj = scene.bodies[j].world_vertices() @ nrm
    return max(0.0, float(min(pi.max(), pj.max()) - max(pi.min(), pj.min())))


def _scaled_pairs(st: State, spec: RepairSpec):
    """Contact candidates with free bodies scaled by s and fixed bodies at full size."""
    sc = st.scene
    # Temporarily scale free bodies' vertices for detection.
    saved = []
    for k in st.free:
        b = sc.bodies[k]
        saved.append(b.verts)
        b.verts = b.verts * st.scale
    try:
        # every pair that can activate within one scale step: d_hat + ds * (full-scale extents)
        pre = spec.d_hat + spec.ds_max * 2.0 * max((b.diag / max(st.scale, 1e-9) for b in sc.free()), default=0.0)
        pairs = pair_signed_distances(sc.bodies, scale=1.0, prefilter=pre)
    finally:
        for k, v in zip(st.free, saved):
            sc.bodies[k].verts = v
    return pairs


def repair_upright(scene: Scene, spec: RepairSpec, verbose: bool = False, on_step=None) -> RepairResult:
    """on_step(state): optional hook called after every accepted step with the bodies posed at the
    step's scale and tilt (for trajectory recording / visualization)."""
    up = scene.up
    free = [k for k, b in enumerate(scene.bodies) if not b.fixed]
    var = {k: 3 * i for i, k in enumerate(free)}
    n = len(free)
    xy = np.array([scene.bodies[k].center[:2] for k in free], dtype=np.float64)
    xy0 = xy.copy()
    yaw = np.zeros(n); roll = np.zeros(n); pitch = np.zeros(n)
    for i, k in enumerate(free):
        yaw[i], roll[i], pitch[i] = decompose_zyx(scene.bodies[k].rotation)
    for nm, (px, py, pyaw) in spec.place.items():        # placement targets: applied while every body is small
        k = next((kk for kk in free if scene.bodies[kk].name == nm), None)
        if k is not None:
            i = free.index(k)
            xy[i] = (px, py)
            if pyaw is not None:
                yaw[i] = np.radians(pyaw)
    st = State(scene, free, var, xy, yaw, roll, pitch, spec.s_min, 1.0)

    # Penetration before repair, at full scale and original poses.
    pen_before = sum(1 for *_, s, _n, _a, _b in [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in pair_signed_distances(scene.bodies)] if s < 0.0)

    persistent = {}      # (i, j) -> consecutive tail iterations still penetrating

    def solve_step(tail: bool):
        st.tilt = 0.0 if tail else tilt_schedule(st.scale, spec.s_min)
        _pose_bodies(st, spec)
        pairs = _scaled_pairs(st, spec)
        if tail:
            now = {(i, j) for (i, j, signed, *_) in pairs if signed < 0.0}
            for key in list(persistent):
                if key not in now:
                    del persistent[key]
            for key in now:
                persistent[key] = persistent.get(key, 0) + 1
        ds = 0.0 if tail else spec.ds_max
        rows, cols, data, lo, hi = [], [], [], [], []
        r = 0
        pen_now = 0
        for (i, j, signed, nrm, wi, wj) in pairs:
            if signed < 0.0:
                pen_now += 1
            bi, bj = scene.bodies[i], scene.bodies[j]
            ext = 0.0
            if not bi.fixed:
                ext += bi.extent_along(nrm, 1.0)
            if not bj.fixed:
                ext += bj.extent_along(nrm, 1.0)
            rhs = spec.d_hat - signed + ds * ext
            if rhs <= 0.0:
                continue
            # A body resting on its declared support meets it with a near-vertical normal; that
            # relation is carried by the support equality, not a row. Any other contact with the
            # support (a rim, a wall, an apron) keeps its row.
            if _is_support_pair(scene, i, j, spec) and abs(float(np.dot(nrm, up))) > VERTICAL_COS:
                continue
            if signed < 0.0 and abs(float(np.dot(nrm, up))) > VERTICAL_COS:
                # a penetrating pair whose FCL normal is vertical gives the in-plane QP no handle;
                # take the separating direction from the footprint overlap of the two bodies instead
                nrm = _inplane_separation(scene, i, j, nrm)
            if signed < 0.0 and persistent.get((i, j), 0) >= 3:
                # a pair the local linearization keeps oscillating on (a thin body threaded through a
                # concave one: FCL's deepest contact flips side every step): separate the footprints
                nrm = _inplane_separation(scene, i, j, nrm)
                rhs = spec.d_hat + _footprint_overlap(scene, i, j, nrm)
                wi = wj = 0.5 * (bi.center + bj.center)
            entries = []
            for body, sign, w in ((i, -1.0, wi), (j, +1.0, wj)):
                if scene.bodies[body].fixed:
                    continue
                c0 = var[body]
                lever = w - scene.bodies[body].center
                yaw_c = float(np.dot(np.cross(lever, nrm), up))
                for col, val in ((c0, sign * nrm[0]), (c0 + 1, sign * nrm[1]), (c0 + 2, sign * yaw_c)):
                    if abs(val) > 1e-9:
                        entries.append((col, val))
            if not entries:
                continue   # no in-plane handle on this pair: nothing the QP can do
            for col, val in entries:
                rows.append(r); cols.append(col); data.append(val)
            lo.append(rhs); hi.append(np.inf); r += 1
        n_contact = r
        # DSL rows.
        extra = spec.rows_fn(st) if spec.rows_fn is not None else []
        for coeffs, l, h in extra:
            any_col = False
            for (body, v), val in coeffs.items():
                if scene.bodies[body].fixed or abs(val) < 1e-12:
                    continue
                rows.append(r); cols.append(var[body] + {"x": 0, "y": 1, "yaw": 2}[v]); data.append(val); any_col = True
            if any_col:
                lo.append(l); hi.append(h); r += 1
        if r == 0:
            return 0, pen_now, 0
        # Yaw trust region.
        for i in range(n):
            rows.append(r); cols.append(3 * i + 2); data.append(1.0)
            lo.append(-spec.max_yaw_step); hi.append(spec.max_yaw_step); r += 1
        A = sp.csc_matrix((data, (rows, cols)), shape=(r, 3 * n))
        pd = []
        q = np.zeros(3 * n)
        for i, k in enumerate(free):
            b = scene.bodies[k]
            yw = spec.yaw_weight.get(b.name, max(b.extent_along(np.array([1.0, 0, 0])), b.extent_along(np.array([0, 1.0, 0]))) ** 2)
            wx = 1.0
            if b.name in spec.prefer:
                tgt, w = spec.prefer[b.name]
                wx += w
                q[3 * i:3 * i + 2] += w * (xy[i] - np.asarray(tgt))
            pd.extend([wx, wx, yw])
        P = sp.diags(pd, format="csc")
        solver = osqp.OSQP()
        solver.setup(P, q, A, np.asarray(lo), np.asarray(hi), verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                     max_iter=8000, polish=True)
        res = solver.solve()
        if res.info.status not in ("solved", "solved_inaccurate") and r > n_contact + n:
            # DSL rows conflict with the contacts at this step: keep the contacts only,
            # so inflation never proceeds with unresolved penetration; the predicates
            # report the DSL violation at the end.
            keep = [k for k in range(len(rows)) if rows[k] < n_contact or rows[k] >= r - n]
            rows2 = [rows[k] for k in keep]; cols2 = [cols[k] for k in keep]; data2 = [data[k] for k in keep]
            remap = {}
            for rr in sorted(set(rows2)):
                remap[rr] = len(remap)
            rows2 = [remap[rr] for rr in rows2]
            lo2 = [lo[rr] for rr in sorted(remap)]; hi2 = [hi[rr] for rr in sorted(remap)]
            A2 = sp.csc_matrix((data2, (rows2, cols2)), shape=(len(remap), 3 * n))
            solver = osqp.OSQP(); solver.setup(P, q, A2, np.asarray(lo2), np.asarray(hi2), verbose=False,
                                                eps_abs=1e-6, eps_rel=1e-6, max_iter=8000, polish=True)
            res = solver.solve()
            if verbose:
                print(f"[repair] DSL rows dropped at s={st.scale:.3f}: {res.info.status}")
        if res.info.status not in ("solved", "solved_inaccurate"):
            if verbose:
                print(f"[repair] QP {res.info.status} at s={st.scale:.3f} rows={r}")
            return n_contact, pen_now, -1
        step = res.x.reshape(n, 3)
        cap = spec.max_xy_step * (0.6 if tail else 1.0)
        for i in range(n):
            d = step[i, :2]
            nn = float(np.linalg.norm(d))
            if nn > cap:
                d = d * (cap / nn)
            xy[i] += d
            yaw[i] += float(step[i, 2])
        return n_contact, pen_now, 1

    trace = []
    steps = 0
    s = spec.s_min
    while s < 1.0 - 1e-12:
        st.scale = s
        n_rows, pen_now, status = solve_step(tail=False)
        trace.append((s, n_rows, pen_now))
        steps += 1
        if on_step is not None:
            _pose_bodies(st, spec); on_step(st)
        s = min(1.0, s + spec.ds_max)
    st.scale = 1.0
    st.tilt = 0.0
    pen_now = None
    idle = 0
    for _ in range(spec.tail_iters):
        n_rows, pen_now, status = solve_step(tail=True)
        steps += 1
        trace.append((1.0, n_rows, pen_now))
        idle = idle + 1 if (n_rows == 0 or status == -1) else 0
        if idle >= 3:
            if verbose:
                print("[repair] tail: no usable rows for 3 iterations, stopping")
            _pose_bodies(st, spec)
            if on_step is not None:
                on_step(st)
            break
        _pose_bodies(st, spec)          # re-pose before judging the DSL rows post-step
        if on_step is not None:
            on_step(st)
        pen_check = sum(1 for p in pair_signed_distances(scene.bodies) if p[2] < 0.0)
        pen_now = pen_check
        if pen_check == 0 and _rows_satisfied(st, spec, tol=1e-4) and status != -1:
            break
    _pose_bodies(st, spec)
    pen_after = sum(1 for p in pair_signed_distances(scene.bodies) if p[2] < 0.0)
    disp = xy - xy0
    rmsd = float(np.sqrt(np.mean(np.sum(disp ** 2, axis=1)))) if n else 0.0
    if verbose:
        print(f"[repair] pen {pen_before} -> {pen_after}, steps={steps}, rmsd={rmsd:.4f}")
    return RepairResult(scene, pen_before, pen_after, steps, disp, rmsd, trace)


def reseat_on_supports(scene: Scene, spec: RepairSpec) -> None:
    """Re-seat every free body on its declared support using its current (full-resolution)
    mesh and the support height under its FINAL position; call after repair ran on proxies."""
    for b in scene.free():
        sup = spec.support_of.get(b.name)
        if sup is None:
            continue
        from ..dsl.compile import support_height_for
        h0 = support_height_for(scene, b, scene[sup])
        spec.supports[b.name] = h0
        b.center = np.array([b.center[0], b.center[1], h0 + b.support_offset(scene.up)])


def polish_full_mesh(scene: Scene, spec: RepairSpec, iters: int = 10, verbose: bool = False, on_step=None):
    """Tail-only pass on the full meshes after a proxy repair: fixes the millimetre
    residuals the decimated proxies could not see. Returns the RepairResult."""
    from dataclasses import replace
    spec2 = replace(spec, s_min=1.0, tail_iters=iters, place={})    # targets are applied at s_min only
    return repair_upright(scene, spec2, verbose=verbose, on_step=on_step)
