"""G3: S4R repair on a support plane (in-plane translation + yaw per body).

Library port of the S4R upright-on-plane variant: free bodies are scaled
about their reference centres from s_min to 1, kept resting on a declared
support height, roll/pitch driven to zero by a smooth tilt homotopy for
bodies tagged upright, and every scale increment solves one minimum-norm QP
whose rows are (a) frozen-witness contact rows from exact mesh detection and
(b) extra linear rows supplied by the DSL compiler. Fixed bodies are
obstacles with no variables. A body under a clearance statement keeps that clearance, instead of
the program's margin, from every body but its support (the contact rows' right-hand side).

The step QP keeps the contact rows hard and the per-step translation and yaw
trust regions as bounds inside the problem; every DSL row carries a
non-negative slack with a quadratic penalty, so a statement that conflicts
with the contacts yields to them instead of making the step infeasible, and
the final predicates report what it cost. When the contacts cannot be met
inside the trust region (deep penetration, more than one step's move), the
step serves the contacts alone: they become elastic, the program rows are set
aside for that step, every body moves as far as the region allows, and the
remainder is carried into the next linearization. The scale never advances on
a failed solve.

A soft statement's rows carry its weight, which scales a slack penalty a hundred times smaller
than a requirement's: a preference gives way to a requirement and to the contacts, and it does not
hold up the end of the tail. When pairs still penetrate in the closing fifth of the tail (a program the
contacts contradict), the rest of the tail serves the contacts alone. With `project_iters`, the program's rows are first solved alone at s_min,
where every body is small enough to pass the others (the relational counterpart of a place
target: a row, a grid, a region is reached before the scale is restored).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp

from ..scene.model import Body, Scene
from ..gates.verify import pair_signed_distances

RHO_CONTACT = 1e3     # quadratic penalty on contact-row slack when a step has to relax the contacts
RHO_DSL = 1e2         # quadratic penalty on DSL-row slack (per m^2); the multiplier over it is the row's residual
RHO_SOFT = 1.0        # the same for a soft statement's rows, times its weight: a preference gives way to a requirement
SLACK_EPS = 1e-3      # a DSL slack above 1 mm is a row the step did not meet; below it is the penalty's own residual
MAX_STEP_FAILURES = 4  # retries of one scale with a halved look-ahead before the continuation stops


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
    place: dict = field(default_factory=dict)   # name -> (x, y, yaw_deg|None): the pose given at the shrunken scale
    yaw_weight: dict = field(default_factory=dict)  # name -> weight (default size^2)
    margins: dict = field(default_factory=dict)     # name -> clearance kept from every body but its support
    project_iters: int = 0                          # program-only solves at s_min before the continuation
    project_step: float = 0.25                      # trust region of a projection solve (m per axis)
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
    trace: list = field(default_factory=list)   # (scale, contact rows, penetrating pairs, contact slack, DSL slack)
    notes: list = field(default_factory=list)   # what the solver could not do as asked
    relaxed_steps: int = 0                      # steps at which a DSL row was not met by more than SLACK_EPS
    continuation_complete: bool = True          # every scale step up to 1 was accepted
    last_accepted_scale: float = 1.0            # the last scale whose step was applied
    termination: str = "complete"               # "complete", or "qp_failures" when the continuation stopped early


def _validate_spec(spec: RepairSpec) -> None:
    for key in ("d_hat", "ds_max", "s_min", "max_yaw_step", "max_xy_step"):
        v = getattr(spec, key)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v):
            raise ValueError(f"{key} must be a finite number, got {v!r}")
    if spec.d_hat < 0.0:
        raise ValueError(f"d_hat must be non-negative, got {spec.d_hat}")
    if not 0.0 < spec.s_min <= 1.0:
        raise ValueError(f"s_min must lie in (0, 1], got {spec.s_min}")
    if not 0.0 < spec.ds_max <= 1.0:
        raise ValueError(f"ds_max must lie in (0, 1], got {spec.ds_max}")
    if spec.max_xy_step <= 0.0 or spec.max_yaw_step <= 0.0:
        raise ValueError("max_xy_step and max_yaw_step must be positive")
    if isinstance(spec.tail_iters, bool) or not isinstance(spec.tail_iters, (int, np.integer)) or spec.tail_iters < 0:
        raise ValueError(f"tail_iters must be a non-negative integer, got {spec.tail_iters!r}")
    if (1.0 - spec.s_min) / spec.ds_max + spec.tail_iters > 100000:
        raise ValueError("more than 100000 steps requested; raise ds_max or lower tail_iters")


def _status_codes():
    """OSQP's numeric status codes for the installed version (1.x names them in SolverStatus;
    0.6 used 1, 2 for solved and 3, -3 for primal infeasible)."""
    try:
        from osqp import SolverStatus
        return ({int(SolverStatus.OSQP_SOLVED), int(SolverStatus.OSQP_SOLVED_INACCURATE)},
                {int(SolverStatus.OSQP_PRIMAL_INFEASIBLE), int(SolverStatus.OSQP_PRIMAL_INFEASIBLE_INACCURATE)})
    except (ImportError, AttributeError):
        return {1, 2}, {3, -3}


def _status_text(info) -> str:
    return str(getattr(info, "status", "")).replace("_", " ").strip().lower()


def _qp_solved(info) -> bool:
    """Solved or solved inaccurately, by the status text first (it is the same across versions)
    and by the version's numeric code otherwise."""
    text = _status_text(info)
    if text:
        return text in ("solved", "solved inaccurate")
    sv = getattr(info, "status_val", None)
    return sv is not None and int(sv) in _status_codes()[0]


def _qp_infeasible(info) -> bool:
    """Primal infeasible, accurately or not."""
    text = _status_text(info)
    if text:
        return text.startswith("primal infeasible")
    sv = getattr(info, "status_val", None)
    return sv is not None and int(sv) in _status_codes()[1]


def _is_support_pair(scene: Scene, i: int, j: int, spec: RepairSpec) -> bool:
    """(free body, its declared support)."""
    bi, bj = scene.bodies[i], scene.bodies[j]
    return spec.support_of.get(bi.name) == bj.name or spec.support_of.get(bj.name) == bi.name


VERTICAL_COS = 0.866   # |n . up| above cos 30 deg: the resting contact carried by the support equality


def _rows_satisfied(st: State, spec: RepairSpec, tol: float = 1e-4) -> bool:
    """Every hard program row holds at the current poses (a soft row is a preference)."""
    if spec.rows_fn is None:
        return True
    for row in spec.rows_fn(st):
        if len(row) > 3 and row[3] is not None:
            continue
        coeffs, lo, hi = row
        if lo > tol or hi < -tol:
            return False
    return True


def _pair_margin(scene: Scene, i: int, j: int, spec: RepairSpec) -> float:
    """Clearance a contact row keeps for the pair: the program's margin, or a clearance statement's
    gap for a body other than the declared support of the other (or a body resting on it)."""
    if not spec.margins or _is_support_pair(scene, i, j, spec):
        return spec.d_hat
    return max(spec.d_hat, spec.margins.get(scene.bodies[i].name, 0.0), spec.margins.get(scene.bodies[j].name, 0.0))


def _margins_satisfied(scene: Scene, spec: RepairSpec, tol: float = 1e-4) -> bool:
    """Every pair under a clearance statement keeps it (the tail does not stop before)."""
    if not spec.margins:
        return True
    reach = max(spec.margins.values())
    for i, j, s, *_ in pair_signed_distances(scene.bodies, prefilter=reach):
        m = _pair_margin(scene, i, j, spec)
        if m > spec.d_hat and s < m - tol:
            return False
    return True


def _dsl_rows(st: State, spec: RepairSpec):
    """The program's rows at the current poses as solver rows [(entries, lo, hi, weight)], the
    columns of fixed bodies dropped; the weight is None for a requirement, a soft statement's
    weight for a preference."""
    out = []
    scene = st.scene
    for row in (spec.rows_fn(st) if spec.rows_fn is not None else []):
        coeffs, l, h = row[:3]
        wgt = float(row[3]) if len(row) > 3 and row[3] is not None else None
        ent = [(st.var[body] + {"x": 0, "y": 1, "yaw": 2}[v], val) for (body, v), val in coeffs.items()
               if not scene.bodies[body].fixed and abs(val) >= 1e-12]
        if ent:
            out.append((ent, l, h, wgt))
    return out


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


def _scaled_pairs(st: State, spec: RepairSpec, ds: float):
    """Contact candidates with free bodies scaled by s and fixed bodies at full size."""
    sc = st.scene
    # Temporarily scale free bodies' vertices for detection (arrays are rebound, not edited).
    saved = []
    for k in st.free:
        b = sc.bodies[k]
        saved.append(b.verts)
        b.verts = b.verts * st.scale
    try:
        # every pair that can activate within one scale step: the largest margin + ds * (full-scale extents)
        pre = max([spec.d_hat] + list(spec.margins.values())) + ds * 2.0 * max((b.diag / max(st.scale, 1e-9) for b in sc.free()), default=0.0)
        pairs = pair_signed_distances(sc.bodies, scale=1.0, prefilter=pre)
    finally:
        for k, v in zip(st.free, saved):
            sc.bodies[k].verts = v
    return pairs


def _solve_qp(n, contact_rows, dsl_rows, elastic_contacts, bounded, cap, yaw_cap, pd_x, q_x, verbose, scale):
    """One step QP over x = (dx, dy, dyaw per free body) plus one slack per elastic row.

    contact_rows: [(entries, rhs)] with a.x >= rhs;  dsl_rows: [(entries, lo, hi[, weight])]; `bounded`
    puts the trust region into the problem (the continuation always does). Returns (status,
    step (n, 3), max contact slack, max DSL slack); status 1 = solved, -1 = failed, -2 = primal
    infeasible (only possible while the contacts are hard)."""
    rows, cols, data, lo, hi = [], [], [], [], []
    r = 0
    nx = 3 * n
    slack = []            # (row index, sign, penalty) for elastic rows

    def add_row(entries, l, h, penalty):
        nonlocal r
        for col, val in entries:
            rows.append(r); cols.append(col); data.append(val)
        if penalty is not None:
            slack.append((r, 1.0 if h == np.inf else -1.0, penalty))
        lo.append(l); hi.append(h); r += 1

    for entries, rhs in contact_rows:
        add_row(entries, rhs, np.inf, RHO_CONTACT if elastic_contacts else None)
    n_contact_slack = len(slack)
    for row in dsl_rows:
        entries, l, h = row[:3]
        pen = RHO_SOFT * float(row[3]) if len(row) > 3 and row[3] is not None else RHO_DSL
        # one elastic row per finite bound; a two-sided row (a region) becomes two, so a body wider
        # than its region still gives a valid problem whose slack reports the shortfall
        if l > -np.inf:
            add_row(entries, l, np.inf, pen)
        if h < np.inf:
            add_row(entries, -np.inf, h, pen)
    n_slack = len(slack)
    # The slack of a row with penalty rho enters as sigma = sqrt(rho) s, so that its cost is
    # sigma^2 / 2 and P stays near unit diagonal: a.x + sigma / sqrt(rho) >= lo  |  a.x - sigma / sqrt(rho) <= hi.
    scale_s = np.array([math.sqrt(pen) for _, _, pen in slack])
    for k, (row, sign, _) in enumerate(slack):
        rows.append(row); cols.append(nx + k); data.append(sign / scale_s[k])
    for k in range(n_slack):                       # sigma >= 0
        rows.append(r); cols.append(nx + k); data.append(1.0); lo.append(0.0); hi.append(np.inf); r += 1
    for i in range(n if bounded else 0):           # trust regions, inside the problem
        for ax in (0, 1):
            rows.append(r); cols.append(3 * i + ax); data.append(1.0); lo.append(-cap); hi.append(cap); r += 1
        rows.append(r); cols.append(3 * i + 2); data.append(1.0); lo.append(-yaw_cap); hi.append(yaw_cap); r += 1
    A = sp.csc_matrix((data, (rows, cols)), shape=(r, nx + n_slack))
    P = sp.diags(list(pd_x) + [1.0] * n_slack, format="csc")
    q = np.concatenate([q_x, np.zeros(n_slack)])
    try:
        solver = osqp.OSQP()
        # a tail step with a few hundred program rows can need more than OSQP's default 4000
        # iterations to reach 1e-6; the cap is generous because a failed step applies nothing
        solver.setup(P, q, A, np.asarray(lo), np.asarray(hi), verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                     max_iter=50000, polishing=True)
        res = solver.solve(raise_error=False)      # an infeasible problem is a status, not an exception
    except TypeError:
        raise                     # a wrong call into the solver library is a bug, not a failed step
    except Exception as exc:  # the solver rejected the data: a failed step, nothing applied
        if verbose:
            print(f"[repair] QP setup failed at s={scale:.3f}: {type(exc).__name__}: {exc}")
        return -1, None, 0.0, 0.0
    if _qp_infeasible(res.info) and not elastic_contacts:
        return -2, None, 0.0, 0.0
    if not _qp_solved(res.info) or res.x is None or not np.all(np.isfinite(res.x)):
        if verbose:
            print(f"[repair] QP {res.info.status} at s={scale:.3f} rows={r}")
        return -1, None, 0.0, 0.0
    x = np.asarray(res.x, dtype=np.float64)
    s_all = np.maximum(x[nx:], 0.0) / scale_s if n_slack else np.zeros(0)
    c_sl = float(s_all[:n_contact_slack].max()) if n_contact_slack else 0.0
    d_sl = float(s_all[n_contact_slack:].max()) if n_slack > n_contact_slack else 0.0
    return 1, x[:nx].reshape(n, 3), c_sl, d_sl


def repair_upright(scene: Scene, spec: RepairSpec, verbose: bool = False, on_step=None) -> RepairResult:
    """on_step(state): optional hook called after every accepted step with the bodies posed at the
    step's scale and tilt (for trajectory recording / visualization)."""
    _validate_spec(spec)
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
    pen_before = sum(1 for p in pair_signed_distances(scene.bodies) if p[2] < 0.0)

    persistent = {}      # (i, j) -> consecutive tail iterations still penetrating
    notes = []
    contact_relaxed = 0

    def objective(pull: bool = True):
        """Diagonal weights and linear term of the step QP: minimum norm, yaw weighted by size^2,
        plus (with `pull`) the pull of prefer / place targets."""
        pd_x, q_x = [], np.zeros(3 * n)
        for i, k in enumerate(free):
            b = scene.bodies[k]
            yw = spec.yaw_weight.get(b.name, max(b.extent_along(np.array([1.0, 0, 0])), b.extent_along(np.array([0, 1.0, 0]))) ** 2)
            wx = 1.0
            if pull and b.name in spec.prefer:
                tgt, w = spec.prefer[b.name]
                wx += w
                q_x[3 * i:3 * i + 2] += w * (xy[i] - np.asarray(tgt))
            pd_x.extend([wx, wx, yw])
        return pd_x, q_x

    projected = 0
    if spec.project_iters and spec.rows_fn is not None and n:
        # the program alone at the shrunken scale: no contact rows (the bodies are small enough to
        # pass each other), a wide trust region; stops when every hard row holds or a solve fails
        st.tilt = tilt_schedule(st.scale, spec.s_min)
        for _ in range(int(spec.project_iters)):
            _pose_bodies(st, spec)
            rows = _dsl_rows(st, spec)
            if not rows or all(lo <= 1e-4 and hi >= -1e-4 for _, lo, hi, _ in rows):
                break
            pd_x, q_x = objective()
            status, step, _, _ = _solve_qp(n, [], rows, False, True, spec.project_step, math.pi / 4, pd_x, q_x, verbose, st.scale)
            if status != 1:
                break
            xy += step[:, :2]
            yaw += step[:, 2]
            projected += 1
        _pose_bodies(st, spec)

    def solve_step(tail: bool, ds: float, program: bool = True):
        """Returns (contact rows, penetrating pairs now, status, contact slack, DSL slack);
        status 1 = step applied, 0 = nothing to do, -1 = the QP failed and nothing was applied.
        Without `program` the step serves the contacts alone (no program rows, no target pull)."""
        nonlocal contact_relaxed
        st.tilt = 0.0 if tail else tilt_schedule(st.scale, spec.s_min)
        _pose_bodies(st, spec)
        ds = 0.0 if tail else ds
        pairs = _scaled_pairs(st, spec, ds if not tail else spec.ds_max)
        if tail:
            now = {(i, j) for (i, j, signed, *_) in pairs if signed < 0.0}
            for key in list(persistent):
                if key not in now:
                    del persistent[key]
            for key in now:
                persistent[key] = persistent.get(key, 0) + 1
        contact_rows = []
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
            rhs = _pair_margin(scene, i, j, spec) - signed + ds * ext
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
            contact_rows.append((entries, rhs))
        n_contact = len(contact_rows)
        dsl_rows = _dsl_rows(st, spec) if program else []
        if not contact_rows and not dsl_rows and not (program and spec.prefer):
            return 0, pen_now, 0, 0.0, 0.0
        pd_x, q_x = objective(pull=program)
        cap = spec.max_xy_step * (0.6 if tail else 1.0)
        status, step, c_sl, d_sl = _solve_qp(n, contact_rows, dsl_rows, False, True, cap, spec.max_yaw_step, pd_x, q_x, verbose, st.scale)
        if status == -2:
            # the contacts cannot all be met inside this step's trust region: serve them alone,
            # elastically, as far as the region allows; the program rows return at the next step
            status, step, c_sl, d_sl = _solve_qp(n, contact_rows, [], True, True, cap, spec.max_yaw_step, pd_x, q_x, verbose, st.scale)
            if status == 1:
                contact_relaxed += 1
                d_sl = math.inf          # every program row was set aside for this step
        if status != 1:
            return n_contact, pen_now, -1, 0.0, 0.0
        for i in range(n):
            xy[i] += step[i, :2]
            yaw[i] += float(step[i, 2])
        return n_contact, pen_now, 1, c_sl, d_sl

    trace = []
    steps = 0
    relaxed = 0
    s = spec.s_min
    ds = spec.ds_max
    failures = 0
    complete, last_accepted, termination = True, spec.s_min, "complete"
    while s < 1.0 - 1e-12:
        st.scale = s
        n_rows, pen_now, status, c_sl, d_sl = solve_step(tail=False, ds=ds)
        steps += 1
        if status == -1:
            failures += 1
            if failures <= MAX_STEP_FAILURES:
                ds *= 0.5      # nothing was applied: retry this scale with a shorter look-ahead
                continue
            complete, termination = False, "qp_failures"
            notes.append(f"continuation stopped at scale {s:.3f} after {failures} failed steps; the bodies were restored to "
                         f"full size at their last accepted positions and the tail ran there as a recovery, not as an accepted continuation")
            break
        failures = 0
        last_accepted = s
        trace.append((s, n_rows, pen_now, c_sl, d_sl))
        if d_sl > SLACK_EPS:
            relaxed += 1
        if on_step is not None:
            _pose_bodies(st, spec); on_step(st)
        s = min(1.0, s + ds)
        ds = spec.ds_max
    if complete:
        last_accepted = 1.0
    st.scale = 1.0
    st.tilt = 0.0
    idle = 0
    cleanup = 0
    pen_check = None
    last = max(1, spec.tail_iters // 5)              # the closing steps of the tail
    for it in range(spec.tail_iters):
        # the program yields to the contacts: when pairs still penetrate in the closing steps of the
        # tail, the rest of it serves the contacts alone (no program rows, no target pull)
        program = cleanup == 0 and not (spec.tail_iters - it <= last and (pen_check if pen_check is not None else 1) > 0)
        n_rows, pen_now, status, c_sl, d_sl = solve_step(tail=True, ds=0.0, program=program)
        steps += 1
        cleanup += 0 if program else 1
        trace.append((1.0, n_rows, pen_now, c_sl, d_sl))
        if d_sl > SLACK_EPS:
            relaxed += 1
        idle = idle + 1 if status in (0, -1) else 0     # nothing to solve, or nothing applied
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
        if pen_check == 0 and status != -1 and (cleanup or (_rows_satisfied(st, spec, tol=1e-4) and _margins_satisfied(scene, spec))):
            break           # done; once the closing steps set the program aside, a clean contact state ends the tail
    _pose_bodies(st, spec)
    pen_after = sum(1 for p in pair_signed_distances(scene.bodies) if p[2] < 0.0)
    if cleanup:
        notes.append(f"pairs still penetrated in the closing steps of the tail; {cleanup} step(s) served the contacts alone, the program rows set aside")
    disp = xy - xy0
    rmsd = float(np.sqrt(np.mean(np.sum(disp ** 2, axis=1)))) if n else 0.0
    if relaxed:
        notes.append(f"a program row was not met within the step at {relaxed} step(s) (a contact in the way, or the step's trust region); the predicates report the outcome")
    if contact_relaxed:
        notes.append(f"the contacts exceeded the trust region at {contact_relaxed} step(s); those steps served the contacts alone")
    if projected:
        notes.append(f"the program was solved alone at scale {spec.s_min:g} in {projected} step(s) before the scale was restored")
    if verbose:
        print(f"[repair] pen {pen_before} -> {pen_after}, steps={steps}, rmsd={rmsd:.4f}")
    return RepairResult(scene, pen_before, pen_after, steps, disp, rmsd, trace, notes, relaxed, complete, last_accepted, termination)


def reseat_on_supports(scene: Scene, spec: RepairSpec) -> None:
    """Re-seat every free body on its declared support using its current (full-resolution)
    mesh and the support height under its final position; call after repair ran on proxies."""
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
    spec2 = replace(spec, s_min=1.0, tail_iters=iters, place={}, project_iters=0)    # targets and the projection belong to s_min
    return repair_upright(scene, spec2, verbose=verbose, on_step=on_step)
