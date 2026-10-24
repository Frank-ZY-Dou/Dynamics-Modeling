"""Compile a DSL program against a scene into a RepairSpec and final predicates.

Frame convention = RoboLab's (robot at the origin looking along +x, table in front of it):
  left_of(a,b)     : a.y >= b.y + gap        right_of(a,b)  : a.y <= b.y - gap
  in_front_of(a,b) : a.x >= b.x + gap        behind(a,b)    : a.x <= b.x - gap
(RoboLab predicates.md: left-of = +Y, front-of = +X.) Regions name their sides near (-x, toward
the robot), far (+x), left (+y) and right (-y).

Per-step rows are linear in (dx, dy, dyaw) of free bodies. Region rows use the exact
projection interval of the scaled, yawed footprint (not a symmetric half-extent), and regions
attached to a free reference body are re-evaluated from that body's current pose every step.
The v1 statements (reach, row, grid, keep_out, ...) compile in dsl/statements.py. Defaults for
omitted keyword arguments come from dsl.schema.DEFAULTS and dsl.schema.OP_DEFAULTS in both the
text and the JSON path.
"""
from __future__ import annotations

import math
import numpy as np

from ..scene.model import Scene, Body
from ..repair.upright_s4r import RepairSpec, decompose_zyx
from .model import Program, Statement
from .schema import DEFAULTS
from . import statements as V1

AXIS = {"x": 0, "y": 1}
REL = {"left_of": ("y", +1), "right_of": ("y", -1), "in_front_of": ("x", +1), "behind": ("x", -1)}
BINARY = {"on_support", "within", "inside", "min_distance", "near", "parallel", "centered", "offset", "attach",
          "balanced_on", "keep_out", "facing", "beside"} | set(REL)
V1_ROWS = {"reach", "at", "beside", "facing", "yaw", "parallel", "max_turn", "row", "align", "evenly_spaced", "grid", "cluster",
           "symmetric", "between", "centered", "offset", "attach", "keep_out", "approach", "max_move", "balanced_on", "stack"}
PLACE_TOL = 0.05     # a placed body that ended within 5 cm of its target counts as placed
ANGLE_TOL = 0.5      # degrees of slack a heading predicate gets on top of its statement's tolerance


class CompileError(ValueError):
    pass


class Predicate(tuple):
    """(name, ok, value); `soft` is True for the predicate of a preference (a statement with soft=),
    which is reported but does not decide whether the scene passes."""
    soft = False


def _pred(name, ok, value, soft=False):
    p = Predicate((name, bool(ok), value))
    p.soft = soft
    return p


def _bodies(scene: Scene, tok: str, exclude=()):
    if tok == "*":
        return [b.name for b in scene.free() if b.name not in exclude]
    if tok not in scene.names():
        raise CompileError(f"unknown body '{tok}' (known: {', '.join(scene.names())})")
    return [tok]


def _ref(tok: str):
    return tok.split(".")[0]


def _region(scene: Scene, st: Statement, inset: float):
    """XY rectangle of a within/inside statement's region: the reference body's world AABB (or the
    plane, for bounds alone), the part or strip that side= selects, the bounds, then the inset."""
    ref = _ref(st.args[1]) if len(st.args) > 1 else None
    try:
        x0, x1, y0, y1 = V1.region_rect(*V1.reference_rect(scene, ref), st.kw, repr(st))
    except V1.StatementError as e:
        raise CompileError(str(e)) from None
    r = (x0 + inset, x1 - inset, y0 + inset, y1 - inset)
    if r[0] > r[1] or r[2] > r[3]:
        raise CompileError(f"inset {inset} leaves no room inside {st.args[1] if ref else 'the region'}")
    return r


def _rect_of(scene: Scene, tok: str, inset: float):
    """XY rectangle of `name` or `name.top` from the body's world AABB, inset."""
    lo, hi = scene[_ref(tok)].world_aabb()
    r = (lo[0] + inset, hi[0] - inset, lo[1] + inset, hi[1] - inset)
    if r[0] > r[1] or r[2] > r[3]:
        raise CompileError(f"inset {inset} leaves no room inside {tok}")
    return r


def _inset_of(scene: Scene, st: Statement) -> float:
    n = st.name
    inset = float(st.kw.get("inset", DEFAULTS["inset"] if n == "within" and len(st.args) > 1 and not
                              ({"side", "x0", "x1", "y0", "y1"} & set(st.kw)) else 0.0))
    if n == "inside":
        lo, hi = scene[_ref(st.args[1])].world_aabb()
        inset = max(inset, 0.1 * float(min(hi[0] - lo[0], hi[1] - lo[1])))
    return inset


def _footprint_interval(b: Body, s: float, axis: int):
    """[min, max] of the body's projection on world axis x/y at continuation scale s,
    relative to its reference centre (exact for any footprint shape)."""
    p = (b.rotation @ (s * b.verts).T).T[:, axis]
    return float(p.min()), float(p.max())


def _low_footprint(b: Body, interior: int = 6):
    """XY of the body's lowest vertices (within 10 % of its height above the minimum) — where it
    will touch a surface — used to seat it on the actual support surface, plus a grid of points
    inside their convex hull: a box's bottom has vertices only at its corners, and a support whose
    top is not flat (a carton's folds) can rise between them."""
    w = b.world_vertices()
    z = w[:, 2]
    band = z <= z.min() + 0.1 * max(z.max() - z.min(), 1e-6)
    pts = w[band][:, :2]
    if interior and len(pts) >= 3:
        lo, hi = pts.min(0), pts.max(0)
        if float(np.min(hi - lo)) > 1e-6:
            try:
                from scipy.spatial import Delaunay
                gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], interior + 2)[1:-1], np.linspace(lo[1], hi[1], interior + 2)[1:-1])
                grid = np.column_stack([gx.ravel(), gy.ravel()])
                inside = grid[Delaunay(pts).find_simplex(grid) >= 0]
                pts = np.vstack([pts, inside])
            except Exception:  # noqa: BLE001  collinear points: the vertices alone
                pass
    return pts


def support_height_for(scene: Scene, body: Body, support: Body, require_hit: bool = False):
    """Seat height of `body` on `support` from the support surface under the body's bottom.

    With `require_hit` the height is taken only from downward rays under the body's lowest
    vertices and its centre; when none of them meets the support the result is None (the body
    is not over the support at all). Without it, the vertex fallback in `Scene.support_height`
    supplies an initial guess for the repair, which is never used as evidence."""
    foot = _low_footprint(body)
    if require_hit:
        pts = np.vstack([foot, body.center[None, :2]])
        hits = scene._surface_hits(support, pts)
        if not hits.size:
            return None
        return max(float(hits.max()), _highest_vertex_under(support, foot, below=float(body.center[2])) or -np.inf)
    h = scene.support_height(support, at=body.center[:2], radius=0.5 * float(np.linalg.norm(body.world_aabb()[1][:2] - body.world_aabb()[0][:2])) + 0.02,
                             footprint=foot)
    hv = _highest_vertex_under(support, foot, below=float(body.center[2]) if body.center[2] > h else None)
    return max(h, hv) if hv is not None else h


def _highest_vertex_under(support: Body, foot: np.ndarray, below: float | None = None):
    """Highest vertex of the support inside the convex hull of a footprint (and under `below`, the
    body's centre, when given): with the rays at the footprint's points, the top of a piecewise-flat
    surface over the footprint, a ridge narrower than the ray grid included. None when no vertex is
    inside."""
    if len(foot) < 3:
        return None
    v = support.world_vertices()
    lo, hi = foot.min(0), foot.max(0)
    m = (v[:, 0] >= lo[0]) & (v[:, 0] <= hi[0]) & (v[:, 1] >= lo[1]) & (v[:, 1] <= hi[1])
    if below is not None:
        m &= v[:, 2] < below
    if not m.any():
        return None
    try:
        from scipy.spatial import Delaunay
        inside = Delaunay(foot).find_simplex(v[m][:, :2]) >= 0
    except Exception:  # noqa: BLE001  a degenerate footprint: the box test alone
        inside = np.ones(int(m.sum()), dtype=bool)
    return float(v[m][inside][:, 2].max()) if inside.any() else None


def effective_supports(program: Program, scene: Scene) -> dict:
    """body -> the support it rests on. A body named in on_support or in a stack keeps that support
    over one that a wildcard on_support(*, ...) gives it, whatever the statement order; between
    statements of the same kind the later one wins."""
    out, prio = {}, {}
    for st in program.statements:
        if st.name == "on_support":
            sup = _ref(st.args[1])
            if sup not in scene.names():
                raise CompileError(f"unknown support '{sup}'")
            p = 1 if st.args[0] == "*" else 2
            pairs = [(nm, sup) for nm in _bodies(scene, st.args[0], exclude=(sup,))]
        elif st.name == "stack":
            for nm in st.args:
                _bodies(scene, nm)
            p = 2
            pairs = list(zip(st.args[1:], st.args[:-1]))
        else:
            continue
        for nm, sup in pairs:
            if p >= prio.get(nm, 0):
                out[nm], prio[nm] = sup, p
    return out


def compile_program(program: Program, scene: Scene, **overrides) -> RepairSpec:
    spec = RepairSpec(supports={})
    spec.d_hat = DEFAULTS["margin"]
    for k, v in overrides.items():
        setattr(spec, k, v)
    # `fixed` statements first: wildcards must not depend on statement order
    for st in program.statements:
        if st.name == "fixed":
            for nm in st.args:
                if nm == "*":
                    raise CompileError("fixed(*) would fix every body")
                scene[nm].fixed = True
    for st in program.statements:
        if st.name in BINARY and len(st.args) >= 2:
            if st.args[0] == _ref(st.args[1]) and st.args[0] != "*":
                raise CompileError(f"{st.name} relates a body to itself: {st!r}")
        for nm in st.args:
            if st.name != "minimize" and nm != "*" and _ref(nm) not in scene.names():
                raise CompileError(f"unknown body '{nm}' in {st!r}")
        if st.name in REL and st.kw.get("axis", REL[st.name][0]) not in AXIS:
            raise CompileError(f"axis must be x or y: {st!r}")
    rects_static = {}   # body -> [(rect, label, weight)] for fixed references and bounds
    rects_dyn = {}      # body -> [(statement, ref_body, inset, label, weight)] for free references
    rels = []           # (a, b, axis, sign, gap, weight)
    dists = []          # (a, b, kind, r, weight)
    dyn_support = []    # (body, free support)
    v1 = []             # row generators of the v1 statements
    supports = effective_supports(program, scene)
    for nm, sup in supports.items():
        b = scene[nm]
        spec.supports[nm] = support_height_for(scene, b, scene[sup])
        spec.support_of[nm] = sup
        if not scene[sup].fixed:
            dyn_support.append((nm, sup))
    ctx = V1.Ctx(scene, {b.name: k for k, b in enumerate(scene.bodies)}, supports,
                 lambda tok, exclude=(): _bodies(scene, tok, exclude=exclude))
    for st in program.statements:
        n = st.name
        w = float(st.kw["soft"]) if "soft" in st.kw else None
        if n in ("fixed", "minimize", "on_support"):
            continue
        if n == "no_penetration":
            spec.d_hat = float(st.kw.get("margin", DEFAULTS["margin"]))
        elif n == "upright":
            spec.upright.update(_bodies(scene, st.args[0]))
        elif n in ("within", "inside"):
            inset = _inset_of(scene, st)
            ref = _ref(st.args[1]) if len(st.args) > 1 else None
            for nm in _bodies(scene, st.args[0], exclude=(ref,) if ref else ()):
                if ref is None or scene[ref].fixed:
                    rects_static.setdefault(nm, []).append((_region(scene, st, inset), n, w))
                else:
                    rects_dyn.setdefault(nm, []).append((st, ref, inset, n, w))
        elif n in REL:
            axis, sign = REL[n]
            axis = st.kw.get("axis", axis)
            gap = float(st.kw.get("gap", DEFAULTS["gap"]))
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                rels.append((a, st.args[1], axis, sign, gap, w))
        elif n in ("min_distance", "near"):
            if "r" not in st.kw:
                raise CompileError(f"{n} needs r=: {st!r}")
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                dists.append((a, st.args[1], n, float(st.kw["r"]), w))
        elif n == "prefer":
            w = float(st.kw.get("w", 0.1))
            for a in _bodies(scene, st.args[0]):
                spec.prefer[a] = (scene[a].center[:2].copy(), w)
        elif n == "place":
            a = st.args[0]
            if a == "*" or a not in scene.names():
                raise CompileError(f"place needs one known body: {st!r}")
            if "x" not in st.kw or "y" not in st.kw:
                raise CompileError(f"place needs x= and y=: {st!r}")
            yaw = float(st.kw["yaw"]) if "yaw" in st.kw else None
            spec.place[a] = (float(st.kw["x"]), float(st.kw["y"]), yaw)
            spec.prefer[a] = (np.array([float(st.kw["x"]), float(st.kw["y"])]), float(st.kw.get("w", 1.0)))
        elif n == "clearance":
            for nm in _bodies(scene, st.args[0]):
                spec.margins[nm] = max(spec.margins.get(nm, 0.0), float(st.kw["gap"]))
        elif n in V1_ROWS:
            try:
                v1.append(V1.compile_statement(st, ctx))
            except V1.StatementError as e:
                raise CompileError(str(e)) from None
        else:
            raise CompileError(f"unsupported statement {n}")
    if v1 and not spec.project_iters and "project_iters" not in overrides:
        spec.project_iters = 12          # relational layouts are reached at the shrunken scale first
    for a, b, *_ in rels + dists:
        if b not in scene.names():
            raise CompileError(f"unknown body '{b}'")
    idx = {b.name: k for k, b in enumerate(scene.bodies)}

    def emit(out, co, lo, hi, w):
        out.append((co, lo, hi) if w is None else (co, lo, hi, w))

    def region_rows(out, sc, k, b, s, xmin, xmax, ymin, ymax, ref_k=None, w=None):
        px0, px1 = _footprint_interval(b, s, 0); py0, py1 = _footprint_interval(b, s, 1)
        cx, cy = b.center[0], b.center[1]
        if ref_k is None:
            emit(out, {(k, "x"): 1.0}, xmin - px0 - cx, xmax - px1 - cx, w)
            emit(out, {(k, "y"): 1.0}, ymin - py0 - cy, ymax - py1 - cy, w)
        else:   # the region moves with the free reference body: rows couple both columns
            emit(out, {(k, "x"): 1.0, (ref_k, "x"): -1.0}, xmin - px0 - cx, xmax - px1 - cx, w)
            emit(out, {(k, "y"): 1.0, (ref_k, "y"): -1.0}, ymin - py0 - cy, ymax - py1 - cy, w)

    def rows_fn(state):
        out = []
        sc = state.scene
        for nm, sup in dyn_support:                      # a body on a free support follows its top
            spec.supports[nm] = support_height_for(sc, sc[nm], sc[sup])
        for nm, lst in rects_static.items():
            k = idx[nm]; b = sc.bodies[k]
            if b.fixed:
                continue
            s = state.body_scale(k)
            for (xmin, xmax, ymin, ymax), _, w in lst:
                region_rows(out, sc, k, b, s, xmin, xmax, ymin, ymax, w=w)
        for nm, lst in rects_dyn.items():
            k = idx[nm]; b = sc.bodies[k]
            if b.fixed:
                continue
            s = state.body_scale(k)
            for st, ref, inset, _, w in lst:
                xmin, xmax, ymin, ymax = _region(sc, st, inset)
                region_rows(out, sc, k, b, s, xmin, xmax, ymin, ymax, ref_k=idx[ref], w=w)
        for a, bname, axis, sign, gap, w in rels:
            ka, kb = idx[a], idx[bname]
            ax = AXIS[axis]
            cur = sc.bodies[ka].center[ax] - sc.bodies[kb].center[ax]
            emit(out, {(ka, axis): sign, (kb, axis): -sign}, gap - sign * cur, np.inf, w)   # sign*(cur + da - db) >= gap
        for a, bname, kind, r, w in dists:
            ka, kb = idx[a], idx[bname]
            d = sc.bodies[ka].center[:2] - sc.bodies[kb].center[:2]
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                # coincident centres give no direction: break the symmetry along +x, ordered by
                # name so that the row is the same whichever way the statement was written
                u = np.array([1.0 if a < bname else -1.0, 0.0])
            else:
                u = d / dist
            coeffs = {(ka, "x"): u[0], (ka, "y"): u[1], (kb, "x"): -u[0], (kb, "y"): -u[1]}
            if kind == "min_distance":
                emit(out, coeffs, r - dist, np.inf, w)
            else:
                emit(out, coeffs, -np.inf, r - dist, w)
        for gen in v1:
            out.extend(gen(state))
        return out

    spec.rows_fn = rows_fn
    return spec


def check_predicates(program: Program, scene: Scene, tol: float = 2e-3, upright_deg: float = 1.0, place_tol: float = PLACE_TOL,
                     angle_tol: float = ANGLE_TOL):
    """Evaluate every statement on the final scene. Returns [Predicate(name, ok, value)]; the
    predicates of soft statements have `.soft` set.
    on_support is judged by the body's lowest point against the support SURFACE under its bottom
    (downward rays), a geometric fact independent of the numbers the repair used; a body whose
    bottom is over no part of the support fails with value None, whatever its height.
    `tol` (m), `upright_deg`, `angle_tol` (degrees, on top of a heading statement's own tolerance)
    and `place_tol` (m) are the tolerances; the repair's verdict uses the defaults, a check on
    settled poses passes looser ones. max_move, max_turn and attach are measured from the poses
    the program was compiled on."""
    out = []
    supports = effective_supports(program, scene)

    def bodies_fn(tok, exclude=()):
        return _bodies(scene, tok, exclude=exclude)

    def on_support_pred(nm, sup, label):
        b = scene[nm]
        h = support_height_for(scene, b, scene[sup], require_hit=True)
        if h is None:
            return _pred(label, False, None)
        g = float((b.world_vertices() @ scene.up).min()) - h
        touching = True
        if abs(g) <= tol:
            # the rays under the body can meet the support through a hole in the body:
            # the body's mesh must also come within the tolerance of the support's mesh
            from ..gates.verify import pair_signed_distances
            pairs = pair_signed_distances([scene[sup], b], prefilter=tol, skip_fixed_pairs=False, containment=False)
            touching = bool(pairs) and float(pairs[0][2]) <= tol
        return _pred(label, abs(g) <= tol and touching, g)

    for st in program.statements:
        n = st.name
        soft = st.soft
        if n == "on_support":
            sup = _ref(st.args[1])
            for nm in _bodies(scene, st.args[0], exclude=(sup,)):
                if st.args[0] == "*" and supports.get(nm) != sup:
                    continue                  # a body named elsewhere rests on another support
                out.append(on_support_pred(nm, sup, f"on_support({nm},{sup})"))
        elif n == "stack":
            for lower, upper in zip(st.args, st.args[1:]):
                out.append(on_support_pred(upper, lower, f"stack({upper},{lower}):on"))
            out.extend(_pred(*p) for p in V1.check_v1(st, scene, supports, tol, angle_tol, bodies_fn))
        elif n == "upright":
            for nm in _bodies(scene, st.args[0]):
                _, roll, pitch = decompose_zyx(scene[nm].rotation)
                t = max(abs(roll), abs(pitch))
                out.append(_pred(f"upright({nm})", t <= math.radians(upright_deg), t))
        elif n in ("within", "inside"):
            inset = _inset_of(scene, st)
            ref = _ref(st.args[1]) if len(st.args) > 1 else None
            xmin, xmax, ymin, ymax = _region(scene, st, inset)
            label = st.args[1] if ref else "region"
            for nm in _bodies(scene, st.args[0], exclude=(ref,) if ref else ()):
                lo, hi = scene[nm].world_aabb()
                viol = max(xmin - lo[0], hi[0] - xmax, ymin - lo[1], hi[1] - ymax, 0.0)
                out.append(_pred(f"{n}({nm},{label})", viol <= tol, viol, soft))
        elif n in REL:
            axis, sign = REL[n]
            axis = st.kw.get("axis", axis); gap = float(st.kw.get("gap", DEFAULTS["gap"]))
            ax = AXIS[axis]
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                v = sign * (scene[a].center[ax] - scene[st.args[1]].center[ax]) - gap
                out.append(_pred(f"{n}({a},{st.args[1]},gap={gap:g})", v >= -tol, v, soft))
        elif n in ("min_distance", "near"):
            r = float(st.kw["r"])
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                d = float(np.linalg.norm(scene[a].center[:2] - scene[st.args[1]].center[:2]))
                ok = d >= r - tol if n == "min_distance" else d <= r + tol
                out.append(_pred(f"{n}({a},{st.args[1]},r={r:g})", ok, d, soft))
        elif n == "place":
            a = st.args[0]
            d = float(np.linalg.norm(scene[a].center[:2] - np.array([float(st.kw["x"]), float(st.kw["y"])])))
            out.append(_pred(f"place({a})", d <= place_tol, d))      # a soft target: reported against a loose tolerance
        elif n == "clearance" or n in V1_ROWS:
            try:
                for p in V1.check_v1(st, scene, supports, tol, angle_tol, bodies_fn):
                    out.append(_pred(*p, soft=soft))
            except V1.StatementError as e:
                raise CompileError(str(e)) from None
    return out
