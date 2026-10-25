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

import fnmatch
import math
import re

import numpy as np

from ..scene.model import Scene, Body
from ..repair.upright_s4r import RepairSpec, decompose_zyx
from .model import ARITY, GROUP, Program, Statement, is_wildcard
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


def natural_key(name: str):
    """Sort key that orders the numbers in a name by value: can_2 before can_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _matches(scene: Scene, tok: str):
    """The free bodies a pattern (can_*, fruit_?) matches, in natural order; one must match."""
    hits = sorted((b.name for b in scene.free() if fnmatch.fnmatchcase(b.name, tok)), key=natural_key)
    if not hits:
        raise CompileError(f"'{tok}' matches no free body (free: {', '.join(b.name for b in scene.free())})")
    return hits


def _bodies(scene: Scene, tok: str, exclude=()):
    if tok == "*":
        return [b.name for b in scene.free() if b.name not in exclude]
    if is_wildcard(tok):
        return [n for n in _matches(scene, tok) if n not in exclude]
    if tok not in scene.names():
        raise CompileError(f"unknown body '{tok}' (known: {', '.join(scene.names())})")
    return [tok]


# a wildcard first argument of these makes one statement per body
EXPAND_EACH = {"beside", "attach", "balanced_on"}


def expand_names(program: Program, scene: Scene) -> Program:
    """The program with the wildcards ('*', can_*) among a group's members replaced by the free
    bodies they match, in natural order, without the bodies the statement names itself (row(can_3,
    can_*) puts can_3 first, then the other cans); and a statement of EXPAND_EACH whose first
    argument is a wildcard replaced by one statement per body. Other first arguments are resolved by
    _bodies. The expansion is kept on the original statement for each set of free bodies, so that
    the rows compiled and the predicates checked come from the same statements (max_move, max_turn
    and attach record their reference poses on them)."""
    free = tuple(b.name for b in scene.free())
    out = []
    for st in program.statements:
        if (st.name in GROUP and any(is_wildcard(a) for a in st.args)) or (st.name in EXPAND_EACH and is_wildcard(st.args[0])):
            cache = st.__dict__.setdefault("_expanded", {})
            if free not in cache:
                cache[free] = _expand(st, scene)
            out.extend(cache[free])
        else:
            out.append(st)
    return Program(statements=out, gate=program.gate, source=program.source)


def _expand(st: Statement, scene: Scene) -> list:
    if st.name in EXPAND_EACH:
        ref = _ref(st.args[1])
        return [Statement(st.name, [nm] + list(st.args[1:]), dict(st.kw), st.line) for nm in _bodies(scene, st.args[0], exclude=(ref,))]
    named = {a for a in st.args if not is_wildcard(a)}
    members = []
    for a in st.args:
        if not is_wildcard(a):
            hits = [a]
        elif a == "*":
            hits = sorted((b.name for b in scene.free()), key=natural_key)
        else:
            hits = _matches(scene, a)
        for nm in hits:
            if nm not in members and (nm == a or nm not in named):
                members.append(nm)
    if len(members) < ARITY[st.name][0]:
        raise CompileError(f"{st!r} stands for {len(members)} bodies ({', '.join(members)}); {st.name} needs at least {ARITY[st.name][0]}")
    return [Statement(st.name, members, dict(st.kw), st.line)]


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
    """Seat height of `body` on `support`: the height of the body's lowest point at which it rests
    on the support's surface, touching it without sinking in (a drop).

    Vertical rays at sample points under the body (its lowest vertices, a grid inside their hull,
    and the support's vertices inside that hull) meet the support's top S going down and the
    body's underside B going up; the seat is the largest S - (B - lowest point of B). A flat box
    on a ridge rests on the ridge; a carton with a bulging bottom over the slats of a pallet
    settles until its flat parts meet them. With `require_hit` the result is None when no ray
    under the body meets the support (the body is not over it at all); without it, the vertex
    rule of `Scene.support_height` stands in where no ray meets both surfaces (an initial guess
    for the repair, never used as evidence)."""
    h = _drop_height(scene, body, support)
    if require_hit:
        if h is None:
            pts = np.vstack([_low_footprint(body), body.center[None, :2]])
            hits = scene._surface_hits(support, pts)
            return float(hits.max()) if hits.size else None
        return h
    if h is not None:
        return h
    return scene.support_height(support, at=body.center[:2], radius=0.5 * float(np.linalg.norm(body.world_aabb()[1][:2] - body.world_aabb()[0][:2])) + 0.02,
                                footprint=_low_footprint(body))


def _support_samples(support: Body, foot: np.ndarray, below: float) -> np.ndarray:
    """XY where the support's surface can rise highest under a footprint: its vertices inside the
    footprint's hull and the points where its triangles' edges cross the hull's boundary (moved
    1 mm inward, so that the vertical rays meet both bodies there). A ridge narrower than the
    footprint's sampling is caught where it crosses the boundary."""
    if len(foot) < 3 or not len(support.faces):
        return np.zeros((0, 2))
    try:
        from scipy.spatial import ConvexHull
        P = foot[ConvexHull(foot).vertices]                       # counter-clockwise
    except Exception:  # noqa: BLE001  a degenerate footprint
        return np.zeros((0, 2))
    T = support.world_vertices()[np.asarray(support.faces)]
    lo, hi = P.min(0), P.max(0)
    keep = ((T[:, :, 0].max(1) >= lo[0]) & (T[:, :, 0].min(1) <= hi[0]) & (T[:, :, 1].max(1) >= lo[1]) & (T[:, :, 1].min(1) <= hi[1])
            & (T[:, :, 2].min(1) < below))
    T = T[keep]
    if not len(T):
        return np.zeros((0, 2))
    out = []
    E = np.roll(P, -1, axis=0) - P
    pts = T.reshape(-1, 3)[:, :2]
    cross = E[None, :, 0] * (pts[:, None, 1] - P[None, :, 1]) - E[None, :, 1] * (pts[:, None, 0] - P[None, :, 0])
    out.append(pts[(cross >= -1e-12).all(1)])
    centroid = P.mean(0)
    for s0, s1 in ((T[:, 0], T[:, 1]), (T[:, 1], T[:, 2]), (T[:, 2], T[:, 0])):
        d = (s1 - s0)[:, :2]
        r = s0[:, None, :2] - P[None, :, :]
        den = d[:, None, 0] * E[None, :, 1] - d[:, None, 1] * E[None, :, 0]
        good = np.abs(den) > 1e-14
        t = np.where(good, (E[None, :, 0] * r[..., 1] - E[None, :, 1] * r[..., 0]) / np.where(good, den, 1.0), -1.0)
        w = np.where(good, (d[:, None, 0] * r[..., 1] - d[:, None, 1] * r[..., 0]) / np.where(good, den, 1.0), -1.0)
        hit = good & (t >= 0) & (t <= 1) & (w >= 0) & (w <= 1)
        if hit.any():
            xy = (s0[:, None, :2] + t[..., None] * d[:, None, :])[hit]
            v = centroid - xy
            out.append(xy + 1e-3 * v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))
    return np.vstack(out) if out else np.zeros((0, 2))


def _drop_height(scene: Scene, body: Body, support: Body):
    """max over sample points of S - (B - z_low): see support_height_for; None when no point has
    both surfaces above and below it."""
    foot = _low_footprint(body)
    P = np.vstack([foot, _support_samples(support, foot, below=float(body.center[2]))])
    S = scene.vertical_hits(support, P, below=float(body.center[2]))
    B = scene.vertical_hits(body, P, upward=True)
    ok = np.isfinite(S) & np.isfinite(B)
    if not ok.any():
        return None
    z_low = float(body.world_vertices()[:, 2].min())
    return float(np.max(S[ok] - (B[ok] - z_low)))


def effective_supports(program: Program, scene: Scene) -> dict:
    """body -> the support it rests on. A body named in on_support or in a stack keeps that support
    over one that a pattern gives it (on_support(can_*, tray)), and a pattern's over the one of
    on_support(*, ...), whatever the statement order; between statements of the same kind the later
    one wins."""
    out, prio = {}, {}
    for st in expand_names(program, scene).statements:
        if st.name == "on_support":
            sup = _ref(st.args[1])
            if sup not in scene.names():
                raise CompileError(f"unknown support '{sup}'")
            p = 1 if st.args[0] == "*" else 2 if is_wildcard(st.args[0]) else 3
            pairs = [(nm, sup) for nm in _bodies(scene, st.args[0], exclude=(sup,))]
        elif st.name == "stack":
            for nm in st.args:
                _bodies(scene, nm)
            p = 3
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
            for tok in st.args:
                if tok == "*":
                    raise CompileError("fixed(*) would fix every body")
                for nm in _bodies(scene, tok):
                    scene[nm].fixed = True
    for st in program.statements:
        if st.name in BINARY and len(st.args) >= 2:
            if st.args[0] == _ref(st.args[1]) and not is_wildcard(st.args[0]):
                raise CompileError(f"{st.name} relates a body to itself: {st!r}")
        for nm in st.args:
            if st.name == "minimize":
                continue
            if is_wildcard(nm):
                _bodies(scene, nm)               # a pattern must match a free body
            elif _ref(nm) not in scene.names():
                raise CompileError(f"unknown body '{nm}' in {st!r}")
        if st.name in REL and st.kw.get("axis", REL[st.name][0]) not in AXIS:
            raise CompileError(f"axis must be x or y: {st!r}")
    program = expand_names(program, scene)
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
    ctx.regions = rects_static          # filled below; a disjunction reads a body's own regions when it chooses a side
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
    program = expand_names(program, scene)
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
                if is_wildcard(st.args[0]) and supports.get(nm) != sup:
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
