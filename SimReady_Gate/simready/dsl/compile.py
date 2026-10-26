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

from ..scene.model import LEVEL_MIN_HEADROOM, Scene, Body, support_levels
from ..repair.upright_s4r import RepairSpec, decompose_zyx
from .model import ARITY, GROUP, Program, Statement, is_wildcard
from .schema import DEFAULTS
from . import statements as V1

AXIS = {"x": 0, "y": 1}
REL = {"left_of": ("y", +1), "right_of": ("y", -1), "in_front_of": ("x", +1), "behind": ("x", -1)}
BINARY = {"on_support", "within", "inside", "min_distance", "near", "parallel", "centered", "offset", "attach",
          "balanced_on", "keep_out", "facing", "beside", "flush"} | set(REL)
V1_ROWS = {"reach", "at", "beside", "facing", "yaw", "parallel", "max_turn", "row", "align", "evenly_spaced", "grid", "ring", "cluster",
           "symmetric", "between", "centered", "offset", "attach", "keep_out", "flush", "approach", "max_move", "balanced_on", "stack"}
LEVEL_OPS = {"on_support", "within", "keep_out", "flush"}     # the statements that take <fixture>.level<K>
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
        x0, x1, y0, y1 = V1.region_rect(*V1.reference_rect(scene, st.args[1] if ref else None), st.kw, repr(st))
    except V1.StatementError as e:
        raise CompileError(str(e)) from None
    r = (x0 + inset, x1 - inset, y0 + inset, y1 - inset)
    if r[0] > r[1] or r[2] > r[3]:
        raise CompileError(f"inset {inset} leaves no room inside {st.args[1] if ref else 'the region'}")
    return r


def _rect_of(scene: Scene, tok: str, inset: float):
    """XY rectangle of `name` or `name.top` from the body's world AABB, inset."""
    lo, hi = V1.body_aabb(scene[_ref(tok)])
    r = (lo[0] + inset, hi[0] - inset, lo[1] + inset, hi[1] - inset)
    if r[0] > r[1] or r[2] > r[3]:
        raise CompileError(f"inset {inset} leaves no room inside {tok}")
    return r


def _inset_of(scene: Scene, st: Statement) -> float:
    n = st.name
    inset = float(st.kw.get("inset", DEFAULTS["inset"] if n == "within" and len(st.args) > 1 and not
                              ({"side", "x0", "x1", "y0", "y1"} & set(st.kw)) else 0.0))
    if n == "inside":
        lo, hi = V1.body_aabb(scene[_ref(st.args[1])])
        inset = max(inset, 0.1 * float(min(hi[0] - lo[0], hi[1] - lo[1])))
    return inset


def _footprint_interval(b: Body, s: float, axis: int):
    """[min, max] of the body's projection on world axis x/y at continuation scale s,
    relative to its reference centre (exact for any footprint shape)."""
    p = (b.rotation @ (s * V1.footprint_verts(b)).T).T[:, axis]
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


def support_height_for(scene: Scene, body: Body, support: Body, require_hit: bool = False, from_above: bool = False):
    """Seat height of `body` on `support`: the height of the body's lowest point at which it rests
    on the support's surface, touching it without sinking in (a drop).

    The drop starts at the body's centre, so the seat is on the level of the support under it (a
    fixture's lower shelf); with `from_above` it starts above the support and the seat is on the
    highest surface under the body wherever the body is now, as for a body set down on another (a
    carton on a carton, which the body's starting height says nothing about).

    Vertical lines at sample points under the body (its lowest vertices, a grid inside their hull,
    and the support's vertices inside that hull) meet the support's top S going down and the
    body's underside B going up; the seat is the largest S - (B - lowest point of B). A flat box
    on a ridge rests on the ridge; a carton with a bulging bottom over the slats of a pallet
    settles until its flat parts meet them. With `require_hit` the result is None when no line
    under the body meets the support (the body is not over it at all); without it, the vertex
    rule of `Scene.support_height` stands in where no line meets both surfaces (an initial guess
    for the repair, never used as evidence)."""
    h = _drop_height(scene, body, support, below=math.inf if from_above else None)
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
    1 mm inward, so that the vertical lines meet both bodies there). A ridge narrower than the
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


def _whole_footprint(b: Body, interior: int = 6):
    """XY of the corners of the body's whole footprint (the convex hull of all its vertices) and a grid inside it: where
    a body that is not flat underneath meets what it is set on away from its lowest points (a bowl set in a bowl meets
    it with its wall)."""
    from scipy.spatial import ConvexHull, Delaunay, QhullError
    xy = b.world_vertices()[:, :2]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:          # a footprint that is a line or a point: its own points
        return xy[:: max(1, len(xy) // 64)]
    lo, hi = hull.min(0), hull.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], interior + 2)[1:-1], np.linspace(lo[1], hi[1], interior + 2)[1:-1])
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return np.vstack([hull, grid[Delaunay(hull).find_simplex(grid) >= 0]])


def _drop_height(scene: Scene, body: Body, support: Body, below: float | None = None):
    """max over sample points of S - (B - z_low): see support_height_for; None when no point has
    both surfaces above and below it. S is the support's highest surface under `below` (the body's
    centre by default). The points are those under the body's lowest part and over its whole
    footprint, with the support's vertices and edges inside the footprint of its lowest part."""
    below = float(body.center[2]) if below is None else below
    foot = _low_footprint(body)
    P = np.vstack([foot, _whole_footprint(body), _support_samples(support, foot, below=below)])
    S = scene.vertical_hits(support, P, below=below)
    B = scene.vertical_hits(body, P, upward=True)
    ok = np.isfinite(S) & np.isfinite(B)
    if not ok.any():
        return None
    z_low = float(body.world_vertices()[:, 2].min())
    return float(np.max(S[ok] - (B[ok] - z_low)))


def effective_supports(program: Program, scene: Scene) -> dict:
    """body -> the support it rests on (see effective_support_refs)."""
    return {nm: _ref(tok) for nm, tok in effective_support_refs(program, scene).items()}


def effective_support_refs(program: Program, scene: Scene) -> dict:
    """body -> the reference of the support it rests on, as written (`shelf.level2` for a level). A body named in
    on_support or in a stack keeps that support over one that a pattern gives it (on_support(can_*, tray)), and a
    pattern's over the one of on_support(*, ...), whatever the statement order; between statements of the same kind
    the later one wins."""
    out, prio = {}, {}
    for st in expand_names(program, scene).statements:
        if st.name == "on_support":
            sup = _ref(st.args[1])
            if sup not in scene.names():
                raise CompileError(f"unknown support '{sup}'")
            p = 1 if st.args[0] == "*" else 2 if is_wildcard(st.args[0]) else 3
            pairs = [(nm, st.args[1]) for nm in _bodies(scene, st.args[0], exclude=(sup,))]
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


def _check_reference(scene: Scene, st: Statement, tok: str) -> None:
    """`name.top` anywhere a body is taken; `name.levelK` only as the support or region of LEVEL_OPS, and only of a fixed
    body that has that level."""
    try:
        V1.check_reference(tok)
        if V1.LEVEL_RE.match(tok.partition(".")[2]):
            if st.name not in LEVEL_OPS or st.args.index(tok) != 1:
                raise V1.StatementError(f"'{tok}' is a level: it is the support of on_support or the region of within, keep_out and flush")
            if not scene[_ref(tok)].fixed:
                raise V1.StatementError(f"'{tok}': levels are a fixed body's")
            V1.level_of(scene, tok)
    except V1.StatementError as e:
        raise CompileError(f"{e} ({st!r})") from None


def _program_margin(program: Program) -> float:
    return next((float(st.kw.get("margin", DEFAULTS["margin"])) for st in program.statements if st.name == "no_penetration"),
                DEFAULTS["margin"])


def _height(scene: Scene, nm: str, upright: set) -> float:
    """A body's height as it will rest: its own up extent when it is kept upright, else its present extent."""
    b = scene[nm]
    return float(np.ptp(b.verts[:, 2])) if nm in upright else b.extent_along(scene.up)


def _level_seat(scene: Scene, program: Program, nm: str, tok: str, upright: set) -> float:
    """Height of the level `tok` a body rests on; the body must fit under the fixture's surface above it, with the
    program's margin."""
    lv = V1.level_of(scene, tok)
    h = _height(scene, nm, upright)
    margin = _program_margin(program)
    if h + margin > lv.headroom:
        room = [f"{_ref(tok)}.level{o.index} ({'open above' if math.isinf(o.headroom) else f'{o.headroom:.3f} m'})"
                for o in support_levels(scene, scene[_ref(tok)]) if h + margin <= o.headroom]
        raise CompileError(f"{nm} is {h:.3f} m tall and {tok} leaves {lv.headroom:.3f} m under the surface above it"
                           + (f"; levels with room for it: {', '.join(room)}" if room else "; no level of it has room"))
    return lv.z


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
            elif "." in nm:
                _check_reference(scene, st, nm)
        if st.name in REL and st.kw.get("axis", REL[st.name][0]) not in AXIS:
            raise CompileError(f"axis must be x or y: {st!r}")
    program = expand_names(program, scene)
    rects_static = {}   # body -> [(rect, label, weight)] for fixed references and bounds
    rects_dyn = {}      # body -> [(statement, ref_body, inset, label, weight)] for free references
    rels = []           # (a, b, axis, sign, gap, weight)
    dists = []          # (a, b, kind, r, weight)
    dyn_support = []    # (body, free support)
    v1 = []             # row generators of the v1 statements
    refs = effective_support_refs(program, scene)
    supports = {nm: _ref(tok) for nm, tok in refs.items()}
    upright = {nm for st in program.statements if st.name == "upright" for nm in _bodies(scene, st.args[0])}
    for nm, sup in supports.items():
        b = scene[nm]
        if V1.level_of(scene, refs[nm]) is not None:
            spec.supports[nm] = _level_seat(scene, program, nm, refs[nm], upright)
        else:
            spec.supports[nm] = support_height_for(scene, b, scene[sup], from_above=not scene[sup].fixed)
        spec.support_of[nm] = sup
        if not scene[sup].fixed:
            dyn_support.append((nm, sup))
    # A body on a level it is given (shelf.level2), or on a level under part of its fixture (a shelf of a bookcase), keeps
    # its footprint on that level, the margin inside its rectangle, as a contact does: the repair takes one contact
    # witness per pair, which for the body and its fixture is the board it rests on, and that one hides the walls and
    # the rounded edge of a top. A body stacked on such a body keeps the same bounds.
    margin = _program_margin(program)
    for nm, sup in supports.items():
        levels = support_levels(scene, scene[sup]) if scene[sup].fixed else []
        lv = V1.level_of(scene, refs[nm]) if levels else None
        if lv is None and len(levels) > 1:
            lv = min(levels, key=lambda o: abs(o.z - spec.supports[nm]))
            lv = lv if lv.index < len(levels) and abs(lv.z - spec.supports[nm]) <= 0.5 * LEVEL_MIN_HEADROOM else None
        if lv is not None:
            spec.bounds[nm] = (lv.x0 + margin, lv.x1 - margin, lv.y0 + margin, lv.y1 - margin)
            foot = sorted(np.ptp(V1.footprint_verts(scene[nm])[:, :2], axis=0))
            room = sorted((lv.x1 - lv.x0 - 2 * margin, lv.y1 - lv.y0 - 2 * margin))
            if foot[0] > room[0] or foot[1] > room[1]:
                raise CompileError(f"{nm} ({foot[1]:.3f} x {foot[0]:.3f} m) does not fit on {_ref(refs[nm])}.level{lv.index} "
                                   f"({room[1]:.3f} x {room[0]:.3f} m inside the margin)")
    for nm in supports:
        base, seen, tall = nm, {nm}, _height(scene, nm, upright)
        while supports.get(base) in supports and not scene[supports[base]].fixed and supports[base] not in seen:
            base = supports[base]
            seen.add(base)
            tall += _height(scene, base, upright)
        if base != nm and base in spec.bounds:
            spec.bounds[nm] = spec.bounds[base]
            lv = V1.level_of(scene, refs[base])
            if lv is not None and tall + margin > lv.headroom:
                raise CompileError(f"the stack from {base} up to {nm} is {tall:.3f} m tall and {refs[base]} leaves {lv.headroom:.3f} m "
                                   f"under the surface above it")
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
            spec.supports[nm] = support_height_for(sc, sc[nm], sc[sup], from_above=True)
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
    (vertical lines down), a geometric fact independent of the numbers the repair used; a body whose
    bottom is over no part of the support fails with value None, whatever its height.
    `tol` (m), `upright_deg`, `angle_tol` (degrees, on top of a heading statement's own tolerance)
    and `place_tol` (m) are the tolerances; the repair's verdict uses the defaults, a check on
    settled poses passes looser ones. max_move, max_turn and attach are measured from the poses
    the program was compiled on."""
    out = []
    program = expand_names(program, scene)
    refs = effective_support_refs(program, scene)
    supports = {nm: _ref(tok) for nm, tok in refs.items()}

    def bodies_fn(tok, exclude=()):
        return _bodies(scene, tok, exclude=exclude)

    def on_support_pred(nm, sup, label, level=None):
        b = scene[nm]
        h = support_height_for(scene, b, scene[sup], require_hit=True, from_above=not scene[sup].fixed)
        if h is None or (level is not None and abs(h - level.z) > max(tol, 0.5 * LEVEL_MIN_HEADROOM)):
            return _pred(label, False, None if h is None else float((b.world_vertices() @ scene.up).min()) - level.z)
        if level is not None:                 # on the level: over its rectangle, not only touching its board
            lo, hi = V1.body_aabb(b)
            if max(level.x0 - lo[0], hi[0] - level.x1, level.y0 - lo[1], hi[1] - level.y1) > tol:
                return _pred(label, False, max(level.x0 - lo[0], hi[0] - level.x1, level.y0 - lo[1], hi[1] - level.y1))
        g = float((b.world_vertices() @ scene.up).min()) - h
        from ..gates.verify import pair_signed_distances

        def gap(down: float = 0.0) -> float:
            """Signed distance of the meshes with the body lowered by `down`."""
            c0 = b.center
            b.center = c0 - down * np.asarray(scene.up, dtype=np.float64)
            try:
                pairs = pair_signed_distances([scene[sup], b], prefilter=down + tol, skip_fixed_pairs=False, containment=False)
            finally:
                b.center = c0
            return float(pairs[0][2]) if pairs else math.inf

        if abs(g) <= tol:
            # the lines under the body can meet the support through a hole in the body:
            # the body's mesh must also come within the tolerance of the support's mesh
            ok = gap() <= tol
        else:
            # the sampled surface under the body can miss its highest point (a board's edge, a fold of a base), and the
            # repair lifts a body clear of what it would cut into: a body higher than the sampled surface rests all the
            # same when its mesh touches the support's and lowering it by twice the tolerance cuts into the support
            ok = g > tol and abs(gap()) <= tol and gap(2.0 * tol) < 0.0
        return _pred(label, ok, g)

    for st in program.statements:
        n = st.name
        soft = st.soft
        if n == "on_support":
            sup, level = _ref(st.args[1]), V1.level_of(scene, st.args[1])
            for nm in _bodies(scene, st.args[0], exclude=(sup,)):
                if is_wildcard(st.args[0]) and refs.get(nm) != st.args[1]:
                    continue                  # a body named elsewhere rests on another support, or on another level
                out.append(on_support_pred(nm, sup, f"on_support({nm},{st.args[1] if level else sup})", level))
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
