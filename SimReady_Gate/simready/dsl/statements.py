"""Rows and predicates of the v1 statements.

`compile_statement(st, ctx)` turns one statement into a callable gen(state) -> [row] that the
repair calls at every step, with the bodies posed at the step's scale; `check_v1(st, scene, ...)`
evaluates the statement on a final scene. A row is (coeffs, lo, hi): coeffs maps
(body index, 'x' | 'y' | 'yaw') to the coefficient of that body's step, and the step must satisfy
lo <= sum(coeffs * step) <= hi. A soft statement's rows carry its weight as a fourth element.

How each kind of statement becomes rows:
  linear in the poses (at, beside, row, align, evenly_spaced, grid, symmetric, between, centered,
    offset in the scene frame, yaw windows, regions; flush, with the first-order turn of its edge):
    exact rows, re-evaluated at each step only because footprints change with the scale and the
    heading;
  convex but not linear (reach's outer radius, cluster, balanced_on with a centre of mass off the
    reference centre): the tangent half-plane at the current poses, a superset that tightens as the
    steps re-linearise;
  non-convex (reach's inner radius, facing, offset in a body's frame, attach): the first-order
    expansion at the current poses;
  disjunctions (keep_out, approach: one of several sides must hold): the side that needs the least
    motion at the current poses, chosen again at every step (for keep_out, among the sides that
    leave the body room in its own required regions);
  max_move: a 16-gon inscribed in the allowed disc, exact rows that are at most 2 % conservative.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..scene.model import support_levels
from .schema import OP_DEFAULTS

INF = np.inf
AXIS_VEC = {"x": (1.0, 0.0, 0.0), "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}
COL = ("x", "y")
SIDE_META = {"front": "front_axis", "handle": "grasp_axis"}   # axis words a body's own data defines (scene summary)
LEVEL_RE = re.compile(r"^level([1-9]\d*)$")
SIDE_VEC = {"left": (0.0, 1.0), "right": (0.0, -1.0), "far": (1.0, 0.0), "near": (-1.0, 0.0)}
CORRIDOR_MARGIN = 0.005    # a body counts as out of an approach corridor 5 mm beyond its edge
POLYGON = 16               # sides of the polygon that stands in for max_move's disc


class StatementError(ValueError):
    pass


def kwd(st, key):
    """A keyword's value, or the statement's recorded default."""
    if key in st.kw:
        return st.kw[key]
    return OP_DEFAULTS.get(st.name, {}).get(key)


START_AXIS = {"left": "-y", "right": "+y", "near": "+x", "far": "-x"}     # the end a row's first member is at


def row_axis(st) -> str:
    """A row's signed axis, the direction its list runs in: away from the end start= names, else
    toward +axis."""
    return START_AXIS[st.kw["start"]] if "start" in st.kw else "+" + kwd(st, "axis")


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_mod(a: float, m: float) -> float:
    """a reduced to [-m/2, m/2): the nearest of the angles equivalent under a turn by m."""
    return (a + 0.5 * m) % m - 0.5 * m


def heading(R: np.ndarray, axis=(1.0, 0.0, 0.0)) -> float | None:
    """Angle about z of a body axis given in its own frame, or None when that axis is vertical."""
    w = R @ np.asarray(axis, dtype=np.float64)
    if math.hypot(w[0], w[1]) < 1e-6:
        return None
    return math.atan2(w[1], w[0])


def footprint_verts(b) -> np.ndarray:
    """The model-frame vertices a footprint is measured on: the body's own mesh, also while a
    repair runs its contacts on a decimated copy (scene.model.MeshProxy)."""
    return b.meta.get("full_verts", b.verts)


def body_aabb(b):
    """World bounding box of the body's own mesh (see footprint_verts)."""
    w = (b.rotation @ footprint_verts(b).T).T + b.center
    return w.min(axis=0), w.max(axis=0)


def body_axis(b, tok: str):
    """(local axis vector, is_line) for an axis word: 'x', '+x', '-y', ..., 'long' / 'short', the
    longer / shorter horizontal local axis of the body, which is a line (a turn by 180 degrees maps
    it onto itself), or 'front' / 'handle', the side the body's own data names (a book's spine, a
    mug's handle)."""
    if tok in SIDE_META:
        word = b.meta.get(SIDE_META[tok])
        if word not in AXIS_VEC:
            raise StatementError(f"{b.name} has no {tok} side in its data (axis={tok} needs one; the scene summary lists the sides it knows)")
        return AXIS_VEC[word], False
    if tok in ("long", "short"):
        ext = np.ptp(footprint_verts(b), axis=0)
        longer_x = ext[0] >= ext[1]
        return ((1.0, 0.0, 0.0) if (longer_x == (tok == "long")) else (0.0, 1.0, 0.0)), True
    return AXIS_VEC[tok], False


def projection(b, s: float, u) -> tuple[float, float]:
    """[min, max] of the body's footprint at scale s along the horizontal unit direction u,
    relative to its reference centre (exact for any footprint)."""
    p = (b.rotation @ (s * footprint_verts(b)).T).T[:, :2] @ np.asarray(u, dtype=np.float64)
    return float(p.min()), float(p.max())


def edge_of(b, s: float, u, low: bool) -> tuple[float, float]:
    """(the footprint's min (low) or max along the horizontal unit direction u at scale s, relative to the reference
    centre; its derivative with the body's yaw): the extreme vertex turns with the body."""
    w = (b.rotation @ (s * footprint_verts(b)).T).T[:, :2]
    p = w @ np.asarray(u, dtype=np.float64)
    i = int(np.argmin(p) if low else np.argmax(p))
    return float(p[i]), float(np.array([-w[i, 1], w[i, 0]]) @ np.asarray(u, dtype=np.float64))


def footprint_hull(b) -> np.ndarray:
    """Convex hull (counter-clockwise) of the body's footprint in the plane."""
    from scipy.spatial import ConvexHull
    xy = ((b.rotation @ footprint_verts(b).T).T + b.center)[:, :2]
    try:
        h = ConvexHull(xy)
        return xy[h.vertices]
    except Exception:  # noqa: BLE001  a degenerate footprint (a line, a point): its bounding box
        lo, hi = xy.min(0), xy.max(0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def polygon_overlap(P: np.ndarray, Q: np.ndarray) -> float:
    """Penetration of two convex polygons: the smallest overlap of their projections over the edge
    normals of both (separating axis theorem); <= 0 when they are apart, and then minus the gap
    along the best separating axis."""
    best = INF
    for poly in (P, Q):
        for k in range(len(poly)):
            e = poly[(k + 1) % len(poly)] - poly[k]
            n = np.array([-e[1], e[0]])
            ln = float(np.linalg.norm(n))
            if ln < 1e-12:
                continue
            n /= ln
            p, q = P @ n, Q @ n
            best = min(best, float(min(p.max(), q.max()) - max(p.min(), q.min())))
    return best


def rect_polygon(x0, x1, y0, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def region_rect(lo, hi, kw: dict, name: str = "") -> tuple[float, float, float, float]:
    """Rectangle of a region: the reference rectangle [lo, hi] (inf where there is none), then the
    part of it that side=/frac= select or the strip beside it that side=/depth= give, then the
    bounds x0=, x1=, y0=, y1=. near is the side toward the robot (-x), far the side away (+x)."""
    x0, y0 = float(lo[0]), float(lo[1])
    x1, y1 = float(hi[0]), float(hi[1])
    side = kw.get("side")
    if side is not None:
        if "depth" in kw:
            d = float(kw["depth"])
            if side == "near":
                x0, x1 = x0 - d, x0
            elif side == "far":
                x0, x1 = x1, x1 + d
            elif side == "left":
                y0, y1 = y1, y1 + d
            elif side == "right":
                y0, y1 = y0 - d, y0
            else:
                raise StatementError(f"{name}: side=center has no strip beside the region; use frac=")
        else:
            f = float(kw.get("frac", OP_DEFAULTS["region"]["frac"]))
            if side == "near":
                x1 = x0 + f * (x1 - x0)
            elif side == "far":
                x0 = x1 - f * (x1 - x0)
            elif side == "left":
                y0 = y1 - f * (y1 - y0)
            elif side == "right":
                y1 = y0 + f * (y1 - y0)
            else:
                cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * f * (x1 - x0), 0.5 * f * (y1 - y0)
                x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
    x0 = max(x0, float(kw.get("x0", -INF))); x1 = min(x1, float(kw.get("x1", INF)))
    y0 = max(y0, float(kw.get("y0", -INF))); y1 = min(y1, float(kw.get("y1", INF)))
    if x0 >= x1 or y0 >= y1:
        raise StatementError(f"{name}: the region is empty (x [{x0:.3f}, {x1:.3f}], y [{y0:.3f}, {y1:.3f}])")
    return x0, x1, y0, y1


def check_reference(tok: str) -> None:
    """A reference is a body, `name.top` (its bounding box, as the body) or `name.levelK`."""
    part = tok.partition(".")[2]
    if part not in ("", "top") and not LEVEL_RE.match(part):
        raise StatementError(f"'{tok}': a reference is a body, <body>.top or <fixture>.level<K> (K from 1 at the bottom)")


def level_of(scene, tok: str):
    """The scene.model.Level that `name.levelK` names, K counted from 1 at the bottom; None for another reference."""
    name, _, part = tok.partition(".")
    m = LEVEL_RE.match(part)
    if not m:
        return None
    levels = support_levels(scene, scene[name])
    k = int(m.group(1))
    if k > len(levels):
        raise StatementError(f"'{tok}': {name} has {len(levels)} level{'' if len(levels) == 1 else 's'}")
    return levels[k - 1]


def reference_rect(scene, tok: str | None):
    """(lo, hi) of a region reference: the body's world bounding box in the plane (`name`, `name.top`), the rectangle
    of one of a fixture's levels (`name.levelK`), or the whole plane."""
    if tok is None:
        return (-INF, -INF), (INF, INF)
    lv = level_of(scene, tok)
    if lv is not None:
        return (lv.x0, lv.y0), (lv.x1, lv.y1)
    lo, hi = body_aabb(scene[tok.split(".")[0]])
    return lo[:2], hi[:2]


class Ctx:
    """What the statements need from the compiler: the scene at compile time, body indices, the
    effective supports and the wildcard expansion."""

    def __init__(self, scene, idx, supports, bodies_fn):
        self.scene, self.idx, self.supports, self.bodies = scene, idx, supports, bodies_fn
        self.regions = {}               # body name -> [(rect, label, weight)] of its within/inside statements with fixed references


def _acc(co: dict, key, val: float):
    co[key] = co.get(key, 0.0) + val


def _row(out, co, lo, hi, w):
    out.append((co, lo, hi) if w is None else (co, lo, hi, w))


def _c(sc, k):
    return sc.bodies[k].center[:2]


def compile_statement(st, ctx: Ctx):
    """gen(state) -> rows for a v1 statement (and `within`/`keep_out` regions); records on the
    statement the reference poses its predicate is measured against."""
    n, kw, sc0, idx = st.name, st.kw, ctx.scene, ctx.idx
    w = float(kw["soft"]) if "soft" in kw else None
    gens = []
    if n == "reach":
        ks = [idx[b] for b in ctx.bodies(st.args[0])]
        r, rmin = float(kw["r"]), float(kwd(st, "min"))
        base = np.array([float(kwd(st, "x")), float(kwd(st, "y"))])

        def gen(state):
            out = []
            for k in ks:
                d = _c(state.scene, k) - base
                rho = float(np.linalg.norm(d))
                u = d / rho if rho > 1e-9 else np.array([1.0, 0.0])
                co = {(k, "x"): float(u[0]), (k, "y"): float(u[1])}
                _row(out, co, -INF, r - rho, w)
                if rmin > 0.0:
                    _row(out, dict(co), rmin - rho, INF, w)
            return out
        gens.append(gen)
    elif n == "at":
        ks = [idx[b] for b in ctx.bodies(st.args[0])]
        tol = float(kwd(st, "tol"))
        target = {0: kw.get("x"), 1: kw.get("y")}

        def gen(state):
            out = []
            for k in ks:
                c = _c(state.scene, k)
                for ax in (0, 1):
                    if target[ax] is not None:
                        cur = float(c[ax]) - float(target[ax])
                        _row(out, {(k, COL[ax]): 1.0}, -tol - cur, tol - cur, w)
            return out
        gens.append(gen)
    elif n == "beside":
        ka, kb = idx[st.args[0]], idx[st.args[1]]
        u = np.array(SIDE_VEC[kw["side"]])
        gap = float(kwd(st, "gap"))
        top = float(kw["max"]) if "max" in kw else INF

        def gen(state):
            sc = state.scene
            lo_a, _ = projection(sc.bodies[ka], state.body_scale(ka), u)
            _, hi_b = projection(sc.bodies[kb], state.body_scale(kb), u)
            g = float((_c(sc, ka) - _c(sc, kb)) @ u) + lo_a - hi_b          # table between the footprints along u
            co = {}
            for col in (0, 1):
                if abs(u[col]) > 0:
                    _acc(co, (ka, COL[col]), float(u[col])); _acc(co, (kb, COL[col]), -float(u[col]))
            out = []
            _row(out, co, gap - g, top - g, w)
            return out
        gens.append(gen)
    elif n == "facing":
        ks = [idx[b] for b in ctx.bodies(st.args[0], exclude=st.args[1:2])]
        axes = {k: body_axis(sc0.bodies[k], kwd(st, "axis"))[0] for k in ks}
        tol = math.radians(float(kwd(st, "tol")))
        tb = idx[st.args[1]] if len(st.args) > 1 else None
        point = np.array([float(kw.get("x", 0.0)), float(kw.get("y", 0.0))])

        def gen(state):
            out = []
            for k in ks:
                phi = heading(state.scene.bodies[k].rotation, axes[k])
                if phi is None:
                    continue
                t = _c(state.scene, tb) if tb is not None else point
                d = t - _c(state.scene, k)
                rho2 = float(d @ d)
                if rho2 < 1e-12:
                    continue
                err = wrap(math.atan2(d[1], d[0]) - phi)
                g = np.array([-d[1], d[0]]) / rho2            # d(target angle)/d(target - body)
                co = {(k, "yaw"): 1.0, (k, "x"): float(g[0]), (k, "y"): float(g[1])}
                if tb is not None:
                    _acc(co, (tb, "x"), -float(g[0])); _acc(co, (tb, "y"), -float(g[1]))
                _row(out, co, err - tol, err + tol, w)
            return out
        gens.append(gen)
    elif n in ("yaw", "parallel", "max_turn"):
        ks = [idx[b] for b in ctx.bodies(st.args[0], exclude=st.args[1:2])]
        if n == "max_turn":
            m, tol = 2.0 * math.pi, math.radians(float(kw["deg"]))
        else:
            m, tol = math.radians(float(kwd(st, "mod"))), math.radians(float(kwd(st, "tol")))
        kb = idx[st.args[1]] if n == "parallel" else None
        target = math.radians(float(kw["deg"])) if n == "yaw" else None
        tok = kwd(st, "axis") if n != "max_turn" else "x"
        axes = {k: body_axis(sc0.bodies[k], tok) for k in ks + ([kb] if kb is not None else [])}
        if n == "max_turn":
            st.ref = {"heading": {k: heading(sc0.bodies[k].rotation) for k in ks}}

        def gen(state):
            out = []
            for k in ks:
                vec, line = axes[k]
                phi = heading(state.scene.bodies[k].rotation, vec)
                if phi is None:
                    continue
                mk = min(m, math.pi) if line else m
                if n == "yaw":
                    err = wrap_mod(target - phi, mk)
                    _row(out, {(k, "yaw"): 1.0}, err - tol, err + tol, w)
                elif n == "max_turn":
                    h0 = st.ref["heading"][k]
                    if h0 is not None:
                        err = wrap(h0 - phi)
                        _row(out, {(k, "yaw"): 1.0}, err - tol, err + tol, w)
                else:
                    vb, line_b = axes[kb]
                    pb = heading(state.scene.bodies[kb].rotation, vb)
                    if pb is not None:
                        err = wrap_mod(pb - phi, min(mk, math.pi) if line_b else mk)
                        _row(out, {(k, "yaw"): 1.0, (kb, "yaw"): -1.0}, err - tol, err + tol, w)
            return out
        gens.append(gen)
    elif n == "row":
        ks = [idx[b] for b in sorted_members(st, sc0)]
        tok = row_axis(st)
        ax = 0 if tok.endswith("x") else 1
        sgn = -1.0 if tok.startswith("-") else 1.0
        pa = 1 - ax
        u = np.zeros(2); u[ax] = sgn
        gap, tol = float(kwd(st, "gap")), float(kwd(st, "tol"))
        even = kw.get("spacing") == "even"
        mid = float(kw["mid"]) if "mid" in kw else None
        edge = kwd(st, "edge")
        v = np.zeros(2); v[pa] = 1.0

        def gen(state):
            out, sc = [], state.scene

            def q(k):                            # what lines up across the row: the centre, or a footprint edge
                if edge == "center":
                    return float(_c(sc, k)[pa])
                lo, hi = projection(sc.bodies[k], state.body_scale(k), v)
                return float(_c(sc, k)[pa]) + (lo if edge == "min" else hi)
            k0 = ks[0]
            for k in ks[1:]:
                cur = q(k) - q(k0)
                _row(out, {(k, COL[pa]): 1.0, (k0, COL[pa]): -1.0}, -tol - cur, tol - cur, w)
            if mid is not None:                  # the midpoint of the end members, along the row
                cur = 0.5 * float(_c(sc, ks[0])[ax] + _c(sc, ks[-1])[ax])
                _row(out, {(ks[0], COL[ax]): 0.5, (ks[-1], COL[ax]): 0.5}, mid - tol - cur, mid + tol - cur, w)
            for ki, kj in zip(ks, ks[1:]):
                _, hi_i = projection(sc.bodies[ki], state.body_scale(ki), u)
                lo_j, _ = projection(sc.bodies[kj], state.body_scale(kj), u)
                cur = sgn * float(_c(sc, kj)[ax] - _c(sc, ki)[ax]) + lo_j - hi_i
                _row(out, {(kj, COL[ax]): sgn, (ki, COL[ax]): -sgn}, gap - cur, INF, w)
            if even:
                even_rows(out, sc, ks, ax, tol, w)
            return out
        gens.append(gen)
    elif n == "align":
        ks = [idx[b] for b in st.args]
        ax = 0 if kwd(st, "axis") == "x" else 1
        edge, tol = kwd(st, "edge"), float(kwd(st, "tol"))
        u = np.zeros(2); u[ax] = 1.0

        def gen(state):
            out, sc = [], state.scene

            def q(k):
                if edge == "center":
                    return float(_c(sc, k)[ax])
                lo, hi = projection(sc.bodies[k], state.body_scale(k), u)
                return float(_c(sc, k)[ax]) + (lo if edge == "min" else hi)
            q0 = q(ks[0])
            for k in ks[1:]:
                cur = q(k) - q0
                _row(out, {(k, COL[ax]): 1.0, (ks[0], COL[ax]): -1.0}, -tol - cur, tol - cur, w)
            return out
        gens.append(gen)
    elif n == "evenly_spaced":
        ks = [idx[b] for b in st.args]
        ax = 0 if kwd(st, "axis") == "x" else 1
        tol = float(kwd(st, "tol"))

        def gen(state):
            out = []
            even_rows(out, state.scene, ks, ax, tol, w)
            return out
        gens.append(gen)
    elif n == "grid":
        ks = [idx[b] for b in st.args]
        cols = int(kw["cols"])
        tol = float(kwd(st, "tol"))
        pitch = {0: kw.get("dx"), 1: kw.get("dy")}
        cell = [divmod(m, cols) for m in range(len(ks))]          # (row, column) of each member
        # the member that sets the shared pitch along x (first of the second row) and y (second of the first row)
        ref = {0: cols if len(ks) > cols else None, 1: 1 if cols > 1 and len(ks) > 1 else None}

        def gen(state):
            out, sc = [], state.scene
            k0 = ks[0]
            for ax in (0, 1):
                for m in range(1, len(ks)):
                    t = cell[m][0] if ax == 0 else cell[m][1]          # multiples of the pitch along this axis
                    cur = float(_c(sc, ks[m])[ax] - _c(sc, k0)[ax])
                    co = {}
                    _acc(co, (ks[m], COL[ax]), 1.0); _acc(co, (k0, COL[ax]), -1.0)
                    if pitch[ax] is not None:
                        cur -= t * float(pitch[ax])
                    elif ref[ax] is not None and t:
                        if m == ref[ax]:
                            continue                                    # it defines the pitch
                        kr = ks[ref[ax]]
                        cur -= t * float(_c(sc, kr)[ax] - _c(sc, k0)[ax])
                        _acc(co, (kr, COL[ax]), -float(t)); _acc(co, (k0, COL[ax]), float(t))
                    _row(out, co, -tol - cur, tol - cur, w)
            return out
        gens.append(gen)
    elif n == "ring":
        ks = [idx[b] for b in st.args]
        r, tol, nk = float(kw["r"]), float(kwd(st, "tol")), len(ks)
        centre = np.array([float(kw["x"]), float(kw["y"])]) if "x" in kw else None
        start = math.radians(float(kw["angle"])) if "angle" in kw else None
        face = kw.get("face")
        sgn = 1.0 if face == "in" else -1.0
        axes = {k: body_axis(sc0.bodies[k], kwd(st, "axis"))[0] for k in ks} if face else {}
        turn = math.radians(float(OP_DEFAULTS["facing"]["tol"]))

        def gen(state):
            out, sc = [], state.scene
            P = np.array([_c(sc, k) for k in ks])
            c = centre if centre is not None else P.mean(0)
            U = ring_directions(P, c, start)
            for i, k in enumerate(ks):
                for ax in (0, 1):                  # the member at c + r u_i; a free centre is the members' mean
                    co = {(k, COL[ax]): 1.0}
                    if centre is None:
                        for kj in ks:
                            _acc(co, (kj, COL[ax]), -1.0 / nk)
                    cur = float(P[i, ax] - c[ax] - r * U[i, ax])
                    _row(out, co, -tol - cur, tol - cur, w)
                if face:                           # its axis toward (in) or away from (out) the centre, as facing
                    phi = heading(sc.bodies[k].rotation, axes[k])
                    d = sgn * (c - P[i])
                    rho2 = float(d @ d)
                    if phi is None or rho2 < 1e-12:
                        continue
                    err = wrap(math.atan2(d[1], d[0]) - phi)
                    g = sgn * np.array([-d[1], d[0]]) / rho2
                    co = {(k, "yaw"): 1.0}
                    _acc(co, (k, "x"), float(g[0])); _acc(co, (k, "y"), float(g[1]))
                    if centre is None:
                        for kj in ks:
                            _acc(co, (kj, "x"), -float(g[0]) / nk); _acc(co, (kj, "y"), -float(g[1]) / nk)
                    _row(out, co, err - turn, err + turn, w)
            return out
        gens.append(gen)
    elif n == "cluster":
        ks = [idx[b] for b in st.args]
        r = float(kw["r"])

        def gen(state):
            out, sc = [], state.scene
            P = np.array([_c(sc, k) for k in ks])
            mu = P.mean(0)
            for i, k in enumerate(ks):
                d = P[i] - mu
                rho = float(np.linalg.norm(d))
                if rho < 1e-9:
                    continue
                u = d / rho
                co = {}
                for j, kj in enumerate(ks):
                    f = (1.0 if j == i else 0.0) - 1.0 / len(ks)
                    _acc(co, (kj, "x"), f * float(u[0])); _acc(co, (kj, "y"), f * float(u[1]))
                _row(out, co, -INF, r - rho, w)
            return out
        gens.append(gen)
    elif n == "symmetric":
        a, b, c = (idx[x] for x in st.args)
        ax = 0 if kwd(st, "axis") == "x" else 1
        pa, tol = 1 - ax, float(kwd(st, "tol"))

        def gen(state):
            out, sc = [], state.scene
            cur = float(_c(sc, a)[ax] + _c(sc, b)[ax] - 2.0 * _c(sc, c)[ax])
            co = {}
            _acc(co, (a, COL[ax]), 1.0); _acc(co, (b, COL[ax]), 1.0); _acc(co, (c, COL[ax]), -2.0)
            _row(out, co, -tol - cur, tol - cur, w)
            cur = float(_c(sc, a)[pa] - _c(sc, b)[pa])
            _row(out, {(a, COL[pa]): 1.0, (b, COL[pa]): -1.0}, -tol - cur, tol - cur, w)
            return out
        gens.append(gen)
    elif n == "between":
        a, b, c = (idx[x] for x in st.args)
        ax = 0 if kwd(st, "axis") == "x" else 1
        gap = float(kwd(st, "gap"))
        lo_k, hi_k = (b, c) if sc0.bodies[b].center[ax] <= sc0.bodies[c].center[ax] else (c, b)   # the order the scene starts in

        def gen(state):
            out, sc = [], state.scene
            cur = float(_c(sc, a)[ax] - _c(sc, lo_k)[ax])
            _row(out, {(a, COL[ax]): 1.0, (lo_k, COL[ax]): -1.0}, gap - cur, INF, w)
            cur = float(_c(sc, hi_k)[ax] - _c(sc, a)[ax])
            _row(out, {(hi_k, COL[ax]): 1.0, (a, COL[ax]): -1.0}, gap - cur, INF, w)
            return out
        gens.append(gen)
    elif n in ("centered", "offset", "attach"):
        a, b = idx[st.args[0]], idx[st.args[1]]
        tol = float(kwd(st, "tol"))
        local = n == "attach" or kwd(st, "frame") == "local"
        if n == "attach":
            hb, ha = heading(sc0.bodies[b].rotation), heading(sc0.bodies[a].rotation)
            hb = 0.0 if hb is None else hb
            c, s_ = math.cos(hb), math.sin(hb)
            d0 = sc0.bodies[a].center[:2] - sc0.bodies[b].center[:2]
            o = np.array([c * d0[0] + s_ * d0[1], -s_ * d0[0] + c * d0[1]])     # the offset in b's frame
            st.ref = {"offset": o, "turn": None if ha is None else wrap(ha - hb)}
        elif n == "offset":
            o = np.array([float(kwd(st, "dx")), float(kwd(st, "dy"))])
        else:
            o = np.zeros(2)
        tol_turn = math.radians(float(OP_DEFAULTS["attach"]["deg"]))

        def gen(state):
            out, sc = [], state.scene
            ow, jw = o, np.zeros(2)
            hb = heading(sc.bodies[b].rotation) if local else None
            if local and hb is not None:
                c, s_ = math.cos(hb), math.sin(hb)
                ow = np.array([c * o[0] - s_ * o[1], s_ * o[0] + c * o[1]])
                jw = np.array([-ow[1], ow[0]])                     # d(offset in the world)/d(b's heading)
            res = _c(sc, a) - _c(sc, b) - ow
            for ax in (0, 1):
                co = {(a, COL[ax]): 1.0, (b, COL[ax]): -1.0}
                if local and abs(jw[ax]) > 1e-12:
                    co[(b, "yaw")] = -float(jw[ax])
                _row(out, co, -tol - float(res[ax]), tol - float(res[ax]), w)
            if n == "attach" and st.ref["turn"] is not None:
                ha = heading(sc.bodies[a].rotation)
                if ha is not None and hb is not None:
                    err = wrap(st.ref["turn"] - (ha - hb))
                    _row(out, {(a, "yaw"): 1.0, (b, "yaw"): -1.0}, err - tol_turn, err + tol_turn, w)
            return out
        gens.append(gen)
    elif n == "keep_out":
        ref = st.args[1].split(".")[0] if len(st.args) > 1 else None
        ks = [idx[x] for x in ctx.bodies(st.args[0], exclude=(ref,) if ref else ())]
        gap = float(kwd(st, "gap"))
        dyn = ref is not None and not sc0[ref].fixed
        kr = idx[ref] if ref is not None else None
        tok = st.args[1] if len(st.args) > 1 else None
        static = None if dyn else region_rect(*reference_rect(sc0, tok), kw, repr(st))

        def gen(state):
            out, sc = [], state.scene
            x0, x1, y0, y1 = region_rect(*reference_rect(sc, tok), kw, repr(st)) if dyn else static
            x0, x1, y0, y1 = x0 - gap, x1 + gap, y0 - gap, y1 + gap
            for k in ks:
                s = state.body_scale(k)
                px0, px1 = projection(sc.bodies[k], s, (1.0, 0.0)); py0, py1 = projection(sc.bodies[k], s, (0.0, 1.0))
                cx, cy = _c(sc, k)
                options = []                                         # (column, sense, motion still needed)
                if x0 > -INF: options.append(("x", -1.0, cx + px1 - x0))
                if x1 < INF: options.append(("x", +1.0, x1 - (cx + px0)))
                if y0 > -INF: options.append(("y", -1.0, cy + py1 - y0))
                if y1 < INF: options.append(("y", +1.0, y1 - (cy + py0)))
                if not options:
                    continue
                # a side that leaves the body no room inside its own required regions is not a way out
                own = [r for r, _, wr in ctx.regions.get(sc.bodies[k].name, []) if wr is None]
                fits = [o for o in options if all(_room(o, rect, (x0, x1, y0, y1), (px0, px1, py0, py1)) for rect in own)]
                col, sense, need = min(fits or options, key=lambda o: o[2])
                co = {(k, col): 1.0}
                if dyn:
                    _acc(co, (kr, col), -1.0)
                if sense < 0:
                    _row(out, co, -INF, -need, w)
                else:
                    _row(out, co, need, INF, w)
            return out
        gens.append(gen)
    elif n == "flush":
        ref = st.args[1].split(".")[0]
        if not sc0[ref].fixed:
            raise StatementError(f"{st!r}: flush is against a fixed body, its top or one of its levels")
        ks = [idx[x] for x in ctx.bodies(st.args[0], exclude=(ref,))]
        tol = float(kwd(st, "tol"))
        col, target, low = flush_target(sc0, st)
        u = (1.0, 0.0) if col == 0 else (0.0, 1.0)

        def gen(state):
            out, sc = [], state.scene
            for k in ks:
                e, de = edge_of(sc.bodies[k], state.body_scale(k), u, low)
                err = target - (float(_c(sc, k)[col]) + e)
                _row(out, {(k, COL[col]): 1.0, (k, "yaw"): de}, err - tol, err + tol, w)
            return out
        gens.append(gen)
    elif n == "approach":
        ks = [idx[x] for x in ctx.bodies(st.args[0])]
        length, half = float(kwd(st, "length")), 0.5 * float(kwd(st, "width"))
        toward = kwd(st, "toward")
        base = np.array([float(kwd(st, "x")), float(kwd(st, "y"))])
        fixed_dir = {"near": (-1.0, 0.0), "far": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}.get(toward)
        others = {k: approach_obstacles(ctx, k) for k in ks}
        m = CORRIDOR_MARGIN

        def gen(state):
            out, sc = [], state.scene
            for k in ks:
                ca = _c(sc, k)
                if fixed_dir is not None:
                    u = np.array(fixed_dir)
                else:
                    d = base - ca
                    nd = float(np.linalg.norm(d))
                    if nd < 1e-6:
                        continue
                    u = d / nd
                v = np.array([-u[1], u[0]])
                amin, amax = projection(sc.bodies[k], state.body_scale(k), u)
                s1 = amax + length
                for j in others[k]:
                    q = _c(sc, j) - ca
                    al0, al1 = projection(sc.bodies[j], state.body_scale(j), u)
                    la0, la1 = projection(sc.bodies[j], state.body_scale(j), v)
                    U0, U1 = float(q @ u) + al0, float(q @ u) + al1
                    V0, V1 = float(q @ v) + la0, float(q @ v) + la1
                    options = [(v, +1.0, half + m - V0), (v, -1.0, V1 + half + m),     # beside the corridor
                               (u, +1.0, s1 + m - U0), (u, -1.0, U1 - (amin - m))]     # beyond its end, behind a
                    dirv, sense, need = min(options, key=lambda o: o[2])
                    co = {}
                    for col, val in ((0, dirv[0]), (1, dirv[1])):
                        if abs(val) > 1e-12:
                            _acc(co, (j, COL[col]), float(val)); _acc(co, (k, COL[col]), -float(val))
                    if sense > 0:
                        _row(out, co, need, INF, w)
                    else:
                        _row(out, co, -INF, -need, w)
            return out
        gens.append(gen)
    elif n == "max_move":
        ks = [idx[x] for x in ctx.bodies(st.args[0])]
        d = float(kw["d"])
        st.ref = {"center": {k: sc0.bodies[k].center[:2].copy() for k in ks}}
        normals = [np.array([math.cos(2 * math.pi * (i + 0.5) / POLYGON), math.sin(2 * math.pi * (i + 0.5) / POLYGON)]) for i in range(POLYGON)]
        apothem = d * math.cos(math.pi / POLYGON)

        def gen(state):
            out = []
            for k in ks:
                delta = _c(state.scene, k) - st.ref["center"][k]
                for nv in normals:
                    _row(out, {(k, "x"): float(nv[0]), (k, "y"): float(nv[1])}, -INF, apothem - float(nv @ delta), w)
            return out
        gens.append(gen)
    elif n == "balanced_on":
        a, b = idx[st.args[0]], idx[st.args[1]]
        margin = float(kwd(st, "margin"))
        com = np.asarray(sc0.bodies[a].meta.get("com", np.zeros(3)), dtype=np.float64)

        def gen(state):
            # in b's own frame, where b's extents do not change as it turns: r = Rb^T (com - cb)
            out, sc = [], state.scene
            ba, bb = sc.bodies[a], sc.bodies[b]
            cw = (ba.rotation @ (state.body_scale(a) * com))[:2]
            jw = np.array([-cw[1], cw[0]])                         # d(centre of mass)/d(a's heading)
            hb = heading(bb.rotation) or 0.0
            c, s_ = math.cos(hb), math.sin(hb)
            RT = np.array([[c, s_], [-s_, c]])
            r = RT @ (_c(sc, a) + cw - _c(sc, b))
            dr_db = np.array([r[1], -r[0]])                        # d r / d(b's heading)
            for i in (0, 1):
                lo, hi = projection(bb, state.body_scale(b), RT[i])
                co = {}
                for col in (0, 1):
                    if abs(RT[i, col]) > 1e-12:
                        _acc(co, (a, COL[col]), float(RT[i, col])); _acc(co, (b, COL[col]), -float(RT[i, col]))
                if abs(RT[i] @ jw) > 1e-12:
                    co[(a, "yaw")] = float(RT[i] @ jw)
                if abs(dr_db[i]) > 1e-12:
                    _acc(co, (b, "yaw"), float(dr_db[i]))
                if hi - lo > 2.0 * margin:
                    _row(out, co, lo + margin - float(r[i]), hi - margin - float(r[i]), w)
                else:
                    mid = 0.5 * (lo + hi) - float(r[i])
                    _row(out, co, mid, mid, w)
            return out
        gens.append(gen)
    elif n in ("stack",):
        ks = [idx[x] for x in st.args]
        tol = float(kwd(st, "tol"))

        def gen(state):
            out, sc = [], state.scene
            for kb, ka in zip(ks, ks[1:]):
                res = _c(sc, ka) - _c(sc, kb)
                for ax in (0, 1):
                    _row(out, {(ka, COL[ax]): 1.0, (kb, COL[ax]): -1.0}, -tol - float(res[ax]), tol - float(res[ax]), None)
            return out
        gens.append(gen)
    else:
        raise StatementError(f"no rows for {n}")

    def all_rows(state):
        out = []
        for g in gens:
            out.extend(g(state))
        return out
    return all_rows


def even_rows(out, sc, ks, ax, tol, w):
    """Every spacing of consecutive members within tol of the mean spacing (c_last - c_first)/(n-1):
    linear in the centres, one row per spacing."""
    n = len(ks)
    if n < 3:
        return
    c = [float(_c(sc, k)[ax]) for k in ks]
    mean = (c[-1] - c[0]) / (n - 1)
    for i in range(n - 1):
        co = {}
        _acc(co, (ks[i + 1], COL[ax]), 1.0); _acc(co, (ks[i], COL[ax]), -1.0)
        _acc(co, (ks[-1], COL[ax]), -1.0 / (n - 1)); _acc(co, (ks[0], COL[ax]), 1.0 / (n - 1))
        cur = (c[i + 1] - c[i]) - mean
        _row(out, co, -tol - cur, tol - cur, w)


def even_error(values) -> float:
    """Largest deviation of a consecutive spacing from the mean spacing."""
    d = np.diff(np.asarray(values, dtype=np.float64))
    return float(np.abs(d - d.mean()).max()) if len(d) else 0.0


def member_key(b, prop: str) -> float:
    """A body's height, width (shorter side), length (longer side) or mass (its asset's, else its
    volume at unit density) for ordering a group."""
    if prop == "mass":
        m = (b.meta.get("physics") or {}).get("mass")
        if m:
            return float(m)
        import trimesh
        try:
            return float(abs(trimesh.Trimesh(b.verts, b.faces, process=False).volume))
        except Exception:  # noqa: BLE001
            return float(np.prod(np.ptp(b.verts, axis=0)))
    if prop == "height":
        return float(np.ptp(b.world_vertices()[:, 2]))
    ext = sorted(np.ptp(b.verts[:, :2], axis=0))
    return float(ext[0] if prop == "width" else ext[1])


def sorted_members(st, scene) -> list:
    """The members of a group statement, sorted by `order=` when it is given (ties keep the order
    written; a leading '-' sorts descending)."""
    order = st.kw.get("order")
    if not order:
        return list(st.args)
    prop, rev = order.lstrip("-"), order.startswith("-")
    keyed = [(member_key(scene[n], prop), i, n) for i, n in enumerate(st.args)]
    keyed.sort(key=lambda t: ((-t[0] if rev else t[0]), t[1]))
    return [n for _, _, n in keyed]


def _room(option, rect, region, foot) -> bool:
    """Whether a keep-out side leaves the footprint (relative extents `foot`) room inside a required
    rectangle: leaving across x0 means fitting in [rect.x0, x0], and so on."""
    col, sense, _ = option
    rx0, rx1, ry0, ry1 = rect
    x0, x1, y0, y1 = region
    px0, px1, py0, py1 = foot
    if col == "x":
        lo, hi = (rx0, min(rx1, x0)) if sense < 0 else (max(rx0, x1), rx1)
        return hi - lo >= px1 - px0
    lo, hi = (ry0, min(ry1, y0)) if sense < 0 else (max(ry0, y1), ry1)
    return hi - lo >= py1 - py0


FLUSH_EDGE = {"near": (0, True), "far": (0, False), "right": (1, True), "left": (1, False)}   # (axis, the low edge)


def flush_target(scene, st):
    """(axis, the coordinate the footprint's edge goes to, whether that is its low edge) of flush(a, b, side, gap):
    the edge on `side` sits `gap` inside the same edge of b's rectangle (its bounding box, or the level it names)."""
    x0, x1, y0, y1 = region_rect(*reference_rect(scene, st.args[1]), {}, repr(st))
    col, low = FLUSH_EDGE[st.kw["side"]]
    gap = float(kwd(st, "gap"))
    lo, hi = ((x0, x1), (y0, y1))[col]
    return col, (lo + gap if low else hi - gap), low


def ring_directions(P: np.ndarray, c: np.ndarray, start: float | None) -> np.ndarray:
    """Unit directions from the centre c of n members evenly spaced counter-clockwise in their order: from the angle
    `start`, or, without it, from the angle that fits the members' present places best (least squares)."""
    n = len(P)
    a = 2.0 * math.pi * np.arange(n) / n
    if start is None:
        d = P - c
        e = np.column_stack([np.cos(a), np.sin(a)])
        start = math.atan2(float(np.sum(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0])), float(np.sum(e * d)))
    return np.column_stack([np.cos(start + a), np.sin(start + a)])


def approach_obstacles(ctx: Ctx, k: int) -> list:
    """Bodies that can block an approach to body k: everything but k, the surfaces (fixed bodies
    tagged support or ground), k's own support and the bodies resting on k."""
    sc = ctx.scene
    name = sc.bodies[k].name
    out = []
    for j, bj in enumerate(sc.bodies):
        if j == k or (bj.fixed and ({"support", "ground"} & bj.tags)):
            continue
        if ctx.supports.get(name) == bj.name or ctx.supports.get(bj.name) == name:
            continue
        out.append(j)
    return out


def check_v1(st, scene, supports: dict, tol: float, angle_tol: float, bodies_fn):
    """[(name, ok, value)] of a v1 statement on a final scene; `tol` (m) and `angle_tol` (degrees)
    are the slack of the check on top of the statement's own tolerance."""
    n, kw = st.name, st.kw
    out = []
    sc = scene
    if n == "reach":
        r, rmin = float(kw["r"]), float(kwd(st, "min"))
        base = np.array([float(kwd(st, "x")), float(kwd(st, "y"))])
        for nm in bodies_fn(st.args[0]):
            rho = float(np.linalg.norm(sc[nm].center[:2] - base))
            out.append((f"reach({nm},r={r:g})", rmin - tol <= rho <= r + tol, rho))
    elif n == "at":
        tol_s = float(kwd(st, "tol"))
        for nm in bodies_fn(st.args[0]):
            c = sc[nm].center[:2]
            v = max(abs(float(c[0]) - float(kw["x"])) if "x" in kw else 0.0, abs(float(c[1]) - float(kw["y"])) if "y" in kw else 0.0)
            out.append((f"at({nm})", v <= tol_s + tol, v))
    elif n == "beside":
        u = np.array(SIDE_VEC[kw["side"]])
        a, b = sc[st.args[0]], sc[st.args[1]]
        g = float((a.center[:2] - b.center[:2]) @ u) + projection(a, 1.0, u)[0] - projection(b, 1.0, u)[1]
        top = float(kw["max"]) if "max" in kw else INF
        out.append((f"beside({st.args[0]},{st.args[1]},{kw['side']})", float(kwd(st, "gap")) - tol <= g <= top + tol, g))
    elif n == "clearance":
        gap = float(kw["gap"])
        for nm in bodies_fn(st.args[0]):
            g, other = clearance_of(sc, nm, gap, supports)
            out.append((f"clearance({nm},gap={gap:g})", g >= gap - tol, g))
    elif n == "approach":
        length, half = float(kwd(st, "length")), 0.5 * float(kwd(st, "width"))
        toward = kwd(st, "toward")
        base = np.array([float(kwd(st, "x")), float(kwd(st, "y"))])
        fixed_dir = {"near": (-1.0, 0.0), "far": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}.get(toward)
        names = sc.names()
        idx = {nm: k for k, nm in enumerate(names)}
        ctx = Ctx(sc, idx, supports, bodies_fn)
        for nm in bodies_fn(st.args[0]):
            k = idx[nm]
            ca = sc[nm].center[:2]
            if fixed_dir is not None:
                u = np.array(fixed_dir)
            else:
                d = base - ca
                if float(np.linalg.norm(d)) < 1e-6:
                    out.append((f"approach({nm})", True, 0.0)); continue
                u = d / float(np.linalg.norm(d))
            v = np.array([-u[1], u[0]])
            amin, amax = projection(sc[nm], 1.0, u)
            corridor = np.array([ca + u * s + v * t for s, t in ((amax, -half), (amax + length, -half), (amax + length, half), (amax, half))])
            worst = -INF
            for j in approach_obstacles(ctx, k):
                worst = max(worst, polygon_overlap(corridor, footprint_hull(sc.bodies[j])))
            worst = max(worst, 0.0) if worst > -INF else 0.0
            out.append((f"approach({nm})", worst <= tol, worst))
    elif n == "facing":
        tol_s = float(kwd(st, "tol"))
        point = sc[st.args[1]].center[:2] if len(st.args) > 1 else np.array([float(kw.get("x", 0.0)), float(kw.get("y", 0.0))])
        for nm in bodies_fn(st.args[0], exclude=st.args[1:2]):
            phi = heading(sc[nm].rotation, body_axis(sc[nm], kwd(st, "axis"))[0])
            d = point - sc[nm].center[:2]
            if phi is None or float(d @ d) < 1e-12:
                out.append((f"facing({nm})", False, None)); continue
            e = abs(math.degrees(wrap(math.atan2(d[1], d[0]) - phi)))
            out.append((f"facing({nm})", e <= tol_s + angle_tol, e))
    elif n in ("yaw", "parallel", "max_turn"):
        tok = kwd(st, "axis") if n != "max_turn" else "x"
        for nm in bodies_fn(st.args[0], exclude=st.args[1:2]):
            vec, line = body_axis(sc[nm], tok)
            phi = heading(sc[nm].rotation, vec)
            if n == "yaw":
                m, tol_s = math.radians(float(kwd(st, "mod"))), float(kwd(st, "tol"))
                m = min(m, math.pi) if line else m
                e = None if phi is None else abs(math.degrees(wrap_mod(math.radians(float(kw["deg"])) - phi, m)))
                label = f"yaw({nm},deg={float(kw['deg']):g})"
            elif n == "parallel":
                m, tol_s = math.radians(float(kwd(st, "mod"))), float(kwd(st, "tol"))
                vb, line_b = body_axis(sc[st.args[1]], tok)
                pb = heading(sc[st.args[1]].rotation, vb)
                m = min(m, math.pi) if (line or line_b) else m
                e = None if phi is None or pb is None else abs(math.degrees(wrap_mod(pb - phi, m)))
                label = f"parallel({nm},{st.args[1]})"
            else:
                tol_s = float(kw["deg"])
                h0 = _ref(st, "heading", sc, nm)
                e = None if phi is None or h0 is None else abs(math.degrees(wrap(h0 - phi)))
                label = f"max_turn({nm},deg={tol_s:g})"
            out.append((label, e is not None and e <= tol_s + angle_tol, e))
    elif n in ("row", "align", "evenly_spaced", "grid", "cluster", "stack"):
        members = sorted_members(st, sc) if n == "row" else list(st.args)
        P = np.array([sc[x].center[:2] for x in members])
        label = f"{n}({','.join(st.args)})"
        if n == "row":
            tok = row_axis(st)
            ax = 0 if tok.endswith("x") else 1
            sgn = -1.0 if tok.startswith("-") else 1.0
            u = np.zeros(2); u[ax] = sgn
            v = np.zeros(2); v[1 - ax] = 1.0
            edge = kwd(st, "edge")
            q = [float(P[i, 1 - ax]) + (0.0 if edge == "center" else projection(sc[x], 1.0, v)[0 if edge == "min" else 1])
                 for i, x in enumerate(members)]
            spread = max(abs(qi - q[0]) for qi in q[1:])
            gaps = [sgn * float(P[i + 1, ax] - P[i, ax]) + projection(sc[members[i + 1]], 1.0, u)[0] - projection(sc[members[i]], 1.0, u)[1]
                    for i in range(len(members) - 1)]
            out.append((label + ":aligned", spread <= float(kwd(st, "tol")) + tol, spread))
            out.append((label + ":order", min(gaps) >= float(kwd(st, "gap")) - tol, min(gaps)))
            if kw.get("spacing") == "even" and len(members) >= 3:
                v = even_error(P[:, ax])
                out.append((label + ":even", v <= float(kwd(st, "tol")) + tol, v))
            if "mid" in kw:
                v = abs(0.5 * float(P[0, ax] + P[-1, ax]) - float(kw["mid"]))
                out.append((label + ":mid", v <= float(kwd(st, "tol")) + tol, v))
        elif n == "align":
            ax = 0 if kwd(st, "axis") == "x" else 1
            u = np.zeros(2); u[ax] = 1.0
            edge = kwd(st, "edge")
            q = [float(P[i, ax]) + (0.0 if edge == "center" else projection(sc[x], 1.0, u)[0 if edge == "min" else 1])
                 for i, x in enumerate(st.args)]
            v = max(abs(qi - q[0]) for qi in q[1:])
            out.append((label, v <= float(kwd(st, "tol")) + tol, v))
        elif n == "evenly_spaced":
            ax = 0 if kwd(st, "axis") == "x" else 1
            v = even_error(P[:, ax])
            out.append((label, v <= float(kwd(st, "tol")) + tol, v))
        elif n == "grid":
            cols = int(kw["cols"])
            cells = np.array([divmod(m, cols) for m in range(len(st.args))], dtype=np.float64)
            v = 0.0
            for ax, key in ((0, "dx"), (1, "dy")):
                t = cells[:, ax]
                if key in kw:
                    res = P[:, ax] - P[0, ax] - t * float(kw[key])
                elif np.ptp(t) > 0:                                  # the best common pitch: least squares through the first member
                    p = float(np.dot(t, P[:, ax] - P[0, ax]) / np.dot(t, t))
                    res = P[:, ax] - P[0, ax] - t * p
                else:
                    res = P[:, ax] - P[0, ax]
                v = max(v, float(np.abs(res).max()))
            out.append((label, v <= float(kwd(st, "tol")) + tol, v))
        elif n == "cluster":
            v = float(np.linalg.norm(P - P.mean(0), axis=1).max())
            out.append((label, v <= float(kw["r"]) + tol, v))
        else:
            tol_s = float(kwd(st, "tol"))
            for lower, upper in zip(st.args, st.args[1:]):
                v = float(np.abs(sc[upper].center[:2] - sc[lower].center[:2]).max())
                out.append((f"stack({upper},{lower}):centered", v <= tol_s + tol, v))
    elif n == "symmetric":
        a, b, c = (sc[x].center[:2] for x in st.args)
        ax = 0 if kwd(st, "axis") == "x" else 1
        v = max(abs(float(a[ax] + b[ax] - 2.0 * c[ax])), abs(float(a[1 - ax] - b[1 - ax])))
        out.append((f"symmetric({','.join(st.args)})", v <= float(kwd(st, "tol")) + tol, v))
    elif n == "between":
        a, b, c = (sc[x].center[:2] for x in st.args)
        ax = 0 if kwd(st, "axis") == "x" else 1
        lo, hi = min(b[ax], c[ax]), max(b[ax], c[ax])
        v = min(float(a[ax] - lo), float(hi - a[ax])) - float(kwd(st, "gap"))
        out.append((f"between({','.join(st.args)})", v >= -tol, v))
    elif n in ("centered", "offset", "attach"):
        a, b = sc[st.args[0]], sc[st.args[1]]
        if n == "centered":
            o, local = np.zeros(2), False
        elif n == "offset":
            o, local = np.array([float(kwd(st, "dx")), float(kwd(st, "dy"))]), kwd(st, "frame") == "local"
        else:
            o, local = _ref(st, "offset", sc, st.args[0]), True
        ow = o
        hb = heading(b.rotation)
        if local and hb is not None:
            c, s_ = math.cos(hb), math.sin(hb)
            ow = np.array([c * o[0] - s_ * o[1], s_ * o[0] + c * o[1]])
        v = float(np.abs(a.center[:2] - b.center[:2] - ow).max())
        out.append((f"{n}({st.args[0]},{st.args[1]})", v <= float(kwd(st, "tol")) + tol, v))
        if n == "attach":
            turn = _ref(st, "turn", sc, st.args[0])
            ha = heading(a.rotation)
            if turn is not None and ha is not None and hb is not None:
                e = abs(math.degrees(wrap(turn - (ha - hb))))
                out.append((f"attach({st.args[0]},{st.args[1]}):turn", e <= float(OP_DEFAULTS["attach"]["deg"]) + angle_tol, e))
    elif n == "keep_out":
        ref = st.args[1].split(".")[0] if len(st.args) > 1 else None
        gap = float(kwd(st, "gap"))
        x0, x1, y0, y1 = region_rect(*reference_rect(sc, st.args[1] if ref else None), kw, repr(st))
        big = 1e3                                                     # an open side of the region, far beyond the scene
        R = rect_polygon(max(x0, -big) - gap, min(x1, big) + gap, max(y0, -big) - gap, min(y1, big) + gap)
        for nm in bodies_fn(st.args[0], exclude=(ref,) if ref else ()):
            v = max(polygon_overlap(R, footprint_hull(sc[nm])), 0.0)
            out.append((f"keep_out({nm},{st.args[1] if ref else 'region'})", v <= tol, v))
    elif n == "flush":
        col, target, low = flush_target(sc, st)
        u = (1.0, 0.0) if col == 0 else (0.0, 1.0)
        for nm in bodies_fn(st.args[0], exclude=(st.args[1].split(".")[0],)):
            lo, hi = projection(sc[nm], 1.0, u)
            v = abs(target - (float(sc[nm].center[col]) + (lo if low else hi)))
            out.append((f"flush({nm},{st.args[1]},{kw['side']})", v <= float(kwd(st, "tol")) + tol, v))
    elif n == "ring":
        P = np.array([sc[x].center[:2] for x in st.args])
        c = np.array([float(kw["x"]), float(kw["y"])]) if "x" in kw else P.mean(0)
        U = ring_directions(P, c, math.radians(float(kw["angle"])) if "angle" in kw else None)
        v = float(np.abs(P - c - float(kw["r"]) * U).max())
        label = f"ring({','.join(st.args)})"
        out.append((label, v <= float(kwd(st, "tol")) + tol, v))
        if kw.get("face"):
            sgn = 1.0 if kw["face"] == "in" else -1.0
            for i, nm in enumerate(st.args):
                phi = heading(sc[nm].rotation, body_axis(sc[nm], kwd(st, "axis"))[0])
                d = sgn * (c - P[i])
                e = None if phi is None or float(d @ d) < 1e-12 else abs(math.degrees(wrap(math.atan2(d[1], d[0]) - phi)))
                out.append((f"{label}:face({nm})", e is not None and e <= float(OP_DEFAULTS["facing"]["tol"]) + angle_tol, e))
    elif n == "max_move":
        d = float(kw["d"])
        for nm in bodies_fn(st.args[0]):
            c0 = _ref(st, "center", sc, nm)
            v = float(np.linalg.norm(sc[nm].center[:2] - c0))
            out.append((f"max_move({nm},d={d:g})", v <= d + tol, v))
    elif n == "balanced_on":
        a, b = sc[st.args[0]], sc[st.args[1]]
        margin = float(kwd(st, "margin"))
        com = np.asarray(a.meta.get("com", np.zeros(3)), dtype=np.float64)
        g = a.center[:2] + (a.rotation @ com)[:2]
        hb = heading(b.rotation) or 0.0
        c, s_ = math.cos(hb), math.sin(hb)
        RT = np.array([[c, s_], [-s_, c]])
        r = RT @ (g - b.center[:2])
        v = 0.0
        for i in (0, 1):
            lo, hi = projection(b, 1.0, RT[i])
            v = max(v, lo + margin - float(r[i]), float(r[i]) - (hi - margin))
        out.append((f"balanced_on({st.args[0]},{st.args[1]})", v <= tol, float(v)))
    else:
        raise StatementError(f"no predicate for {n}")
    return out


def _ref(st, key, scene, name):
    """A reference pose the compiler recorded on the statement (where a body started), by body name."""
    ref = getattr(st, "ref", None)
    if not ref or key not in ref:
        raise StatementError(f"{st!r}: compile the program on the scene it starts from before checking it")
    val = ref[key]
    if isinstance(val, dict):
        k = scene.names().index(name)
        return val[k]
    return val


def clearance_of(scene, name: str, gap: float, supports: dict):
    """(smallest mesh distance from body `name` to any other body, that body's name), leaving out its
    support, the bodies resting on it and fixed-fixed pairs; inf when nothing is within gap + 1 cm."""
    from ..gates.verify import pair_signed_distances
    b = scene[name]
    lo, hi = b.world_aabb()
    best, who = INF, None
    for o in scene.bodies:
        if o is b or (o.fixed and b.fixed):
            continue
        if supports.get(name) == o.name or supports.get(o.name) == name:
            continue
        olo, ohi = o.world_aabb()
        if float(np.maximum(lo - ohi, olo - hi).max()) > gap + 0.01:
            continue                                                  # certified farther than gap by the boxes
        pairs = pair_signed_distances([b, o], prefilter=gap + 0.01, skip_fixed_pairs=False)
        for _, _, s, *_ in pairs:
            if s < best:
                best, who = float(s), o.name
    return best, who
