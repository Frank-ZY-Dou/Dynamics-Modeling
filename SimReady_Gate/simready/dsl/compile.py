"""Compile a DSL program against a scene into a RepairSpec and final predicates.

Frame convention = RoboLab's (robot at the origin looking along +x, table in front of it):
  left_of(a,b)     : a.y >= b.y + gap        right_of(a,b)  : a.y <= b.y - gap
  in_front_of(a,b) : a.x >= b.x + gap        behind(a,b)    : a.x <= b.x - gap
(RoboLab predicates.md: left-of = +Y, front-of = +X.)

Per-step rows are linear in (dx, dy, dyaw) of free bodies. Region rows use the exact
projection interval of the scaled, yawed footprint (not a symmetric half-extent), and regions
attached to a free reference body are re-evaluated from that body's current pose every step.
Defaults for omitted keyword arguments come from dsl.schema.DEFAULTS in both the text and the
JSON path.
"""
from __future__ import annotations

import math
import numpy as np

from ..scene.model import Scene, Body
from ..repair.upright_s4r import RepairSpec, decompose_zyx
from .model import Program, Statement
from .schema import DEFAULTS

AXIS = {"x": 0, "y": 1}
REL = {"left_of": ("y", +1), "right_of": ("y", -1), "in_front_of": ("x", +1), "behind": ("x", -1)}
BINARY = {"on_support", "within", "inside", "min_distance", "near"} | set(REL)
PLACE_TOL = 0.05     # a placed body that ended within 5 cm of its target counts as placed


class CompileError(ValueError):
    pass


def _bodies(scene: Scene, tok: str, exclude=()):
    if tok == "*":
        return [b.name for b in scene.free() if b.name not in exclude]
    if tok not in scene.names():
        raise CompileError(f"unknown body '{tok}' (known: {', '.join(scene.names())})")
    return [tok]


def _ref(tok: str):
    return tok.split(".")[0]


def _rect_of(scene: Scene, tok: str, inset: float):
    """XY rectangle of `name` or `name.top` from the body's world AABB, inset."""
    lo, hi = scene[_ref(tok)].world_aabb()
    r = (lo[0] + inset, hi[0] - inset, lo[1] + inset, hi[1] - inset)
    if r[0] > r[1] or r[2] > r[3]:
        raise CompileError(f"inset {inset} leaves no room inside {tok}")
    return r


def _footprint_interval(b: Body, s: float, axis: int):
    """[min, max] of the body's projection on world axis x/y at continuation scale s,
    relative to its reference centre (exact for any footprint shape)."""
    p = (b.rotation @ (s * b.verts).T).T[:, axis]
    return float(p.min()), float(p.max())


def _low_footprint(b: Body):
    """XY of the body's lowest vertices (within 10 % of its height above the minimum) — where it
    will touch a surface — used to seat it on the actual support surface."""
    w = b.world_vertices()
    z = w[:, 2]
    band = z <= z.min() + 0.1 * max(z.max() - z.min(), 1e-6)
    return w[band][:, :2]


def support_height_for(scene: Scene, body: Body, support: Body, require_hit: bool = False):
    """Seat height of `body` on `support` from the support surface under the body's bottom.

    With `require_hit` the height is taken only from downward rays under the body's lowest
    vertices and its centre; when none of them meets the support the result is None (the body
    is not over the support at all). Without it, the vertex fallback in `Scene.support_height`
    supplies an initial guess for the repair, which is never used as evidence."""
    if require_hit:
        pts = np.vstack([_low_footprint(body), body.center[None, :2]])
        hits = scene._surface_hits(support, pts)
        return float(hits.max()) if hits.size else None
    return scene.support_height(support, at=body.center[:2], radius=0.5 * float(np.linalg.norm(body.world_aabb()[1][:2] - body.world_aabb()[0][:2])) + 0.02,
                                footprint=_low_footprint(body))


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
        if st.name in BINARY:
            if len(st.args) < 2:
                raise CompileError(f"{st.name} needs two arguments: {st!r}")
            if st.args[0] == _ref(st.args[1]) and st.args[0] != "*":
                raise CompileError(f"{st.name} relates a body to itself: {st!r}")
            for extra in st.kw:
                if extra not in ("gap", "axis", "inset", "r", "margin", "x", "y", "yaw", "w"):
                    raise CompileError(f"unknown keyword '{extra}' in {st!r}")
        if st.name in REL and st.kw.get("axis", REL[st.name][0]) not in AXIS:
            raise CompileError(f"axis must be x or y: {st!r}")
    rects_static = {}   # body -> [(rect, label)] for fixed references
    rects_dyn = {}      # body -> [(ref_body, inset, label)] for free references
    rels = []           # (a, b, axis, sign, gap)
    dists = []          # (a, b, kind, r)
    dyn_support = []    # (body, free support)
    for st in program.statements:
        n = st.name
        if n in ("fixed", "minimize"):
            continue
        if n == "no_penetration":
            spec.d_hat = float(st.kw.get("margin", DEFAULTS["margin"]))
        elif n == "on_support":
            sup = _ref(st.args[1])
            if sup not in scene.names():
                raise CompileError(f"unknown support '{sup}'")
            for nm in _bodies(scene, st.args[0], exclude=(sup,)):
                b = scene[nm]
                spec.supports[nm] = support_height_for(scene, b, scene[sup])
                spec.support_of[nm] = sup
                if not scene[sup].fixed:
                    dyn_support.append((nm, sup))
        elif n == "upright":
            spec.upright.update(_bodies(scene, st.args[0]))
        elif n in ("within", "inside"):
            inset = float(st.kw.get("inset", DEFAULTS["inset"] if n == "within" else 0.0))
            ref = _ref(st.args[1])
            if ref not in scene.names():
                raise CompileError(f"unknown region body '{ref}'")
            if n == "inside":
                lo, hi = scene[ref].world_aabb()
                inset = max(inset, 0.1 * float(min(hi[0] - lo[0], hi[1] - lo[1])))
            for nm in _bodies(scene, st.args[0], exclude=(ref,)):
                if scene[ref].fixed:
                    rects_static.setdefault(nm, []).append((_rect_of(scene, ref, inset), n))
                else:
                    rects_dyn.setdefault(nm, []).append((ref, inset, n))
        elif n in REL:
            axis, sign = REL[n]
            axis = st.kw.get("axis", axis)
            gap = float(st.kw.get("gap", DEFAULTS["gap"]))
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                rels.append((a, st.args[1], axis, sign, gap))
        elif n in ("min_distance", "near"):
            if "r" not in st.kw:
                raise CompileError(f"{n} needs r=: {st!r}")
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                dists.append((a, st.args[1], n, float(st.kw["r"])))
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
        else:
            raise CompileError(f"unsupported statement {n}")
    for a, b, *_ in rels + dists:
        if b not in scene.names():
            raise CompileError(f"unknown body '{b}'")
    idx = {b.name: k for k, b in enumerate(scene.bodies)}

    def region_rows(out, sc, k, b, s, xmin, xmax, ymin, ymax, ref_k=None):
        px0, px1 = _footprint_interval(b, s, 0); py0, py1 = _footprint_interval(b, s, 1)
        cx, cy = b.center[0], b.center[1]
        if ref_k is None:
            out.append(({(k, "x"): 1.0}, xmin - px0 - cx, xmax - px1 - cx))
            out.append(({(k, "y"): 1.0}, ymin - py0 - cy, ymax - py1 - cy))
        else:   # the region moves with the free reference body: rows couple both columns
            out.append(({(k, "x"): 1.0, (ref_k, "x"): -1.0}, xmin - px0 - cx, xmax - px1 - cx))
            out.append(({(k, "y"): 1.0, (ref_k, "y"): -1.0}, ymin - py0 - cy, ymax - py1 - cy))

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
            for (xmin, xmax, ymin, ymax), _ in lst:
                region_rows(out, sc, k, b, s, xmin, xmax, ymin, ymax)
        for nm, lst in rects_dyn.items():
            k = idx[nm]; b = sc.bodies[k]
            if b.fixed:
                continue
            s = state.body_scale(k)
            for ref, inset, _ in lst:
                xmin, xmax, ymin, ymax = _rect_of(sc, ref, inset)
                region_rows(out, sc, k, b, s, xmin, xmax, ymin, ymax, ref_k=idx[ref])
        for a, bname, axis, sign, gap in rels:
            ka, kb = idx[a], idx[bname]
            ax = AXIS[axis]
            cur = sc.bodies[ka].center[ax] - sc.bodies[kb].center[ax]
            out.append(({(ka, axis): sign, (kb, axis): -sign}, gap - sign * cur, np.inf))   # sign*(cur + da - db) >= gap
        for a, bname, kind, r in dists:
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
                out.append((coeffs, r - dist, np.inf))
            else:
                out.append((coeffs, -np.inf, r - dist))
        return out

    spec.rows_fn = rows_fn
    return spec


def check_predicates(program: Program, scene: Scene, tol: float = 2e-3, upright_deg: float = 1.0, place_tol: float = PLACE_TOL):
    """Evaluate every statement on the final scene. Returns [(statement, ok, value)].
    on_support is judged by the body's lowest point against the support SURFACE under its bottom
    (downward rays), a geometric fact independent of the numbers the repair used; a body whose
    bottom is over no part of the support fails with value None, whatever its height.
    `tol` (m), `upright_deg` and `place_tol` (m) are the tolerances; the repair's verdict uses the
    defaults, a check on settled poses passes looser ones."""
    out = []
    for st in program.statements:
        n = st.name
        if n == "on_support":
            sup = _ref(st.args[1])
            for nm in _bodies(scene, st.args[0], exclude=(sup,)):
                b = scene[nm]
                h = support_height_for(scene, b, scene[sup], require_hit=True)
                if h is None:
                    out.append((f"on_support({nm},{sup})", False, None))
                    continue
                g = float((b.world_vertices() @ scene.up).min()) - h
                out.append((f"on_support({nm},{sup})", abs(g) <= tol, g))
        elif n == "upright":
            for nm in _bodies(scene, st.args[0]):
                _, roll, pitch = decompose_zyx(scene[nm].rotation)
                t = max(abs(roll), abs(pitch))
                out.append((f"upright({nm})", t <= math.radians(upright_deg), t))
        elif n in ("within", "inside"):
            inset = float(st.kw.get("inset", DEFAULTS["inset"] if n == "within" else 0.0))
            ref = _ref(st.args[1])
            if n == "inside":
                lo, hi = scene[ref].world_aabb()
                inset = max(inset, 0.1 * float(min(hi[0] - lo[0], hi[1] - lo[1])))
            xmin, xmax, ymin, ymax = _rect_of(scene, ref, inset)
            for nm in _bodies(scene, st.args[0], exclude=(ref,)):
                lo, hi = scene[nm].world_aabb()
                viol = max(xmin - lo[0], hi[0] - xmax, ymin - lo[1], hi[1] - ymax, 0.0)
                out.append((f"{n}({nm},{st.args[1]})", viol <= tol, viol))
        elif n in REL:
            axis, sign = REL[n]
            axis = st.kw.get("axis", axis); gap = float(st.kw.get("gap", DEFAULTS["gap"]))
            ax = AXIS[axis]
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                v = sign * (scene[a].center[ax] - scene[st.args[1]].center[ax]) - gap
                out.append((f"{n}({a},{st.args[1]},gap={gap:g})", v >= -tol, v))
        elif n in ("min_distance", "near"):
            r = float(st.kw["r"])
            for a in _bodies(scene, st.args[0], exclude=(st.args[1],)):
                d = float(np.linalg.norm(scene[a].center[:2] - scene[st.args[1]].center[:2]))
                ok = d >= r - tol if n == "min_distance" else d <= r + tol
                out.append((f"{n}({a},{st.args[1]},r={r:g})", ok, d))
        elif n == "place":
            a = st.args[0]
            d = float(np.linalg.norm(scene[a].center[:2] - np.array([float(st.kw["x"]), float(st.kw["y"])])))
            out.append((f"place({a})", d <= place_tol, d))      # a soft target: reported against a loose tolerance
    return out
