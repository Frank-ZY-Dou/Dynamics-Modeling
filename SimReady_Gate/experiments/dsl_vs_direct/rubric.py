"""Checks the tasks are graded with: geometric facts about a final scene, written from each
request's wording and applied in the same way to every condition (the poses a model typed, or the
poses the gate found for a model's program). Nothing here reads a program.

Every check returns (name, ok, value). Lengths in metres, angles in degrees; SLACK and ANGLE_SLACK
are the tolerances a check grants on top of the numbers the request states.
"""
from __future__ import annotations

import math

import numpy as np

from simready.dsl.statements import projection, footprint_hull, polygon_overlap, rect_polygon, heading
from simready.dsl.compile import support_height_for
from simready.gates.verify import verify_scene, pair_signed_distances

SLACK = 0.005
ANGLE_SLACK = 2.0


def _c(sc, n):
    return sc[n].center[:2]


def long_axis_angle(b) -> float:
    """Heading (radians) of the body's longer horizontal local axis."""
    ext = np.ptp(b.verts, axis=0)
    axis = (1.0, 0.0, 0.0) if ext[0] >= ext[1] else (0.0, 1.0, 0.0)
    return heading(b.rotation, axis) or 0.0


def _angle_mod(a, m=math.pi):
    return abs((a + 0.5 * m) % m - 0.5 * m)


def centre_on(sc, n, x=None, y=None, tol=0.01):
    c = _c(sc, n)
    v = max(abs(c[0] - x) if x is not None else 0.0, abs(c[1] - y) if y is not None else 0.0)
    return (f"centre_on({n})", v <= tol + SLACK, v)


def dist_to_base(sc, n, lo=0.0, hi=np.inf, base=(0.0, 0.0)):
    d = float(np.linalg.norm(_c(sc, n) - np.asarray(base)))
    return (f"dist_to_base({n})", lo - SLACK <= d <= hi + SLACK, d)


def long_axis_along(sc, n, axis="x", tol=5.0):
    a = long_axis_angle(sc[n]) - (0.0 if axis == "x" else math.pi / 2)
    v = math.degrees(_angle_mod(a))
    return (f"long_axis_along({n},{axis})", v <= tol + ANGLE_SLACK, v)


def parallel_long(sc, a, b, tol=5.0):
    v = math.degrees(_angle_mod(long_axis_angle(sc[a]) - long_axis_angle(sc[b])))
    return (f"parallel({a},{b})", v <= tol + ANGLE_SLACK, v)


def mirror(sc, a, b, c, axis="y", tol=0.01):
    ax = 0 if axis == "x" else 1
    pa, pb, pc = _c(sc, a), _c(sc, b), _c(sc, c)
    v = max(abs(pa[ax] + pb[ax] - 2 * pc[ax]), abs(pa[1 - ax] - pb[1 - ax]))
    return (f"mirror({a},{b},{c})", v <= tol + SLACK, v)


def footprint_gap(sc, a, b, direction, lo=0.0, hi=np.inf):
    """Gap between the footprints of a and b along `direction` ('+y': b lies on a's +y side)."""
    ax = 0 if direction.endswith("x") else 1
    s = -1.0 if direction.startswith("-") else 1.0
    u = np.zeros(2); u[ax] = s
    _, a_hi = projection(sc[a], 1.0, u)
    b_lo, _ = projection(sc[b], 1.0, u)
    g = s * float(_c(sc, b)[ax] - _c(sc, a)[ax]) + b_lo - a_hi
    return (f"gap({a}->{b},{direction})", lo - SLACK <= g <= hi + SLACK, g)


def side_of(sc, a, b, direction, gap=0.0):
    """a's centre lies on b's `direction` side by at least gap."""
    ax = 0 if direction.endswith("x") else 1
    s = -1.0 if direction.startswith("-") else 1.0
    v = s * float(_c(sc, a)[ax] - _c(sc, b)[ax]) - gap
    return (f"side_of({a},{b},{direction})", v >= -SLACK, v)


def near(sc, a, b, r):
    d = float(np.linalg.norm(_c(sc, a) - _c(sc, b)))
    return (f"near({a},{b})", d <= r + SLACK, d)


def apart(sc, a, b, r):
    d = float(np.linalg.norm(_c(sc, a) - _c(sc, b)))
    return (f"apart({a},{b})", d >= r - SLACK, d)


def facing(sc, n, local_axis, target=(0.0, 0.0), tol=20.0):
    vec = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}[local_axis]
    phi = heading(sc[n].rotation, vec)
    d = np.asarray(target) - _c(sc, n)
    if phi is None:
        return (f"facing({n})", False, None)
    v = abs(math.degrees((math.atan2(d[1], d[0]) - phi + math.pi) % (2 * math.pi) - math.pi))
    return (f"facing({n})", v <= tol + ANGLE_SLACK, v)


def in_region(sc, names, x0=-np.inf, x1=np.inf, y0=-np.inf, y1=np.inf, label="region"):
    worst = 0.0
    for n in names:
        lo, hi = sc[n].world_aabb()
        worst = max(worst, x0 - lo[0], hi[0] - x1, y0 - lo[1], hi[1] - y1)
    return (f"in_{label}({','.join(names) if len(names) < 4 else str(len(names)) + ' bodies'})", worst <= SLACK, worst)


def outside_region(sc, names, x0, x1, y0, y1, label="region"):
    R = rect_polygon(x0, x1, y0, y1)
    worst = max([max(polygon_overlap(R, footprint_hull(sc[n])), 0.0) for n in names] + [0.0])
    return (f"outside_{label}", worst <= SLACK, worst)


def cluster(sc, names, r):
    P = np.array([_c(sc, n) for n in names])
    v = float(np.linalg.norm(P - P.mean(0), axis=1).max())
    return (f"cluster({len(names)} bodies)", v <= r + SLACK, v)


def row(sc, names, axis="y", min_gap=0.0, align_tol=0.01, order=None):
    """The bodies on one line parallel to `axis` (centres within align_tol of their mean), footprints
    at least min_gap apart; with `order` ('+y' / '-y' ...) they must also come in the order given."""
    ax = 0 if axis == "x" else 1
    P = np.array([_c(sc, n) for n in names])
    spread = float(np.abs(P[:, 1 - ax] - P[:, 1 - ax].mean()).max())
    out = [(f"row_aligned({len(names)} bodies)", spread <= align_tol + SLACK, spread)]
    seq = list(names)
    if order is None:
        seq = [names[i] for i in np.argsort(P[:, ax])]
        direction = "+" + axis
    else:
        direction = order
    gaps = [footprint_gap(sc, seq[i], seq[i + 1], direction)[2] for i in range(len(seq) - 1)]
    out.append((f"row_gaps({len(names)} bodies)", min(gaps) >= min_gap - SLACK, min(gaps)))
    return out


def evenly(sc, names, axis="y", tol=0.01):
    ax = 0 if axis == "x" else 1
    P = sorted(float(_c(sc, n)[ax]) for n in names)
    d = np.diff(P)
    v = float(np.abs(d - d.mean()).max()) if len(d) else 0.0
    return (f"evenly_spaced({len(names)} bodies)", v <= tol + SLACK, v)


def rows_parallel_to(sc, names, axis="y", per_row=3, align_tol=0.01, join=0.03):
    """The bodies form lines parallel to `axis` of at most per_row bodies each: centres grouped by
    their other coordinate (a body joins a line within `join` of it), each line aligned within align_tol."""
    ax = 0 if axis == "x" else 1
    other = sorted((float(_c(sc, n)[1 - ax]), n) for n in names)
    lines, cur = [], [other[0]]
    for v, n in other[1:]:
        if v - cur[-1][0] <= join:
            cur.append((v, n))
        else:
            lines.append(cur); cur = [(v, n)]
    lines.append(cur)
    worst_align = max(max(abs(v - np.mean([w for w, _ in ln])) for v, _ in ln) for ln in lines)
    worst_count = max(len(ln) for ln in lines)
    return [(f"rows({len(lines)} lines)", worst_count <= per_row, worst_count),
            (f"rows_aligned({len(lines)} lines)", worst_align <= align_tol + SLACK, worst_align)]


def stacked(sc, upper, lower, tol=0.01):
    b, s = sc[upper], sc[lower]
    h = support_height_for(sc, b, s, require_hit=True)
    gap = None if h is None else float(b.world_aabb()[0][2] - h)
    on = h is not None and abs(gap) <= 0.004
    v = float(np.abs(_c(sc, upper) - _c(sc, lower)).max())
    return [(f"on({upper},{lower})", on, gap), (f"centred({upper},{lower})", v <= tol + SLACK, v)]


def moved(sc, start, names, d):
    worst = max(float(np.linalg.norm(_c(sc, n) - _c(start, n))) for n in names)
    return ("max_moved", worst <= d + SLACK, worst)


def turned(sc, start, names, deg):
    worst = 0.0
    for n in names:
        a, b = heading(sc[n].rotation), heading(start[n].rotation)
        if a is not None and b is not None:
            worst = max(worst, abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi)))
    return ("max_turned", worst <= deg + ANGLE_SLACK, worst)


def yaw_square(sc, names, tol=3.0):
    worst = max(math.degrees(_angle_mod(heading(sc[n].rotation) or 0.0, math.pi / 2)) for n in names)
    return ("square_to_axes", worst <= tol + ANGLE_SLACK, worst)


def min_clearance(sc, names, gap):
    """Smallest mesh distance between any two of the bodies."""
    bodies = [sc[n] for n in names]
    pairs = pair_signed_distances(bodies, prefilter=gap + 0.01, skip_fixed_pairs=False)
    worst = min([float(s) for _, _, s, *_ in pairs] + [np.inf])
    return ("min_clearance", worst >= gap - SLACK * 0.4, worst)


def corridor_clear(sc, n, width, length=None, base=(0.0, 0.0), skip=()):
    """No body other than n (and the surfaces) intrudes into the strip of that width from n's
    footprint toward the base point (to the base when no length is given)."""
    ca = _c(sc, n)
    d = np.asarray(base) - ca
    dist = float(np.linalg.norm(d))
    u = d / dist
    v = np.array([-u[1], u[0]])
    _, amax = projection(sc[n], 1.0, u)
    end = dist if length is None else amax + length
    poly = np.array([ca + u * s + v * t for s, t in ((amax, -width / 2), (end, -width / 2), (end, width / 2), (amax, width / 2))])
    worst = 0.0
    for b in sc.bodies:
        if b.name == n or b.name in skip or (b.fixed and {"support", "ground"} & b.tags):
            continue
        worst = max(worst, polygon_overlap(poly, footprint_hull(b)))
    return (f"corridor_clear({n})", worst <= SLACK * 0.4, worst)


def generic(sc, support="table", upright_deg=5.0):
    """What every task asks: nothing interpenetrates, every free body rests on the support (or on
    a body that rests on it), upright, and over the support's top."""
    rep = verify_scene(sc)
    out = [("no_penetration", rep.pen_pairs == 0, rep.pen_pairs), ("supported", not rep.floating, len(rep.floating))]
    from simready.repair.upright_s4r import decompose_zyx
    tilt = max([math.degrees(max(abs(r), abs(p))) for b in sc.free() for _, r, p in [decompose_zyx(b.rotation)]] + [0.0])
    out.append(("upright", tilt <= upright_deg, tilt))
    lo, hi = sc[support].world_aabb()
    out.append(in_region(sc, [b.name for b in sc.free()], lo[0], hi[0], lo[1], hi[1], label="support_top"))
    return out
