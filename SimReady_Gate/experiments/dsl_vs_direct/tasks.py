"""The tasks of the DSL-versus-direct comparison: SimReady scenes, the request a model is given,
and the rubric the result is graded with.

Every scene is a layout of NVIDIA SimReady packages (assets.json, pinned to one dataset commit) on
the stainless prep table, turned so that its 0.70 x 1.80 m top spans x in [0.20, 0.90] and
y in [-0.90, 0.90] in front of the robot base at the origin, top at z = 0 (or, for palletizing, on
a 0.80 x 1.20 m pallet). A request states every number its rubric checks, in the frame both
conditions are told: +x away from the robot, +y to its left. `reference` is a program written for
the task to show that the rubric can be met; no model sees it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import rubric as R

HERE = Path(__file__).resolve().parent
ASSETS = json.loads((HERE / "assets.json").read_text())


@dataclass
class Task:
    name: str
    n: int
    objects: list                       # [(body name, package)]
    start: dict                         # how the scene starts: a heap, a scatter, a nearly valid scatter
    request: str
    checks: object                      # callable(final scene, start scene) -> [(name, ok, value)]
    reference: object                   # a program (JSON) that meets the rubric, or callable(scene) -> program
    support: str = "table"
    tags: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.name}_n{self.n}"


def pkg(cat: str, i: int = 0) -> str:
    pool = ASSETS["pool"][cat]
    return pool[i % len(pool)]


def group(prefix: str, cats: list, count: int):
    """count bodies named prefix_1.. drawn in turn from the categories' packages."""
    out = []
    k = 0
    while len(out) < count:
        cat = cats[k % len(cats)]
        out.append((f"{prefix}_{len(out) + 1}", pkg(cat, k // len(cats))))
        k += 1
    return out


def _st(op, **kw):
    return {"op": op, **kw}


BASE = [_st("no_penetration", margin=0.005), _st("on_support", a="*", b="table"), _st("upright", a="*"),
        _st("within", a="*", b="table.top", inset=0.02)]


# ----------------------------------------------------------------------------------------------
def setting():
    objs = [("plate", pkg("plate")), ("fork", pkg("fork")), ("knife", pkg("knife")), ("spoon", pkg("spoon")),
            ("mug", pkg("mug")), ("glass", pkg("glass"))]
    request = ("Set a single place on the table. The plate's centre on y = 0 (within 1 cm), between 50 and 60 cm from the robot base. "
               "The fork on the left of the plate and the knife on its right, mirror images of each other about the plate (within 1 cm), "
               "both with their long sides parallel to the x axis (within 5 degrees) and with 2 to 5 cm of table between each of them "
               "and the plate. The spoon on the right of the knife, parallel to it, with at least 2 cm of table between them. The glass "
               "farther from the robot than the plate, its centre within 15 cm of the knife's centre. The mug on the left of the fork "
               "with at least 3 cm of table between them, its handle pointing at the robot base (within 20 degrees). Nothing "
               "interpenetrates; everything rests upright on the table.")

    def checks(sc, st):
        out = [R.centre_on(sc, "plate", y=0.0, tol=0.01), R.dist_to_base(sc, "plate", 0.50, 0.60),
               R.mirror(sc, "fork", "knife", "plate", "y", 0.01), R.long_axis_along(sc, "fork", "x", 5), R.long_axis_along(sc, "knife", "x", 5),
               R.side_of(sc, "fork", "plate", "+y"), R.side_of(sc, "knife", "plate", "-y"),
               R.footprint_gap(sc, "plate", "fork", "+y", 0.02, 0.05), R.footprint_gap(sc, "plate", "knife", "-y", 0.02, 0.05),
               R.side_of(sc, "spoon", "knife", "-y"), R.parallel_long(sc, "spoon", "knife", 5), R.footprint_gap(sc, "knife", "spoon", "-y", 0.02),
               R.side_of(sc, "glass", "plate", "+x"), R.near(sc, "glass", "knife", 0.15),
               R.side_of(sc, "mug", "fork", "+y"), R.footprint_gap(sc, "fork", "mug", "+y", 0.03), R.facing(sc, "mug", "-x", (0.0, 0.0), 20)]
        return out + R.generic(sc)
    ref = {"statements": BASE + [
        _st("at", a="plate", x=0.55, y=0.0, tol=0.005),
        _st("yaw", a="fork", deg=0, axis="long", tol=3), _st("yaw", a="knife", deg=0, axis="long", tol=3), _st("parallel", a="spoon", b="knife", tol=3),
        _st("symmetric", a="fork", b="knife", c="plate", axis="y", tol=0.005),
        _st("row", members=["spoon", "knife", "plate", "fork"], axis="+y", gap=0.022),
        _st("near", a="fork", b="plate", r=0.14), _st("near", a="knife", b="plate", r=0.14),
        _st("left_of", a="mug", b="fork", gap=0.085), _st("facing", a="mug", axis="-x", tol=10),
        _st("in_front_of", a="glass", b="plate", gap=0.12), _st("near", a="glass", b="knife", r=0.14)]}
    return Task("setting", len(objs), objs, {"kind": "heap", "center": (0.55, 0.0), "radius": 0.10}, request, checks, ref)


def drinks_row(n=6):
    if n <= 6:
        objs = group("can", ["can"], n - 1) + [("bottle_1", pkg("bottle", 0))]
    else:
        objs = group("can", ["can"], 6) + [("bottle_1", pkg("bottle", 0)), ("bottle_2", pkg("bottle", 1)),
                                           ("glass_1", pkg("glass", 0)), ("glass_2", pkg("glass", 1))]
    request = ("Line up all the drinks in a single row parallel to the y axis: their centres within 1 cm of a common line parallel to the "
               "y axis, that line between x = 0.70 and x = 0.80; ordered by height from the shortest at the left end to the tallest at the "
               "right end; at least 2 cm of table between neighbours; evenly spaced (centre-to-centre distances equal within 1 cm); the "
               "midpoint between the two end drinks' centres on y = 0 (within 2 cm). Nothing interpenetrates; everything rests upright on the table.")
    names = [o[0] for o in objs]

    def checks(sc, st):
        heights = {n_: float(np.ptp(sc[n_].world_vertices()[:, 2])) for n_ in names}
        seq = sorted(names, key=lambda n_: -sc[n_].center[1])                 # left (+y) to right
        hs = [heights[n_] for n_ in seq]
        ordered = all(hs[i] <= hs[i + 1] + 0.005 for i in range(len(hs) - 1))
        ys = [float(sc[n_].center[1]) for n_ in seq]
        xs = np.mean([sc[n_].center[0] for n_ in names])
        out = R.row(sc, names, axis="y", min_gap=0.02, align_tol=0.01) + [
            ("ordered_by_height", ordered, hs), ("line_x", 0.70 - R.SLACK <= xs <= 0.80 + R.SLACK, float(xs)),
            R.evenly(sc, names, "y", 0.01), ("row_middle_on_y0", abs(0.5 * (ys[0] + ys[-1])) <= 0.02 + R.SLACK, 0.5 * (ys[0] + ys[-1]))]
        return out + R.generic(sc)
    def ref(sc):
        seq = sorted(names, key=lambda n_: float(np.ptp(sc[n_].world_vertices()[:, 2])))          # shortest first, at the left
        width = [float(np.ptp(sc[n_].verts[:, :2], axis=0).max()) for n_ in seq]
        pitch = max(0.5 * (width[i] + width[i + 1]) for i in range(len(seq) - 1)) + 0.03
        ys = [0.5 * pitch * (len(seq) - 1) - pitch * i for i in range(len(seq))]
        return {"statements": BASE + [_st("row", members=seq, axis="-y", gap=0.025, tol=0.005),
                                      _st("evenly_spaced", members=seq, axis="y", tol=0.004)] + [
            _st("at", a=n_, x=0.75, y=round(y, 4), tol=0.004) for n_, y in zip(seq, ys)]}
    return Task("drinks_row", len(objs), objs, {"kind": "heap", "center": (0.55, 0.15), "radius": 0.12}, request, checks, ref)


def fruit_cans(n=10):
    k_fruit = {10: 5, 16: 8, 24: 14}[n]
    k_can = {10: 4, 16: 6, 24: 7}[n]
    fruit = group("fruit", ["apple", "orange", "banana", "tangerine"], k_fruit)
    cans = group("can", ["can"], k_can)
    objs = fruit + cans + [("bowl", pkg("bowl"))] + ([("mug", pkg("mug"))] if n >= 16 else [])
    objs += group("jar", ["jar"], n - len(objs))
    r = {10: 0.12, 16: 0.13, 24: 0.20}[n]
    mug = " and the mug's centre within 20 cm of the bowl's centre, the mug also entirely between y = -0.20 and y = 0.20" if n >= 16 else ""
    jars = " The paint jars anywhere on the table at least 3 cm from every other object." if any(o[0].startswith("jar") for o in objs) else ""
    request = (f"Sort the heap. Put all the fruit together on the left of the table: every piece entirely at y > 0.25, every piece's centre "
               f"within {r:.2f} m of the centre of the fruit group. Put all the cans in one line parallel to the x axis on the right of the "
               f"table: every can entirely at y < -0.25, the cans' centres within 1 cm of a common line parallel to the x axis, at least "
               f"2 cm of table between neighbouring cans. Put the bowl entirely between y = -0.20 and y = 0.20 with its centre at least 55 cm "
               f"from the robot base{mug}.{jars} Keep the square from x = 0.30 to 0.50, y = -0.10 to 0.10 completely empty. Nothing "
               f"interpenetrates; everything rests upright on the table.")
    fn, cn, jn = [o[0] for o in fruit], [o[0] for o in cans], [o[0] for o in objs if o[0].startswith("jar")]

    def checks(sc, st):
        out = [R.in_region(sc, fn, y0=0.25, label="left"), R.cluster(sc, fn, r), R.in_region(sc, cn, y1=-0.25, label="right")]
        out += R.row(sc, cn, axis="x", min_gap=0.02, align_tol=0.01)
        out += [R.in_region(sc, ["bowl"], y0=-0.20, y1=0.20, label="middle"), R.dist_to_base(sc, "bowl", lo=0.55)]
        if n >= 16:
            out += [R.near(sc, "mug", "bowl", 0.20), R.in_region(sc, ["mug"], y0=-0.20, y1=0.20, label="middle")]
        if jn:
            others = [b.name for b in sc.free()]
            out += [("jar_clearance", min(R.min_clearance(sc, [j, o], 0.03)[2] for j in jn for o in others if o != j) >= 0.03 - R.SLACK * 0.4,
                     None)]
        out += [R.outside_region(sc, [b.name for b in sc.free()], 0.30, 0.50, -0.10, 0.10, label="square")]
        return out + R.generic(sc)
    ref = {"statements": BASE + [
        _st("within", a=f, b="table.top", y0=0.27) for f in fn] + [_st("cluster", members=fn, r=r - 0.02)] + [
        _st("within", a=c, b="table.top", y1=-0.27) for c in cn] + [_st("row", members=cn, axis="+x", gap=0.025, tol=0.005)] + [
        _st("within", a="bowl", y0=-0.18, y1=0.18), _st("reach", a="bowl", r=0.75, min=0.57),
        _st("keep_out", a="*", x0=0.29, x1=0.51, y0=-0.11, y1=0.11)] + (
        [_st("near", a="mug", b="bowl", r=0.18), _st("within", a="mug", y0=-0.18, y1=0.18)] if n >= 16 else []) + (
        [_st("place", a="bowl", x=0.64, y=0.05), _st("place", a="mug", x=0.70, y=-0.10)] if n >= 24 else []) + (
        [_st("clearance", a=j, gap=0.035) for j in jn])}
    return Task("fruit_cans", n, objs, {"kind": "heap", "center": (0.55, 0.0), "radius": {10: 0.15, 16: 0.18, 24: 0.22}[n]}, request, checks, ref)


def pick_ready():
    objs = [("mug", pkg("mug")), ("bowl", pkg("bowl", 1)), ("apple", pkg("apple")), ("can", pkg("can")), ("bottle", pkg("bottle")),
            ("screwdriver", pkg("screwdriver")), ("tape", pkg("tape")), ("tumbler", pkg("glass", 1))]
    request = ("The robot will next pick up the mug by its handle. Put the mug with its centre 45 to 60 cm from the robot base and its handle "
               "pointing at the robot base (within 20 degrees). Keep a corridor 15 cm wide from the mug straight to the robot base free of "
               "any other object. Keep every other object's centre at least 18 cm from the mug's centre. Keep the drop zone, x from 0.25 to "
               "0.40 and y from 0.20 to 0.40, empty. Every object's centre within 80 cm of the robot base. Nothing interpenetrates; "
               "everything rests upright on the table.")
    others = [o[0] for o in objs[1:]]

    def checks(sc, st):
        out = [R.dist_to_base(sc, "mug", 0.45, 0.60), R.facing(sc, "mug", "-x", (0.0, 0.0), 20), R.corridor_clear(sc, "mug", 0.15)]
        out += [R.apart(sc, o, "mug", 0.18) for o in others]
        out += [R.outside_region(sc, [b.name for b in sc.free()], 0.25, 0.40, 0.20, 0.40, label="drop_zone")]
        out += [R.dist_to_base(sc, b.name, hi=0.80) for b in sc.free()]
        return out + R.generic(sc)
    ref = {"statements": BASE + [_st("reach", a="mug", r=0.58, min=0.47), _st("facing", a="mug", axis="-x", tol=10),
                                 _st("approach", a="mug", length=0.6, width=0.17), _st("min_distance", a="*", b="mug", r=0.19),
                                 _st("keep_out", a="*", x0=0.24, x1=0.41, y0=0.19, y1=0.41), _st("reach", a="*", r=0.78)]}
    return Task("pick_ready", len(objs), objs, {"kind": "heap", "center": (0.5, -0.05), "radius": 0.12}, request, checks, ref)


def minimal_fix():
    objs = [("plate", pkg("plate")), ("bowl", pkg("bowl")), ("mug", pkg("mug")), ("glass", pkg("glass")), ("apple_1", pkg("apple", 0)),
            ("apple_2", pkg("apple", 1)), ("orange", pkg("orange")), ("can_1", pkg("can", 0)), ("can_2", pkg("can", 1)),
            ("bottle", pkg("bottle")), ("notebook", pkg("notebook")), ("tape", pkg("tape"))]
    request = ("Some of these objects interpenetrate. Make the scene free of interpenetration while moving no object more than 3 cm from "
               "where it is now and turning no object more than 10 degrees. Everything rests upright on the table.")
    names = [o[0] for o in objs]

    def checks(sc, st):
        return [R.moved(sc, st, names, 0.03), R.turned(sc, st, names, 10.0)] + R.generic(sc)
    ref = {"statements": BASE + [_st("max_move", a="*", d=0.028), _st("max_turn", a="*", deg=8)]}
    return Task("minimal_fix", len(objs), objs, {"kind": "nearly", "region": (0.25, 0.85, -0.60, 0.60), "overlaps": 4, "depth": 0.012},
                request, checks, ref)


def stack():
    objs = [("box_large", "box_flat_a09"), ("box_medium", "box_cube_a03"), ("box_small", "box_cube_a01"),
            ("apple", pkg("apple")), ("can", pkg("can")), ("cup", pkg("cup"))]
    request = ("Stack the three cardboard boxes into one tower: the largest at the bottom, the smallest on top, each box resting on the one "
               "below it with its centre within 1 cm of the centre of that box. Put the tower on the right half of the table, the bottom box "
               "entirely at y < 0 with its centre within 70 cm of the robot base. Put every other object on the left half of the table, "
               "entirely at y > 0. Nothing interpenetrates; everything rests upright.")

    def checks(sc, st):
        out = R.stacked(sc, "box_medium", "box_large", 0.01) + R.stacked(sc, "box_small", "box_medium", 0.01)
        out += [R.in_region(sc, ["box_large"], y1=0.0, label="right_half"), R.dist_to_base(sc, "box_large", hi=0.70),
                R.in_region(sc, ["apple", "can", "cup"], y0=0.0, label="left_half")]
        return out + R.generic(sc)
    ref = {"statements": BASE + [_st("stack", members=["box_large", "box_medium", "box_small"], tol=0.005),
                                 _st("within", a="box_large", b="table.top", side="right", frac=0.5, inset=0.02),
                                 _st("reach", a="box_large", r=0.68)] + [
        _st("within", a=o, b="table.top", side="left", frac=0.5, inset=0.02) for o in ("apple", "can", "cup")]}
    return Task("stack", len(objs), objs, {"kind": "heap", "center": (0.55, 0.10), "radius": 0.15}, request, checks, ref)


def palletize(n=4):
    boxes = ["box_cardboard_d01", "box_cardboard_d02", "box_cardboard_d03", "box_printer_a08", "box_flat_a09", "box_cube_a03"][:n]
    objs = [(f"box_{i + 1}", p) for i, p in enumerate(boxes)]
    request = ("Palletize the boxes: every box entirely on the pallet at least 2 cm in from its edges; the boxes in rows parallel to the "
               "y axis with at most 3 boxes in a row, the centres of a row within 1 cm of a common line parallel to the y axis; every box's "
               "sides parallel to the pallet's edges (within 3 degrees); at least 1 cm between any two boxes. Nothing interpenetrates; "
               "every box rests upright on the pallet.")
    names = [o[0] for o in objs]

    def checks(sc, st):
        lo, hi = sc["pallet"].world_aabb()
        out = [R.in_region(sc, names, lo[0] + 0.02, hi[0] - 0.02, lo[1] + 0.02, hi[1] - 0.02, label="pallet_inset")]
        out += R.rows_parallel_to(sc, names, "y", 3, 0.01) + [R.yaw_square(sc, names, 3.0), R.min_clearance(sc, names, 0.01)]
        return out + R.generic(sc, support="pallet")
    rows = [names[:2], names[2:]] if n == 4 else [names[i:i + 3] for i in range(0, len(names), 3)]
    ref = {"statements": [_st("no_penetration", margin=0.005), _st("on_support", a="*", b="pallet"), _st("upright", a="*"),
                          _st("within", a="*", b="pallet.top", inset=0.03), _st("yaw", a="*", deg=0, mod=90, tol=2),
                          _st("clearance", a="*", gap=0.015)] + [_st("row", members=r_, axis="+y", gap=0.02, tol=0.005) for r_ in rows] + (
        [_st("in_front_of", a=rows[1][0], b=rows[0][0], gap=0.30)] if len(rows) > 1 else [])}
    return Task("palletize", n, objs, {"kind": "heap", "center": (0.70, 0.0), "radius": 0.15}, request, checks, ref, support="pallet")


def kitchen_prep():
    fruit = [("apple_1", pkg("apple", 0)), ("apple_2", pkg("apple", 1)), ("orange", pkg("orange")), ("banana", pkg("banana")),
             ("tangerine", pkg("tangerine"))]
    drinks = [("bottle", pkg("bottle")), ("can_1", pkg("can", 0)), ("can_2", pkg("can", 1))]
    objs = [("board", pkg("cutting_board")), ("knife", pkg("knife", 1)), ("spoon", pkg("spoon")), ("bowl", pkg("bowl"))] + fruit + drinks
    request = ("Lay the heap out for cooking. The cutting board with its centre on y = 0 (within 2 cm), 40 to 50 cm from the robot base, its "
               "long side parallel to the x axis (within 5 degrees). The knife on the right of the board, parallel to the board's long side "
               "(within 5 degrees), with 2 to 5 cm of table between them. The spoon on the right of the knife, parallel to it, with at "
               "least 2 cm of table between them. The bowl on the left of the board with at least 5 cm of table between them. The fruit "
               "(both apples, the orange, the banana, the tangerine) grouped with every centre within 12 cm of the group's centre, every "
               "piece farther from the robot than the bowl. The bottle and the cans in one row parallel to the y axis, each at least 75 cm "
               "from the robot base, their centres within 1 cm of a common line, at least 2 cm of table between neighbours. No object other "
               "than the knife and the spoon with its centre within 25 cm of the board's centre. Nothing interpenetrates; everything rests "
               "upright on the table.")
    fn, dn = [f[0] for f in fruit], [d[0] for d in drinks]

    def checks(sc, st):
        out = [R.centre_on(sc, "board", y=0.0, tol=0.02), R.dist_to_base(sc, "board", 0.40, 0.50), R.long_axis_along(sc, "board", "x", 5),
               R.side_of(sc, "knife", "board", "-y"), R.parallel_long(sc, "knife", "board", 5), R.footprint_gap(sc, "board", "knife", "-y", 0.02, 0.05),
               R.side_of(sc, "spoon", "knife", "-y"), R.parallel_long(sc, "spoon", "knife", 5), R.footprint_gap(sc, "knife", "spoon", "-y", 0.02),
               R.side_of(sc, "bowl", "board", "+y"), R.footprint_gap(sc, "board", "bowl", "+y", 0.05), R.cluster(sc, fn, 0.12)]
        out += [R.side_of(sc, f, "bowl", "+x") for f in fn]
        out += R.row(sc, dn, axis="y", min_gap=0.02, align_tol=0.01) + [R.dist_to_base(sc, d, lo=0.75) for d in dn]
        out += [R.apart(sc, o.name, "board", 0.25) for o in sc.free() if o.name not in ("board", "knife", "spoon")]
        return out + R.generic(sc)
    ref = {"statements": BASE + [
        _st("at", a="board", x=0.45, y=0.0, tol=0.01), _st("yaw", a="board", deg=0, axis="long", tol=3),
        _st("parallel", a="knife", b="board", tol=3), _st("parallel", a="spoon", b="knife", tol=3),
        _st("row", members=["spoon", "knife", "board"], axis="+y", gap=0.03), _st("near", a="knife", b="board", r=0.17),
        _st("left_of", a="bowl", b="board", gap=0.32), _st("cluster", members=fn, r=0.10)] + [
        _st("in_front_of", a=f, b="bowl", gap=0.12) for f in fn] + [
        _st("row", members=dn, axis="+y", gap=0.03, tol=0.005), _st("reach", a="bottle", r=0.88, min=0.79),
        _st("reach", a="can_1", r=0.88, min=0.79), _st("reach", a="can_2", r=0.88, min=0.79)] + [
        _st("min_distance", a=o[0], b="board", r=0.27) for o in objs if o[0] not in ("board", "knife", "spoon")]}
    return Task("kitchen_prep", len(objs), objs, {"kind": "heap", "center": (0.55, 0.0), "radius": 0.13}, request, checks, ref)


def all_tasks():
    return [setting(), drinks_row(6), drinks_row(10), fruit_cans(10), fruit_cans(16), fruit_cans(24), pick_ready(), minimal_fix(),
            stack(), palletize(4), palletize(6), kitchen_prep()]
