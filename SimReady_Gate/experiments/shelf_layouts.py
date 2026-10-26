"""The layouts of the shelf examples (docs/examples/shelf_*): SimReady fixtures with levels, and the objects left on
them as a generator leaves them, each at a random place and heading on the level it starts on (with a fixed seed), or,
for `shelf_fix`, in rows of a few of a kind, each row a few millimetres too tight. The packages must have been fetched
(simready.cli assets fetch).

    python experiments/shelf_layouts.py docs/examples
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simready.io.nvidia_assets import package_dir, body_from_package, RECORD  # noqa: E402
from simready.scene.model import Scene, support_levels  # noqa: E402

NV, PAL = "nvidia/simready-assets", "PalatialSim/palatial-nv-asset-library"
BOOK = {"up": "+y", "front": "-x"}         # the Palatial books lie on their back cover, height along y, spine at -x

EXAMPLES = {
    # a bookcase open toward the robot; the books standing in a heap on its two lower shelves, the notebooks lying on
    # the bottom and the top shelf
    "shelf_books": {
        "seed": 3,
        "fixtures": [("case", NV, "cabinet_file_d02", (0.65, 0.0, 0.0), 180.0)],
        "objects": [("book_1", PAL, "D00045", BOOK, 1), ("book_2", PAL, "D00188", BOOK, 2), ("book_3", PAL, "D00498", BOOK, 1),
                    ("book_4", PAL, "D00552", BOOK, 2), ("book_5", PAL, "D00553", BOOK, 1), ("book_6", PAL, "D00662", BOOK, 2),
                    ("book_7", PAL, "D00728", BOOK, 1), ("notebook_1", NV, "misc_notebook_a01", {}, 1),
                    ("notebook_2", NV, "misc_notebook_a02", {}, 3), ("notebook_3", NV, "misc_notebook_a03", {}, 1),
                    ("notebook_4", NV, "misc_notebook_a04", {}, 3)]},
    # a four-tier rack, its long side toward the robot; a stock of kitchen things, a few of each kind, heaped along its
    # two lower shelves
    "shelf_pantry": {
        "seed": 5,
        "span": 0.6,
        "fixtures": [("rack", PAL, "D00661", (0.80, 0.0, 0.0), 90.0)],
        "objects": [(f"{name}_{i + 1}", NV, kinds[i % len(kinds)], {}, 1 + i % 2) for name, kinds, n in (
            ("can", ["food_beverage_can_b01", "food_beverage_can_d01", "food_beverage_can_h01", "food_beverage_can_k01",
                     "food_beverage_can_l01", "food_beverage_can_n01"], 18),
            ("soup", ["food_canned_soup_a02"], 3),
            ("bowl", ["kitchenware_bowl_b01", "kitchenware_bowl_q01"], 4),
            ("mug", ["dishware_mug_coffee_a01", "dishware_mug_coffee_b01", "dishware_cup_tea_a01"], 6),
            ("jar", ["paint_container_jar_green_a01", "paint_container_jar_orange_a01", "paint_container_jar_pink_a01",
                     "paint_container_jar_violet_a01"], 8),
            ("bottle", ["food_beverage_bottledWater_l01", "misc_bottle_corked_a01"], 6)) for i in range(n)]},
}
# shelf_fix: the same stock on the rack in rows of a few of a kind, each row too tight (each gap minus 3 mm) and a little
# askew: (level, depth of the row behind the level's front edge, the groups along the row, 4 cm apart)
FIX_ROWS = [(1, 0.04, [["mug_1", "mug_2", "mug_3"], ["mug_4", "mug_5", "mug_6"], ["soup_1", "soup_2", "soup_3"]]),
            (1, 0.22, [["bowl_1", "bowl_2"], ["bowl_3", "bowl_4"]]),
            (2, 0.04, [[f"can_{i}" for i in range(1, 7)], [f"can_{i}" for i in range(7, 13)], [f"can_{i}" for i in range(13, 19)]]),
            (3, 0.04, [["jar_1", "jar_2", "jar_3", "jar_4"], ["jar_5", "jar_6", "jar_7", "jar_8"]]),
            (3, 0.22, [["bottle_1", "bottle_2", "bottle_3"], ["bottle_4", "bottle_5", "bottle_6"]])]
GROUP_GAP = 0.04


def _body(name, repo, package, opts, yaw):
    pkg = package_dir(package, repo)
    if not (pkg / RECORD).exists():
        raise FileNotFoundError(f"{package} is not fetched: python -m simready.cli assets fetch {package} --repo {repo}")
    return body_from_package(name, pkg, (0.0, 0.0, 0.0), yaw_deg=yaw, up=opts.get("up"))


def _entry(name, repo, package, opts, x, y, level, yaw):
    """The layout entry that puts the body's footprint centre at (x, y) and its bottom on the level, turned by yaw."""
    b = _body(name, repo, package, opts, yaw)
    lo, hi = b.world_aabb()
    c = 0.5 * (lo + hi)
    e = {"name": name, "package": package, "repo": repo, "x": round(float(x - c[0]), 5), "y": round(float(y - c[1]), 5),
         "z": round(float(level.z - lo[2]), 5), "yaw": round(float(yaw), 3)}
    e.update(opts)
    return e, hi - lo


def fixtures_scene(spec):
    out, bodies = [], []
    for name, repo, package, pos, yaw in spec["fixtures"]:
        b = body_from_package(name, package_dir(package, repo), pos, yaw_deg=yaw, fixed=True, tags={"support", "fixture"})
        bodies.append(b)
        out.append({"name": name, "package": package, "repo": repo, "position": list(pos), "yaw": yaw})
    return out, Scene(bodies)


def heap(spec):
    """Each object at a uniform place and heading on its starting level, its footprint inside the level's rectangle."""
    rng = np.random.default_rng(spec["seed"])
    fx, sc = fixtures_scene(spec)
    levels = support_levels(sc, sc.bodies[0])
    objects = []
    for name, repo, package, opts, k in spec["objects"]:
        lv = levels[k - 1]
        yaw = float(rng.uniform(-40.0, 40.0))
        _, ext = _entry(name, repo, package, opts, 0.0, 0.0, lv, yaw)
        x = rng.uniform(lv.x0 + 0.5 * ext[0], lv.x1 - 0.5 * ext[0])
        span = spec.get("span", 0.25)                   # the part of a level the heap covers (its middle, in reach)
        y = rng.uniform(max(lv.y0, -span) + 0.5 * ext[1], min(lv.y1, span) - 0.5 * ext[1])
        objects.append(_entry(name, repo, package, opts, x, y, lv, yaw)[0])
    return {"fixtures": fx, "objects": objects}


def tight_rows(spec, rows, squeeze=0.003, skew=6.0, seed=7):
    """The objects in rows along y, each row centred on y = 0 at its depth behind the level's front edge, made of groups
    GROUP_GAP apart; inside a group each gap is `squeeze` short of touching and each heading off by up to `skew` degrees:
    the overlaps a generator that does not check its contacts leaves."""
    rng = np.random.default_rng(seed)
    fx, sc = fixtures_scene(spec)
    levels = support_levels(sc, sc.bodies[0])
    by_name = {o[0]: o for o in spec["objects"]}
    objects = []
    for k, depth, groups in rows:
        lv = levels[k - 1]
        sizes = [[_entry(*by_name[n][:4], 0.0, 0.0, lv, 0.0)[1] for n in g] for g in groups]
        width = sum(sum(s[1] for s in g) - squeeze * (len(g) - 1) for g in sizes) + GROUP_GAP * (len(groups) - 1)
        y = -0.5 * width
        for g, gs in zip(groups, sizes):
            for n, s in zip(g, gs):
                name, repo, package, opts, _ = by_name[n]
                objects.append(_entry(name, repo, package, opts, lv.x0 + depth + 0.5 * s[0], y + 0.5 * s[1], lv,
                                      float(rng.uniform(-skew, skew)))[0])
                y += s[1] - squeeze
            y += squeeze + GROUP_GAP
    return {"fixtures": fx, "objects": objects}


def main(out_dir: str) -> None:
    root = Path(out_dir)
    layouts = {name: heap(spec) for name, spec in EXAMPLES.items()}
    layouts["shelf_fix"] = tight_rows(EXAMPLES["shelf_pantry"], FIX_ROWS)
    for name, layout in layouts.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "layout.json").write_text(json.dumps(layout, indent=1) + "\n")
        print(f"{name}: {len(layout['objects'])} objects -> {root / name / 'layout.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
