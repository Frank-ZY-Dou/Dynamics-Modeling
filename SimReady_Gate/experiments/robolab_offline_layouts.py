"""Reproduce RoboLab's pre-settle placement offline (no Isaac): pick catalog objects, build
predicates the way the scenegen skill does, run its SpatialSolver, and return poses with
z = dims[2]/2 + 0.002 (the skill's convention).

Table frame = the skill's own defaults: `SpatialSolver(table_bounds=(0.25, 0.85, -0.45, 0.45))`
(skills/robolab-scenegen/SKILL.md), i.e. the top of `table` (table_oak) in base_empty.usda,
whose top face spans x in [0.197, 0.897], y in [-0.5, 0.5] at z = 0.

Reproducibility: RoboLab's SpatialSolver draws from Python's `random` module, which nothing
seeds; `make_layout` seeds it from the NumPy seed so that every script sees the same layout.
Layouts are persisted under results/layouts/ and reused by every downstream experiment; asset
paths are stored with a `<ROBOLAB_DIR>/` prefix and expanded to the checkout on load.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np

ROBOLAB = Path(os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "RoboLab")))   # a checkout of NVLabs/RoboLab
sys.path.insert(0, str(ROBOLAB))
from robolab.scene_gen.llm_scene_gen.predicates import (  # noqa: E402
    ObjectState, PlaceOnBasePredicate, RelativePositionPredicate, PredicateType, Predicate)
from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver  # noqa: E402

PLACEHOLDER = "<ROBOLAB_DIR>/"
TABLE_BOUNDS = (0.25, 0.85, -0.45, 0.45)          # the skill's defaults (SKILL.md)
BASE_SCENE = ROBOLAB / "assets/scenes/base_empty.usda"
LAYOUT_DIR = Path(__file__).resolve().parent.parent / "results" / "layouts"
CATALOG = json.load(open(ROBOLAB / "assets/objects/object_catalog.json"))
BY_NAME = {c["name"]: c for c in CATALOG}


def pick_objects(rng, n, exclude_classes=("block",), max_dim=0.35):
    pool = [c for c in CATALOG if c["class"] not in exclude_classes and max(c["dims"]) <= max_dim
            and not c.get("static_body")]
    return [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]


def make_layout(rng, n_objects=6, n_relations=2, margin=0.05, random_rot=True, seed=None):
    """One pre-settle layout from RoboLab's SpatialSolver. `seed` (or the rng's first draw)
    also seeds Python's `random`, which the solver uses internally."""
    if seed is None:
        seed = int(rng.randint(0, 2 ** 31 - 1))
    random.seed(seed)
    objs = pick_objects(rng, n_objects)
    names = [o["name"] for o in objs]
    dims = {o["name"]: tuple(o["dims"]) for o in objs}
    states = {}
    for o in objs:
        st = ObjectState(name=o["name"])
        # the skill hands the solver explicit table coordinates; here they are sampled inside the bounds
        x = float(rng.uniform(TABLE_BOUNDS[0] + 0.05, TABLE_BOUNDS[1] - 0.05))
        y = float(rng.uniform(TABLE_BOUNDS[2] + 0.05, TABLE_BOUNDS[3] - 0.05))
        st.predicates.append(PlaceOnBasePredicate(o["name"], x=x, y=y,
                                                  yaw=float(rng.uniform(0, 360)) if random_rot else 0.0))
        states[o["name"]] = st
    rel_types = [PredicateType.LEFT_OF, PredicateType.RIGHT_OF, PredicateType.FRONT_OF, PredicateType.BACK_OF]
    relations = []
    for _ in range(min(n_relations, n_objects // 2)):
        a, b = rng.choice(n_objects, size=2, replace=False)
        rt = rel_types[rng.randint(4)]
        dist = float(rng.uniform(0.08, 0.2))
        states[names[a]].predicates.append(RelativePositionPredicate(names[a], names[b], rt, dist))
        relations.append((names[a], names[b], rt.value, dist))
    solver = SpatialSolver(TABLE_BOUNDS, collision_margin=margin)
    ok, msg = solver.solve(states, dims)
    layout = []
    for nm in names:
        s = states[nm]
        if s.x is None or s.y is None:
            continue
        z = dims[nm][2] / 2 + 0.002
        layout.append({"name": nm, "usd_path": PLACEHOLDER + BY_NAME[nm]["usd_path"],
                       "x": s.x, "y": s.y, "z": z, "yaw": s.yaw or 0.0, "dims": dims[nm]})
    return {"ok": ok, "msg": msg, "objects": layout, "relations": relations, "margin": margin,
            "table_bounds": TABLE_BOUNDS, "base_scene": PLACEHOLDER + "assets/scenes/base_empty.usda", "seed": seed, "n": n_objects}


def expand_paths(L: dict) -> dict:
    """The layout with `<ROBOLAB_DIR>/` replaced by the checkout path (a copy)."""
    L = json.loads(json.dumps(L))
    def fix(v):
        return str(ROBOLAB) + "/" + v[len(PLACEHOLDER):] if isinstance(v, str) and v.startswith(PLACEHOLDER) else v
    L["base_scene"] = fix(L.get("base_scene"))
    for o in L["objects"]:
        o["usd_path"] = fix(o["usd_path"])
    return L


def get_layout(n, seed, regenerate=False):
    """Persisted layout for (n, seed): generated once with RandomState(seed) and random.seed(seed)."""
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    path = LAYOUT_DIR / f"robolab_N{n}_s{seed}.json"
    if path.exists() and not regenerate:
        with open(path) as f:
            return expand_paths(json.load(f))
    L = make_layout(np.random.RandomState(seed), n_objects=n, seed=seed)
    with open(path, "w") as f:
        json.dump(L, f, indent=1)
    return expand_paths(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("n", type=int, nargs="?", default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save", default=None); ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()
    L = get_layout(a.n, a.seed, regenerate=a.regenerate)
    if a.save:
        with open(a.save, "w") as f:
            json.dump(L, f, indent=1)
        print("saved", a.save)
    print("solver:", L["ok"], L["msg"][:80])
    for o in L["objects"]:
        print(f"  {o['name']:28s} x={o['x']:.3f} y={o['y']:.3f} z={o['z']:.3f} yaw={o['yaw']:.0f} dims={tuple(round(d,3) for d in o['dims'])}")
    print("relations:", L["relations"])
