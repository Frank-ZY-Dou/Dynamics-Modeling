"""Write the --pile placement of make_robolab_scene_video.py as a layout JSON the CLI can load
(base_scene + objects), so the agent loop can be run through `simready.cli` on the same scene."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "experiments"); import simready  # noqa
from robolab_offline_layouts import pick_objects, ROBOLAB, BASE_SCENE

def pile_layout(n, seed, radius=0.13, cx=0.55, cy=0.0):
    rng = np.random.RandomState(seed); objs = pick_objects(rng, n); out = []
    for o in objs:
        r = radius * math.sqrt(rng.uniform(0, 1)); th = rng.uniform(0, 2 * math.pi); yaw = float(rng.uniform(0, 360))
        out.append({"name": o["name"], "usd_path": str(ROBOLAB / o["usd_path"]), "x": cx + r * math.cos(th), "y": cy + r * math.sin(th), "z": 0.3, "yaw": yaw, "dims": o["dims"]})
    return {"base_scene": str(BASE_SCENE), "objects": out, "pile": {"n": n, "seed": seed, "radius": radius}}

if __name__ == "__main__":
    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    json.dump(pile_layout(n, seed), open(out, "w"), indent=1); print("wrote", out)
