"""Write simready/data/asset_rest_orientations.json from the RoboLab catalog and shipped scenes.

    ROBOLAB_DIR=/path/to/RoboLab python experiments/asset_rest_orientations.py

Rule. An asset keeps its authored frame when that pose can stand: the tipping angle
atan(half of the smaller footprint side / half height), with the mass centre taken at the
box centre, is at least TIP_DEG. When the authored pose cannot stand and the shipped scenes
that contain the asset agree on a different authored axis pointing up (the majority axis,
in at least two scenes), that axis becomes the asset's resting up and the table records the
rotation that brings it to +Z. Shipped scenes alone are not used for assets that can stand:
they are physics-settled heaps in which cans, cartons and bottles also lie knocked over.
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from simready.io.asset_rest import TABLE_PATH, rotation_to_up  # noqa: E402
from simready.io.paths import ROBOLAB_ROOT  # noqa: E402

TIP_DEG = 12.0


def tipping_deg(dims) -> float:
    x, y, z = [float(v) for v in dims]
    return math.degrees(math.atan2(0.5 * min(x, y), 0.5 * z))


def shipped_up_axes(robolab: Path, catalog: dict) -> dict:
    """{asset name: [(axis, scene), ...]} over every shipped scene."""
    from pxr import Usd, UsdGeom
    out = collections.defaultdict(list)
    for sp in sorted((robolab / "assets" / "scenes").glob("*.usda")):
        st = Usd.Stage.Open(str(sp))
        cache = UsdGeom.XformCache()
        for prim in st.Traverse():
            name = prim.GetName()
            base = name.rstrip("0123456789_")
            key = name if name in catalog else (base if base in catalog else None)
            if key is None:
                continue
            m = cache.GetLocalToWorldTransform(prim)
            M = np.array([[m[i][j] for j in range(4)] for i in range(4)])
            upc = M[:3, 2]                          # world-Z component of each authored axis
            k = int(np.argmax(np.abs(upc)))
            out[key].append((("+" if upc[k] > 0 else "-") + "xyz"[k], sp.name))
    return out


def main():
    robolab = Path(ROBOLAB_ROOT)
    catalog = {c["name"]: c for c in json.load(open(robolab / "assets" / "objects" / "object_catalog.json"))}
    seen = shipped_up_axes(robolab, catalog)
    assets = {}
    for name, c in sorted(catalog.items()):
        tip = tipping_deg(c["dims"])
        if tip >= TIP_DEG:
            continue
        votes = collections.Counter(a for a, _ in seen.get(name, []))
        if not votes:
            continue
        axis, n = votes.most_common(1)[0]
        if axis == "+z" or n < 2 or n < 2 * (sum(votes.values()) - n):
            continue
        R = rotation_to_up(axis)
        assets[name] = {"usd_path": c["usd_path"], "authored_dims": [float(v) for v in c["dims"]],
                        "tipping_deg": round(tip, 2), "up_axis": axis,
                        "shipped_scenes": sorted({s for a, s in seen[name] if a == axis}),
                        "rotation": [[round(float(v), 12) for v in row] for row in R]}
    doc = {"rule": f"authored pose kept when its tipping angle is at least {TIP_DEG:g} degrees; otherwise the "
                   "authored axis that points up in the shipped scenes (majority, at least two scenes) rests up",
           "assets": assets}
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(TABLE_PATH, "w"), indent=1)
    print("wrote", TABLE_PATH, "assets:", {k: v["up_axis"] for k, v in assets.items()})


if __name__ == "__main__":
    main()
