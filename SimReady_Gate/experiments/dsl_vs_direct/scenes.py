"""Layouts the tasks start from: the support (the prep table turned to span y in front of the robot,
or the pallet) and the objects in their starting arrangement, written as layout JSONs that
`python -m simready.cli` reads (objects name SimReady packages).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from simready.io.nvidia_assets import body_from_package, package_dir
from simready.scene.model import Scene
from simready.gates.verify import pair_signed_distances

from tasks import ASSETS

TOP_Z = 0.0


def support_entry(kind: str) -> dict:
    """The table turned by 90 degrees (its 1.80 m side along y), near edge at x = 0.20, top at z = 0;
    or the pallet, near edge at x = 0.30, top at z = 0."""
    pkg, yaw, near, name = (ASSETS["table"], 90.0, 0.20, "table") if kind == "table" else (ASSETS["pallet"], 0.0, 0.30, "pallet")
    b = body_from_package(name, package_dir(pkg), (0.0, 0.0, 0.0), yaw_deg=yaw, fixed=True, tags={"support", "fixture"})
    lo, hi = b.world_aabb()
    top = Scene([b]).support_height(b, at=0.5 * (lo + hi)[:2])
    return {"name": name, "package": pkg, "position": [near - float(lo[0]), -0.5 * float(lo[1] + hi[1]), TOP_Z - top], "yaw": yaw}


def object_entry(name: str, pkg: str, xy, yaw: float) -> dict:
    """A layout entry whose body has its reference centre at xy and its lowest point on the top."""
    b = body_from_package(name, package_dir(pkg), (0.0, 0.0, 0.0), yaw_deg=yaw)
    lo, _ = b.world_aabb()
    return {"name": name, "package": pkg, "x": float(xy[0] - b.center[0]), "y": float(xy[1] - b.center[1]),
            "z": float(TOP_Z - lo[2]), "yaw": float(yaw)}


def _footprint_box(name, pkg, yaw):
    b = body_from_package(name, package_dir(pkg), (0.0, 0.0, 0.0), yaw_deg=yaw)
    lo, hi = b.world_aabb()
    return 0.5 * (hi[:2] - lo[:2])


def heap(objects, center, radius, rng) -> list:
    """Every body at a random point of a disc and a random heading: the bodies overlap."""
    out = []
    for name, pkg in objects:
        r, t = radius * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        out.append(object_entry(name, pkg, (center[0] + r * math.cos(t), center[1] + r * math.sin(t)), float(rng.uniform(0, 360))))
    return out


def scatter(objects, region, rng, clearance=0.03, tries=4000) -> list:
    """Bodies at random poses in a region with at least `clearance` between their footprint boxes."""
    x0, x1, y0, y1 = region
    placed, out = [], []
    for name, pkg in objects:
        for _ in range(tries):
            yaw = float(rng.uniform(0, 360))
            h = _footprint_box(name, pkg, yaw)
            c = np.array([rng.uniform(x0 + h[0], x1 - h[0]), rng.uniform(y0 + h[1], y1 - h[1])])
            if all(np.any(np.abs(c - pc) > h + ph + clearance) for pc, ph in placed):
                placed.append((c, h))
                out.append(object_entry(name, pkg, c, yaw))
                break
        else:
            raise RuntimeError(f"no room for {name} in {region}")
    return out


def nearly(objects, region, rng, overlaps=4, depth=0.012) -> list:
    """A valid scatter in which `overlaps` bodies were pushed `depth` into their nearest neighbour."""
    entries = scatter(objects, region, rng, clearance=0.04)
    sc = Scene([body_from_package(e["name"], package_dir(e["package"]), (e["x"], e["y"], e["z"]), yaw_deg=e["yaw"]) for e in entries])
    moved, used = 0, set()
    order = rng.permutation(len(entries))
    for i in order:
        if moved >= overlaps or i in used:
            continue
        bi = sc.bodies[i]
        cand = [(float(np.linalg.norm(bj.center[:2] - bi.center[:2])), j) for j, bj in enumerate(sc.bodies) if j != i and j not in used]
        if not cand:
            continue
        _, j = min(cand)
        pairs = pair_signed_distances([bi, sc.bodies[j]], prefilter=1.0)
        gap = float(pairs[0][2]) if pairs else 0.05
        u = sc.bodies[j].center[:2] - bi.center[:2]
        u = u / max(np.linalg.norm(u), 1e-9)
        entries[i]["x"] += float(u[0] * (gap + depth)); entries[i]["y"] += float(u[1] * (gap + depth))
        bi.center = bi.center + np.array([u[0], u[1], 0.0]) * (gap + depth)
        used.update({i, j}); moved += 1
    return entries


def build(task, seed: int, out_dir: Path) -> Path:
    """Write the task's starting layout for a seed; returns its path."""
    rng = np.random.default_rng(1000 + seed)
    st = task.start
    if st["kind"] == "heap":
        objs = heap(task.objects, st["center"], st["radius"], rng)
    elif st["kind"] == "scatter":
        objs = scatter(task.objects, st["region"], rng)
    else:
        objs = nearly(task.objects, st["region"], rng, st.get("overlaps", 4), st.get("depth", 0.012))
    layout = {"task": task.key, "seed": seed, "frame": "robot base at the origin looking along +x; +y left; metres, degrees",
              "assets": {"repo": ASSETS["repo"], "revision": ASSETS["revision"]},
              "table": support_entry(task.support), "objects": objs}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task.key}_s{seed}.json"
    tmp = path.with_name(f".{path.name}.{os.getpid()}")       # the conditions of a seed build it at once: write aside, then rename
    tmp.write_text(json.dumps(layout, indent=1))
    os.replace(tmp, path)
    return path
