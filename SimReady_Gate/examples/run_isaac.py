"""Simulate an exported scene in Isaac Sim (headless) and report how far every free body moved.

    python examples/run_isaac.py <export_dir> [--seconds 2.0]

Opens `<export_dir>/scene.usda` as written by `python -m simready.cli export` (UsdPhysics rigid
bodies, colliders, a physics scene and a ground slab), plays the PhysX simulation, and prints one
JSON object with the peak and final displacement per free body and the bodies that ended below
the ground height. Run it with Isaac Sim's own Python (the `isaacsim` package).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("--seconds", type=float, default=2.0)
    a = ap.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        from isaacsim.core.api import World
        manifest = json.load(open(os.path.join(a.export_dir, "manifest.json")))
        usd_path = os.path.abspath(os.path.join(a.export_dir, manifest["files"]["usd"]))
        omni.usd.get_context().open_stage(usd_path)
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0, physics_dt=float(manifest["timestep"]), rendering_dt=float(manifest["timestep"]))
        world.reset()
        free = [b for b in manifest["bodies"] if not b["fixed"]]
        prims = {b["name"]: stage.GetPrimAtPath(f"/World/{b['prim']}") for b in free}
        cache = UsdGeom.XformCache()

        def pos(prim):
            cache.Clear()
            m = cache.GetLocalToWorldTransform(prim)
            return np.array([m[3][0], m[3][1], m[3][2]], dtype=float)

        p0 = {n: pos(p) for n, p in prims.items()}
        peak = {n: 0.0 for n in prims}
        steps = int(round(a.seconds / float(manifest["timestep"])))
        for _ in range(steps):
            world.step(render=False)
            for n, p in prims.items():
                peak[n] = max(peak[n], float(np.linalg.norm(pos(p) - p0[n])))
        final = {n: round(float(np.linalg.norm(pos(p) - p0[n])), 4) for n, p in prims.items()}
        below = [n for n, p in prims.items() if pos(p)[2] < manifest["ground_z"]]
        print(json.dumps({"engine": "isaac_sim", "steps": steps, "sim_seconds": steps * float(manifest["timestep"]),
                          "peak_disp": round(max(peak.values(), default=0.0), 4), "final_disp": final, "below_ground": below}, indent=1), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
