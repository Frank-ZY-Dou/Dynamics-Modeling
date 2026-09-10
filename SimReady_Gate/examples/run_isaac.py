"""Simulate an exported scene in Isaac Sim (headless) and report how far every free body moved.

    python examples/run_isaac.py <export_dir> [--seconds 2.0]

Opens `<export_dir>/scene.usda` as written by `python -m simready.cli export` (UsdPhysics rigid
bodies, colliders, a physics scene and a ground slab), plays the PhysX simulation, and prints one
JSON object with the peak and final displacement per free body and the bodies that ended below
the ground height. Run it with Isaac Sim's own Python (the `isaacsim` package). `--record
poses.json` also writes the world pose of every free body at `--record-fps`.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--record", default="", help="write the world pose of every free body at --record-fps to this JSON file")
    ap.add_argument("--record-fps", type=float, default=30.0)
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

        def pose(prim):
            cache.Clear()
            m = cache.GetLocalToWorldTransform(prim)
            q = m.RemoveScaleShear().ExtractRotationQuat()
            im = q.GetImaginary()
            return [float(m[3][0]), float(m[3][1]), float(m[3][2])], [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]

        p0 = {n: pos(p) for n, p in prims.items()}
        peak = {n: 0.0 for n in prims}
        dt = float(manifest["timestep"])
        steps = int(round(a.seconds / dt))
        every = max(1, int(round(1.0 / (a.record_fps * dt)))) if a.record else 0
        frames = []

        def snapshot(step):
            frames.append({"t": step * dt, "poses": {n: list(pose(p)) for n, p in prims.items()}})

        if every:
            snapshot(0)
        for k in range(1, steps + 1):
            world.step(render=False)
            for n, p in prims.items():
                peak[n] = max(peak[n], float(np.linalg.norm(pos(p) - p0[n])))
            if every and k % every == 0:
                snapshot(k)
        if a.record:
            json.dump({"engine": "isaac_sim", "timestep": dt, "record_fps": a.record_fps, "quaternion": "wxyz",
                       "frames": frames}, open(a.record, "w"))
        final = {n: round(float(np.linalg.norm(pos(p) - p0[n])), 4) for n, p in prims.items()}
        below = [n for n, p in prims.items() if pos(p)[2] < manifest["ground_z"]]
        print(json.dumps({"engine": "isaac_sim", "steps": steps, "sim_seconds": steps * float(manifest["timestep"]),
                          "peak_disp": round(max(peak.values(), default=0.0), 4), "final_disp": final, "below_ground": below}, indent=1), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
