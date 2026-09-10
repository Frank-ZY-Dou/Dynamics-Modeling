"""Simulate an exported scene in Genesis and report how far every free body moved.

    python examples/run_genesis.py <export_dir> [--seconds 2.0] [--cpu]

Reads `<export_dir>/manifest.json` as written by `python -m simready.cli export`, adds every
body as a mesh entity (fixed bodies as static geometry, the rest as rigid bodies that Genesis
convexifies or decomposes), steps the simulation and prints one JSON object with the peak and
final displacement per free body and the bodies that ended below the ground height.
`--record poses.json` also writes the world pose of every free body at `--record-fps`.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--cpu", action="store_true", help="Genesis CPU backend")
    ap.add_argument("--convex", action="store_true", help="one convex hull per free body instead of Genesis's convex decomposition")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--solver", choices=("cg", "newton"), default="cg", help="Genesis constraint solver")
    ap.add_argument("--record", default="", help="write the world pose of every free body at --record-fps to this JSON file")
    ap.add_argument("--record-fps", type=float, default=30.0)
    a = ap.parse_args()
    import genesis as gs
    manifest = json.load(open(os.path.join(a.export_dir, "manifest.json")))
    gs.init(backend=gs.cpu if a.cpu else gs.gpu, logging_level="warning")
    dt = float(manifest["timestep"]) * a.substeps
    solver = gs.constraint_solver.CG if a.solver == "cg" else gs.constraint_solver.Newton
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=a.substeps, gravity=(0.0, 0.0, -9.81)),
                     rigid_options=gs.options.RigidOptions(constraint_solver=solver), show_viewer=False)
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, float(manifest["ground_z"]))))
    entities = {}
    for b in manifest["bodies"]:
        if b.get("flat"):
            continue                       # the plane above stands in for a flat sheet
        extra = {"decompose_object_error_threshold": 1e9} if a.convex else {}
        morph = gs.morphs.Mesh(file=os.path.join(a.export_dir, b["mesh"]), pos=tuple(b["position"]), quat=tuple(b["quaternion_wxyz"]),
                               fixed=bool(b["fixed"]), file_meshes_are_zup=True, **extra)
        entities[b["name"]] = (scene.add_entity(morph), bool(b["fixed"]))
    scene.build()
    free = [n for n, (e, fixed) in entities.items() if not fixed]

    def as_np(v):
        v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        return np.asarray(v, dtype=float).reshape(-1)

    def pos(entity):
        return as_np(entity.get_pos())[:3]

    def quat(entity):
        return as_np(entity.get_quat())[:4]          # Genesis quaternions are wxyz

    p0 = {n: pos(entities[n][0]).copy() for n in free}
    peak = {n: 0.0 for n in free}
    steps = int(round(a.seconds / dt))
    every = max(1, int(round(1.0 / (a.record_fps * dt)))) if a.record else 0
    frames = []

    def snapshot(step):
        frames.append({"t": step * dt, "poses": {n: [pos(entities[n][0]).tolist(), quat(entities[n][0]).tolist()] for n in free}})

    if every:
        snapshot(0)
    for k in range(1, steps + 1):
        scene.step()
        for n in free:
            peak[n] = max(peak[n], float(np.linalg.norm(pos(entities[n][0]) - p0[n])))
        if every and k % every == 0:
            snapshot(k)
    if a.record:
        json.dump({"engine": "genesis", "version": gs.__version__, "timestep": dt, "record_fps": a.record_fps,
                   "quaternion": "wxyz", "frames": frames}, open(a.record, "w"))
    final = {n: round(float(np.linalg.norm(pos(entities[n][0]) - p0[n])), 4) for n in free}
    below = [n for n in free if float(pos(entities[n][0])[2]) < manifest["ground_z"]]
    print(json.dumps({"engine": "genesis", "version": gs.__version__, "backend": "cpu" if a.cpu else "gpu", "solver": a.solver,
                      "collision": "convex hull" if a.convex else "convex decomposition",
                      "steps": steps, "sim_seconds": steps * dt,
                      "peak_disp": round(max(peak.values(), default=0.0), 4), "final_disp": final, "below_ground": below}, indent=1))


if __name__ == "__main__":
    main()
