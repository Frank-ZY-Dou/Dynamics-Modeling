"""Simulate an exported scene in MuJoCo and report how far every free body moved.

    python examples/run_mujoco.py <export_dir> [--seconds 2.0]

Loads `<export_dir>/scene.xml` as written by `python -m simready.cli export`, steps the
simulation and prints one JSON object: peak speed, peak and final displacement per free body,
the bodies that fell below the ground height, and MuJoCo's warning counters (a reset after a
bad state would show up there). `--record poses.json` also writes the world pose of every free
body at `--record-fps` frames per second (position, quaternion wxyz), for rendering.
"""
import argparse
import json
import os

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--record", default="", help="write the world pose of every free body at --record-fps to this JSON file")
    ap.add_argument("--record-fps", type=float, default=30.0)
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(a.export_dir, "manifest.json")))
    model = mujoco.MjModel.from_xml_path(os.path.join(a.export_dir, manifest["files"]["mjcf"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    free = [b for b in manifest["bodies"] if not b["fixed"]]
    ids = {b["name"]: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["prim"]) for b in free}
    p0 = {n: data.xpos[i].copy() for n, i in ids.items()}
    peak_v, peak_d = 0.0, {n: 0.0 for n in ids}
    steps = int(round(a.seconds / model.opt.timestep))
    every = max(1, int(round(1.0 / (a.record_fps * model.opt.timestep)))) if a.record else 0
    frames = []

    def snapshot(step):
        frames.append({"t": step * model.opt.timestep,
                       "poses": {n: [data.xpos[i].tolist(), data.xquat[i].tolist()] for n, i in ids.items()}})

    if every:
        snapshot(0)
    for k in range(1, steps + 1):
        mujoco.mj_step(model, data)
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_comVel(model, data)   # the state just integrated
        for n, i in ids.items():
            peak_v = max(peak_v, float(np.linalg.norm(data.cvel[i][3:])))
            peak_d[n] = max(peak_d[n], float(np.linalg.norm(data.xpos[i] - p0[n])))
        if every and k % every == 0:
            snapshot(k)
    if a.record:
        json.dump({"engine": "mujoco", "version": mujoco.__version__, "timestep": model.opt.timestep,
                   "record_fps": a.record_fps, "quaternion": "wxyz", "frames": frames}, open(a.record, "w"))
    final = {n: round(float(np.linalg.norm(data.xpos[i] - p0[n])), 4) for n, i in ids.items()}
    warnings = {mujoco.mjtWarning(k).name: int(data.warning[k].number) for k in range(len(data.warning)) if data.warning[k].number}
    print(json.dumps({"engine": "mujoco", "version": mujoco.__version__, "steps": steps, "sim_seconds": steps * model.opt.timestep,
                      "peak_speed": round(peak_v, 3), "peak_disp": round(max(peak_d.values(), default=0.0), 4),
                      "final_disp": final, "below_ground": [n for n, i in ids.items() if data.xpos[i][2] < manifest["ground_z"]],
                      "engine_warnings": warnings}, indent=1))


if __name__ == "__main__":
    main()
