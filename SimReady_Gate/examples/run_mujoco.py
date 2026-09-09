"""Simulate an exported scene in MuJoCo and report how far every free body moved.

    python examples/run_mujoco.py <export_dir> [--seconds 2.0]

Loads `<export_dir>/scene.xml` as written by `python -m simready.cli export`, steps the
simulation and prints one JSON object: peak speed, peak and final displacement per free body,
the bodies that fell below the ground height, and MuJoCo's warning counters (a reset after a
bad state would show up there).
"""
import argparse
import json
import os

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("--seconds", type=float, default=2.0)
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
    for _ in range(steps):
        mujoco.mj_step(model, data)
        for n, i in ids.items():
            peak_v = max(peak_v, float(np.linalg.norm(data.cvel[i][3:])))
            peak_d[n] = max(peak_d[n], float(np.linalg.norm(data.xpos[i] - p0[n])))
    final = {n: round(float(np.linalg.norm(data.xpos[i] - p0[n])), 4) for n, i in ids.items()}
    warnings = {mujoco.mjtWarning(k).name: int(data.warning[k].number) for k in range(len(data.warning)) if data.warning[k].number}
    print(json.dumps({"engine": "mujoco", "version": mujoco.__version__, "steps": steps, "sim_seconds": steps * model.opt.timestep,
                      "peak_speed": round(peak_v, 3), "peak_disp": round(max(peak_d.values(), default=0.0), 4),
                      "final_disp": final, "below_ground": [n for n, i in ids.items() if data.xpos[i][2] < manifest["ground_z"]],
                      "engine_warnings": warnings}, indent=1))


if __name__ == "__main__":
    main()
