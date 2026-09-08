"""Shots for a RoboCasa counter-region video (rendered by viz/blender_render.py, no text).

A SELECTED example, disclosed as such: seeds are scanned from --seed upward for the first draw
where RoboCasa's own placement test (experiments/robocasa_sampler.py) gives up on N objects in
the region; the gate then places all N overlapping and S4R packs them under
within / on_support / upright, verified on the objects' collision geometry (MuJoCo-compiled).

  gate    = [sampler's partial placement (what it managed) -> gate's overlapping placement (tinted)
             -> S4R scale continuation -> repaired]
  settle  = MuJoCo trajectories on the real collision pieces: panel 0 overlapping, panel 1 repaired

Usage: python viz/make_robocasa_video.py --n 10 --region 0.15 --seed 0 --out results/videos/robocasa_n10
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "experiments"); sys.path.insert(0, "viz")
import simready  # noqa: E402,F401
from simready.scene import Body, Scene, MeshProxy  # noqa: E402
from simready.gates import verify_scene  # noqa: E402
from simready.gates.settle_mujoco import settle_and_measure  # noqa: E402
from simready.io.mjcf_io import body_from_mjcf_object, load_mjcf_pieces  # noqa: E402
from simready.dsl import parse_program, compile_program, check_predicates  # noqa: E402
from simready.repair import repair_upright  # noqa: E402
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh  # noqa: E402
import robocasa_layouts as RL  # noqa: E402
import robocasa_capacity as RC  # noqa: E402
from robocasa_sampler import sample_layout  # noqa: E402
from render_frames import interpolate, quat_wxyz_to_R  # noqa: E402
from mjcf_to_obj import export_object, export_box  # noqa: E402

FPS = 30
TOP = RL.COUNTER[2]


def poses(scene):
    return {b.name: (b.center.copy(), b.rotation.copy(), 1.0) for b in scene.free()}


def ser(frame):
    return {k: [np.asarray(c).tolist(), np.asarray(R).tolist(), float(s)] for k, (c, R, s) in frame.items()}


def yaw_R(t):
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--region", type=float, default=0.15); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/videos/robocasa_n10"); ap.add_argument("--repair-seconds", type=float, default=7.0)
    ap.add_argument("--hold-seconds", type=float, default=2.0); ap.add_argument("--settle-seconds", type=float, default=2.5)
    a = ap.parse_args()
    region = (a.region, a.region)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); assets_dir = out / "assets"

    seed = a.seed
    while True:
        objs = RC.pick(np.random.RandomState(1000 * seed + a.n), a.n)
        placed, ok = sample_layout(np.random.RandomState(1000 * seed + a.n + 1), [o[2] for o in objs], (-region[0], region[0]), (-region[1], region[1]),
                                   base=(0.0, 0.0, TOP), partial=True)
        if not ok:
            break
        seed += 1
    print(f"selected seed {seed}: RoboCasa's sampler placed {len(placed)}/{a.n} before giving up", flush=True)

    # gate: overlapping random placement (same draw), then S4R with the capacity study's program
    rng = np.random.RandomState(1000 * seed + a.n + 2)
    bodies = [RC.counter(region), RC.region_body(region)]
    xmls = {}
    for k, (xml, v, spec) in enumerate(objs):
        bot = float(v[:, 2].min())
        x = float(rng.uniform(-region[0], region[0])); y = float(rng.uniform(-region[1], region[1])); yaw = float(rng.uniform(-math.pi, math.pi))
        bodies.append(body_from_mjcf_object(f"o{k}", xml, (x, y, TOP - bot), quat_wxyz=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))); xmls[f"o{k}"] = xml
    sc = Scene(bodies)
    raw_bodies = copy.deepcopy(bodies)
    raw_pose = poses(sc)
    v_raw = verify_scene(sc)
    oo = [(x, y) for x, y, s in v_raw.pairs if s < 0 and not sc[x].fixed and not sc[y].fixed]
    pen_bodies = sorted({n for pr in oo for n in pr})
    prog = parse_program(RC.PROGRAM)
    spec = compile_program(prog, sc, s_min=0.05, ds_max=0.05, tail_iters=30)
    states = []

    def on_step(st):
        states.append((st.scale, {sc.bodies[k].name: (sc.bodies[k].center.copy(), sc.bodies[k].rotation.copy(), st.body_scale(k)) for k in st.free}))

    t0 = time.time()
    with MeshProxy(sc, faces=1500, slab_tops={"counter": TOP}):
        res = repair_upright(sc, spec, on_step=on_step)
    reseat_on_supports(sc, spec)
    v_rep = verify_scene(sc)
    if v_rep.pen_pairs > 0:
        polish_full_mesh(sc, spec, iters=15, on_step=on_step); reseat_on_supports(sc, spec); v_rep = verify_scene(sc)
    t_rep = time.time() - t0
    states.append((1.0, poses(sc)))
    fails = [p[0] for p in check_predicates(prog, sc) if not p[1]]
    rep_pose = poses(sc)

    pieces = {nm: [(v - sc[nm].meta["c_model"], f) for v, f in load_mjcf_pieces(xml)] for nm, xml in xmls.items()}
    settle_raw = Scene([b for b in raw_bodies if b.name != "region"]); settle_rep = Scene([b for b in sc.bodies if b.name != "region"])
    g5 = {tag: settle_and_measure(scn, seconds=a.settle_seconds, timestep=1 / 600, support_tops={"counter": TOP}, pieces=pieces, record_every=20)
          for tag, scn in (("raw", settle_raw), ("rep", settle_rep))}
    summary = {"n": a.n, "region": 2 * a.region, "seed": seed, "selected_example": True, "sampler_placed": len(placed), "pen_raw": v_raw.pen_pairs,
               "pen_raw_obj_obj": len(oo), "pen_rep": v_rep.pen_pairs, "rmsd_xy": res.rmsd, "steps": res.steps, "repair_s": t_rep, "fails": fails,
               "g5": {k: {"vmax": r.peak_speed, "dmax": r.peak_disp, "left": r.left_support} for k, r in g5.items()}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1), flush=True)

    # --- render assets: visual meshes from the MJCF, counter + region marker boxes -------------------
    assets = {}
    for k, (xml, v, spec_o) in enumerate(objs):
        nm = f"o{k}"; b = sc[nm]
        export_object(xml, assets_dir, nm, c_model=b.meta["c_model"])
        assets[nm] = {"obj": str((assets_dir / f"{nm}.obj").resolve()), "c_model": b.meta["c_model"].tolist(), "fixed": False}
    export_box(assets_dir, "counter", (2 * region[0] + 0.2, 2 * region[1] + 0.2, 0.05), (0, 0, 0), color=(0.88, 0.87, 0.84), roughness=0.55)
    export_box(assets_dir, "region_marker", (2 * region[0], 2 * region[1], 0.001), (0, 0, 0), color=(0.62, 0.66, 0.74), roughness=0.9)
    assets["counter"] = {"obj": str((assets_dir / "counter.obj").resolve()), "c_model": [0, 0, 0], "fixed": True, "pose": [[0.0, 0.0, TOP - 0.025], np.eye(3).tolist()]}
    assets["region_marker"] = {"obj": str((assets_dir / "region_marker.obj").resolve()), "c_model": [0, 0, 0], "fixed": True, "pose": [[0.0, 0.0, TOP + 0.0005], np.eye(3).tolist()]}
    cam = {"target": [0.0, 0.0, TOP + 0.04], "dist": 1.1, "elev_deg": 40.0, "azim_deg": -35.0}
    shots = {"assets": assets, "camera": cam, "camera_settle": dict(cam, dist=1.35, target=[0.0, 0.0, TOP - 0.05]), "floor_z": 0.0, "gate": [], "settle": []}

    names = [b.name for b in sc.free()]
    sampler_pose = dict(rep_pose); shown = []
    for k, (x, y, z, yaw) in enumerate(placed):
        nm = f"o{k}"; shown.append(nm); R = yaw_R(yaw)
        sampler_pose[nm] = (np.asarray([x, y, z]) + R @ sc[nm].meta["c_model"], R, 1.0)
    hidden = [nm for nm in names if nm not in shown]
    for _ in range(int(2.5 * FPS)):
        shots["gate"].append({"poses": ser(sampler_pose), "red": [], "hidden": hidden})
    for _ in range(int(a.hold_seconds * FPS)):
        shots["gate"].append({"poses": ser(raw_pose), "red": pen_bodies, "hidden": []})
    n_states = len(states); total = int(a.repair_seconds * FPS)
    for f in range(total):
        u = f / max(total - 1, 1) * (n_states - 1); i0 = int(np.floor(u)); i1 = min(i0 + 1, n_states - 1); t = u - i0
        shots["gate"].append({"poses": ser(interpolate(states[i0][1], states[i1][1], t)), "red": [], "hidden": []})
    for _ in range(int(2.5 * FPS)):
        shots["gate"].append({"poses": ser(rep_pose), "red": [], "hidden": []})
    n_frames = min(len(g5["raw"].trajectory), len(g5["rep"].trajectory))
    for k, tag in enumerate(("raw", "rep")):
        for f in range(n_frames):
            t, pose = g5[tag].trajectory[f]
            shots["settle"].append({"panel": k, "frame": f, "poses": ser({nm: (np.asarray(p[0]), quat_wxyz_to_R(p[1]), 1.0) for nm, p in pose.items()}),
                                    "red": list(g5[tag].left_support) if f > n_frames // 2 else [], "hidden": []})
    json.dump(shots, open(out / "shots.json", "w"))
    print("wrote", out / "shots.json", "gate frames", len(shots["gate"]), "settle shots", len(shots["settle"]), flush=True)


if __name__ == "__main__":
    main()
