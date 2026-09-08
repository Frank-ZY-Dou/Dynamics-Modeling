"""Two videos on one RoboLab layout (franka_table, N objects from RoboLab's own SpatialSolver):

  gate.mp4    raw layout (penetrating bodies in red) -> S4R scale continuation -> repaired
              -> G5 verdict and the agent route (within inset) -> routed layout
  settle.mp4  MuJoCo settle side by side: raw | after S4R | after S4R + route (2 s, real time)

Usage: python viz/make_robolab_video.py --n 10 --seed 0 --out results/videos/n10_s0
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "experiments"); sys.path.insert(0, "viz")
import simready  # noqa: E402,F401
from simready.scene import MeshProxy  # noqa: E402
from simready.gates import verify_scene  # noqa: E402
from simready.gates.settle_mujoco import settle_and_measure  # noqa: E402
from simready.dsl import parse_program, compile_program  # noqa: E402
from simready.repair import repair_upright  # noqa: E402
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh  # noqa: E402
from robolab_offline_layouts import make_layout  # noqa: E402
from robolab_diagnostic import scene_from_layout, program_from_layout  # noqa: E402
from render_frames import FrameRenderer, PALETTE, RED, TABLE, encode, hstack, interpolate, quat_wxyz_to_R  # noqa: E402

FPS = 30
INK = (0.12, 0.12, 0.12)
GREY = (0.45, 0.45, 0.45)
GREEN = (0.10, 0.55, 0.25)
CAM = dict(target=(0.50, 0.05, 0.05), dist=1.25, elev_deg=34.0, azim_deg=-40.0)


def title(t, sub=None, sub_rgb=GREY, foot=None):
    out = [(t, (30, 24), 34, INK, True)]
    if sub:
        out.append((sub, (30, 68), 26, sub_rgb, False))
    if foot:
        out.append((foot, (30, 672), 20, GREY, False))
    return out


def poses(scene):
    return {b.name: (b.center.copy(), b.rotation.copy(), 1.0) for b in scene.free()}


def run_repair(L, program_text, tops, record):
    sc = scene_from_layout(L)
    prog = parse_program(program_text)
    spec = compile_program(prog, sc, s_min=0.05, ds_max=0.05, tail_iters=30)
    states = []

    def on_step(st):
        states.append((st.scale, {sc.bodies[k].name: (sc.bodies[k].center.copy(), sc.bodies[k].rotation.copy(), st.body_scale(k)) for k in st.free}))

    t0 = time.time()
    with MeshProxy(sc, faces=2000, slab_tops=tops):
        res = repair_upright(sc, spec, on_step=on_step if record else None)
    reseat_on_supports(sc, spec)
    if verify_scene(sc).pen_pairs > 0:
        polish_full_mesh(sc, spec, iters=15, on_step=on_step if record else None); reseat_on_supports(sc, spec)
    states.append((1.0, poses(sc)))          # final, re-seated on the full mesh
    return sc, res, states, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/videos/n10_s0"); ap.add_argument("--inset", type=float, default=0.10)
    ap.add_argument("--repair-seconds", type=float, default=7.0); ap.add_argument("--settle-seconds", type=float, default=2.5)
    ap.add_argument("--dump", action="store_true", help="write shots.json (poses, labels, highlights per frame) for an external renderer instead of rendering")
    a = ap.parse_args()
    out = Path(a.out); (out / "gate").mkdir(parents=True, exist_ok=True); (out / "settle").mkdir(parents=True, exist_ok=True)

    L = make_layout(np.random.RandomState(a.seed), n_objects=a.n)
    raw = scene_from_layout(L)
    tops = {"table": min(raw.support_height(raw["table"], at=b.center[:2], radius=0.1) for b in raw.free())}
    text = program_from_layout(L)
    v_raw = verify_scene(raw)
    pen_bodies = {x for x, y, s in v_raw.pairs if s < 0} | {y for x, y, s in v_raw.pairs if s < 0}
    names = [b.name for b in raw.free()]
    palette = {nm: PALETTE[i % len(PALETTE)] for i, nm in enumerate(names)}
    colors = dict(palette); colors["table"] = TABLE
    colors_raw = {nm: (RED if nm in pen_bodies else palette[nm]) for nm in names}; colors_raw["table"] = TABLE

    # --- S4R on the generated program, then G5, then the agent route -------------------------------
    rep, res, states, t_rep = run_repair(L, text, tops, record=True)
    v_rep = verify_scene(rep)
    g5 = {m: settle_and_measure(rep, seconds=a.settle_seconds, timestep=1 / 600, support_tops=tops, decompose_free=(m == "coacd"), record_every=20)
          for m in ("hull", "coacd")}
    off_support = sorted(set(g5["hull"].left_support) | set(g5["coacd"].left_support))
    routed_text = text.rstrip() + "".join(f"\n  within({c}, table.top, inset={a.inset})" for c in off_support) + "\n"
    if off_support:
        routed, res2, _, t_route = run_repair(L, routed_text, tops, record=False)
        g5r = {m: settle_and_measure(routed, seconds=a.settle_seconds, timestep=1 / 600, support_tops=tops, decompose_free=(m == "coacd"), record_every=20)
               for m in ("hull", "coacd")}
    else:
        routed, res2, t_route, g5r = rep, res, 0.0, g5
    g5_raw = {m: settle_and_measure(raw, seconds=a.settle_seconds, timestep=1 / 600, support_tops=tops, decompose_free=(m == "coacd"), record_every=20)
              for m in ("hull", "coacd")}
    summary = {"n": a.n, "seed": a.seed, "pen_raw": v_raw.pen_pairs, "pen_rep": v_rep.pen_pairs, "rmsd": res.rmsd, "steps": res.steps,
               "repair_s": t_rep, "off_support": off_support, "route_s": t_route,
               "g5": {k: {m: {"vmax": r.peak_speed, "dmax": r.peak_disp, "left": r.left_support} for m, r in d.items()}
                      for k, d in (("raw", g5_raw), ("rep", g5), ("routed", g5r))}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)

    # --- video 1: gate ------------------------------------------------------------------------------
    shots = {"assets": {b.name: {"usd": None, "c_model": [float(v) for v in b.meta.get("c_model", np.zeros(3))], "fixed": bool(b.fixed)} for b in raw.bodies},
             "camera": CAM, "gate": [], "settle": []}
    for o in L["objects"]:
        shots["assets"][o["name"]]["usd"] = o.get("usd_path")
    if "table" in shots["assets"] and L.get("table"):
        shots["assets"]["table"]["usd"] = L["table"].get("usd_path")
    R = None if a.dump else FrameRenderer(raw.bodies, floor_z=-0.80)
    if R is not None:
        R.set_camera(**CAM)
    fi = 0

    def emit(frame, cols, texts, hidden=()):
        nonlocal fi
        if a.dump:
            shots["gate"].append({"poses": {k: [np.asarray(c).tolist(), np.asarray(Rm).tolist(), float(sc)] for k, (c, Rm, sc) in frame.items()},
                                  "red": sorted(k for k, v in cols.items() if np.allclose(v, RED)), "hidden": list(hidden),
                                  "texts": [[t, list(xy), sz, list(rgb), b] for t, xy, sz, rgb, b in texts]})
        else:
            R.render(frame, cols, out / "gate" / f"f_{fi:05d}.png", texts=texts, hidden=hidden)
        fi += 1

    raw_frame = poses(raw)
    for _ in range(int(2.0 * FPS)):
        emit(raw_frame, colors_raw, title("RoboLab layout as generated", f"{v_raw.pen_pairs} penetrating pairs on the meshes (red)", RED,
                                          foot="franka_table, objects placed by RoboLab's SpatialSolver, before its physics settle"))
    # continuation: interpolate the recorded states over repair-seconds
    n_states = len(states); total = int(a.repair_seconds * FPS)
    for f in range(total):
        u = f / max(total - 1, 1) * (n_states - 1); i0 = int(np.floor(u)); i1 = min(i0 + 1, n_states - 1); t = u - i0
        s0, p0 = states[i0]; s1, p1 = states[i1]
        frame = interpolate(p0, p1, t); s = (1 - t) * s0 + t * s1
        phase = "scale continuation" if s < 1.0 - 1e-9 else "full-scale tail refinement"
        emit(frame, colors, title("S4R repair", f"{phase}   s = {s:.2f}", INK,
                                  foot="every body shrinks about its reference centre, stays upright on the plate, and moves in-plane (x, y, yaw) as it grows back"))
    rep_frame = states[-1][1]
    for _ in range(int(2.0 * FPS)):
        emit(rep_frame, colors, title("After S4R", f"{v_rep.pen_pairs} penetrating pairs   ·   RMSD {100 * res.rmsd:.1f} cm   ·   {t_rep:.0f} s", GREEN,
                                      foot="verified on the full-resolution meshes with the shared evaluator"))
    if off_support:
        cols_c = dict(colors); [cols_c.__setitem__(c, RED) for c in off_support]
        for _ in range(int(2.5 * FPS)):
            emit(rep_frame, cols_c, title("G5: settle in MuJoCo and measure", f"{', '.join(off_support)} left the table  ->  agent adds  within({off_support[0]}, table.top, inset={a.inset})", RED,
                                          foot="the verdict names the body; the route is a DSL statement, not a manual fix"))
        routed_frame = poses(routed); n_m = int(1.5 * FPS)
        for f in range(n_m):
            t = f / max(n_m - 1, 1)
            emit(interpolate(rep_frame, routed_frame, t), cols_c, title("Re-repair with the added statement", f"{t_route:.0f} s", INK))
        for _ in range(int(2.5 * FPS)):
            emit(routed_frame, colors, title("After S4R + agent route", "0 penetrating pairs   ·   G5 pass with hull and CoACD proxies", GREEN,
                                             foot="peak speed %.2f / %.2f m/s, nothing leaves the table" % (g5r["hull"].peak_speed, g5r["coacd"].peak_speed)))
    if not a.dump:
        encode(out / "gate", out / "gate.mp4", fps=FPS)
        print("wrote", out / "gate.mp4", "frames", fi, flush=True)

    # --- video 2: settle, three panels --------------------------------------------------------------
    panels = [("RoboLab layout as generated", raw, g5_raw["coacd"]), ("after S4R", rep, g5["coacd"]), ("after S4R + agent route", routed, g5r["coacd"])]
    CAM2 = dict(target=(0.48, 0.05, -0.08), dist=1.55, elev_deg=30.0, azim_deg=-40.0)
    shots["camera_settle"] = CAM2
    R2 = None if a.dump else FrameRenderer(raw.bodies, width=960, height=720, floor_z=-0.80)
    if R2 is not None:
        R2.set_camera(**CAM2)
    n_frames = min(len(p[2].trajectory) for p in panels)
    tmp = out / "settle"
    for f in range(n_frames):
        pngs = []
        for k, (label, sc, rep_k) in enumerate(panels):
            t, pose = rep_k.trajectory[f]
            frame = {nm: (np.asarray(p[0]), quat_wxyz_to_R(p[1]), 1.0) for nm, p in pose.items()}
            cols = dict(colors)
            for nm in rep_k.left_support:
                cols[nm] = RED
            texts = title(label, f"t = {t:.2f} s   peak speed {rep_k.peak_speed:.2f} m/s" + (f"   off the table: {', '.join(rep_k.left_support)}" if rep_k.left_support else ""),
                          RED if rep_k.left_support else GREEN)
            if k == 0:
                texts.append(("MuJoCo, CoACD convex pieces (PhysX-like), real time", (30, 672), 20, GREY, False))
            if a.dump:
                shots["settle"].append({"panel": k, "frame": f, "poses": {nm: [np.asarray(c).tolist(), np.asarray(Rm).tolist(), float(sc)] for nm, (c, Rm, sc) in frame.items()},
                                        "red": sorted(nm for nm, v in cols.items() if np.allclose(v, RED)), "hidden": [],
                                        "texts": [[t, list(xy), sz, list(rgb), b] for t, xy, sz, rgb, b in texts]})
                continue
            png = tmp / f"p{k}_{f:05d}.png"
            R2.render(frame, cols, png, texts=texts); pngs.append(png)
        if not a.dump:
            hstack(pngs, tmp / f"f_{f:05d}.png")
    if a.dump:
        json.dump(shots, open(out / "shots.json", "w"))
        print("wrote", out / "shots.json", "gate frames", len(shots["gate"]), "settle shots", len(shots["settle"]), flush=True)
    else:
        encode(tmp, out / "settle.mp4", fps=FPS)
        print("wrote", out / "settle.mp4", "frames", n_frames, flush=True)


if __name__ == "__main__":
    main()
