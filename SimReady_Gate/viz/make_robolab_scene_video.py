"""Shots for a RoboLab scene video, rendered later by viz/blender_render.py (no text overlays).

Two inputs are supported:
  --scene <file.usda>        one of RoboLab's shipped scenes (their fixtures, their objects, their poses)
  --layout N SEED            a pre-settle layout from RoboLab's own SpatialSolver, staged in their base
                             scene (table_oak on franka_table with a ground plane) - larger overlaps

Shots: gate = [raw hold (penetrating bodies tinted) -> S4R scale continuation -> repaired hold]
       settle = MuJoCo trajectories, panel 0 = raw layout, panel 1 = after S4R (and panel 2 = after
       the agent route when G5 named a body that left its support).
Usage: python viz/make_robolab_scene_video.py --scene .../workdesk_snacks.usda --out results/videos/workdesk
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "experiments"); sys.path.insert(0, "viz")
import simready  # noqa: E402,F401
from simready.scene import Scene, MeshProxy  # noqa: E402
from simready.gates import verify_scene  # noqa: E402
from simready.gates.settle_mujoco import settle_and_measure  # noqa: E402
from simready.io.usd_io import load_scene_usda, body_from_object_usd  # noqa: E402
from simready.dsl import parse_program, compile_program, check_predicates  # noqa: E402
from simready.repair import repair_upright  # noqa: E402
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh  # noqa: E402
from render_frames import interpolate, quat_wxyz_to_R  # noqa: E402
from usd_to_obj import export_scene, export  # noqa: E402

FPS = 30
ROBOLAB = Path(os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "RoboLab")))
BASE_SCENE = ROBOLAB / "assets/scenes/base_empty.usda"


def poses(scene):
    return {b.name: (b.center.copy(), b.rotation.copy(), 1.0) for b in scene.free()}


def ser(frame):
    return {k: [np.asarray(c).tolist(), np.asarray(R).tolist(), float(s)] for k, (c, R, s) in frame.items()}


def program_for(scene, support, margin=0.003, inset=0.0, extra=""):
    fixed = "  ".join(f"fixed({b.name})" for b in scene.bodies if b.fixed)
    return (f"program\n  no_penetration(*, margin={margin})\n  {fixed}\n  on_support(*, {support})   upright(*)\n"
            f"  within(*, {support}.top, inset={inset})\n  minimize displacement(*)\n{extra}")


def repair(scene, text, tops, record=True, faces=2000, s_min=0.05):
    sc = copy.deepcopy(scene)
    prog = parse_program(text)
    spec = compile_program(prog, sc, s_min=s_min, ds_max=0.05, tail_iters=30)
    states = []

    def on_step(st):
        states.append((st.scale, {sc.bodies[k].name: (sc.bodies[k].center.copy(), sc.bodies[k].rotation.copy(), st.body_scale(k)) for k in st.free}))

    t0 = time.time()
    with MeshProxy(sc, faces=faces, slab_tops=tops):
        res = repair_upright(sc, spec, on_step=on_step if record else None)
    reseat_on_supports(sc, spec)
    if verify_scene(sc).pen_pairs > 0:
        polish_full_mesh(sc, spec, iters=15, on_step=on_step if record else None); reseat_on_supports(sc, spec)
    states.append((1.0, poses(sc)))
    fails = [p[0] for p in check_predicates(prog, sc) if not p[1]]
    repair.last_spec = spec                  # the storyboard reads the placement targets and s_min from it
    return sc, res, states, time.time() - t0, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene"); ap.add_argument("--layout", nargs=2, type=int, metavar=("N", "SEED"))
    ap.add_argument("--pile", nargs=2, type=int, metavar=("N", "SEED"), help="N catalog objects dropped into a small region of RoboLab's base scene (heavy overlap)")
    ap.add_argument("--pile-radius", type=float, default=0.13); ap.add_argument("--program", help="DSL program file for --pile (default: generic)")
    ap.add_argument("--orbit", type=float, default=0.0, help="camera azimuth sweep (deg) over the gate sequence")
    ap.add_argument("--s-min", type=float, default=0.05, help="initial scale of the continuation (larger values keep the shrunken bodies visible)")
    ap.add_argument("--out", required=True); ap.add_argument("--support", default="table")
    ap.add_argument("--repair-seconds", type=float, default=7.0); ap.add_argument("--hold-seconds", type=float, default=2.0)
    ap.add_argument("--settle-seconds", type=float, default=2.5); ap.add_argument("--inset", type=float, default=0.10)
    ap.add_argument("--elev", type=float, default=30.0); ap.add_argument("--azim", type=float, default=-35.0); ap.add_argument("--dist", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); assets_dir = out / "assets"

    # --- scene + render assets ------------------------------------------------------------------------
    if a.pile:
        from robolab_offline_layouts import pick_objects, ROBOLAB as RL_ROOT
        from robolab_diagnostic import base_scene, seat_scene
        n, seed = a.pile
        rng = np.random.RandomState(seed)
        objs = pick_objects(rng, n)
        infos = export_scene(str(BASE_SCENE), assets_dir)
        raw_hover = base_scene()
        bodies = list(raw_hover.bodies)
        cx, cy = 0.55, 0.0
        for o in objs:
            r = a.pile_radius * math.sqrt(rng.uniform(0, 1)); th = rng.uniform(0, 2 * math.pi)
            b = body_from_object_usd(o["name"], str(RL_ROOT / o["usd_path"]), (cx + r * math.cos(th), cy + r * math.sin(th), 0.3), yaw_deg=float(rng.uniform(0, 360)), tags={"object"})
            infos[o["name"]] = export(str(RL_ROOT / o["usd_path"]), assets_dir, name=o["name"]); infos[o["name"]]["c_model"] = b.meta["c_model"].tolist()
            bodies.append(b)
        raw = seat_scene(Scene(bodies))                        # every object dropped onto the table inside the pile
        source = {"pile": {"n": n, "seed": seed, "radius": a.pile_radius, "objects": [o["name"] for o in objs]}}
        layout_program = None
        if a.program:
            txt = open(a.program).read()
            if a.program.endswith(".json"):
                from simready.dsl.schema import validate_program_json
                layout_program, _notes = validate_program_json(json.loads(txt), raw)
            else:
                layout_program = txt
    elif a.scene:
        raw = load_scene_usda(a.scene)
        infos = export_scene(a.scene, assets_dir)
        source = {"scene": a.scene}
    else:
        from robolab_offline_layouts import get_layout
        from robolab_diagnostic import scene_from_layout, seat_scene, program_from_layout as program_from_layout_rl
        n, seed = a.layout
        L = get_layout(n, seed)
        infos = export_scene(str(BASE_SCENE), assets_dir)
        raw_as_written = scene_from_layout(L)                 # RoboLab's own z convention (hovering)
        for o in L["objects"]:
            infos[o["name"]] = export(o["usd_path"], assets_dir, name=o["name"])
            infos[o["name"]]["c_model"] = raw_as_written[o["name"]].meta["c_model"].tolist()
        raw = seat_scene(raw_as_written)                        # the like-for-like state the repair starts from
        source = {"layout": {"n": n, "seed": seed, "solver_ok": L["ok"], "relations": L["relations"]}}
        layout_program = program_from_layout_rl(L)
    for b in raw.bodies:                      # the render mesh must be the solver mesh in the same frame
        info = infos[b.name]
        lo_i = np.asarray(info["aabb"][0]) - np.asarray(info["c_model"])
        if not np.allclose(lo_i, b.verts.min(0), atol=1e-6) or ("center" in info and not np.allclose(info["center"], b.center, atol=1e-6)):
            raise RuntimeError(f"render/solver mesh mismatch for {b.name}")
    support = a.support
    tops = {support: min(raw.support_height(raw[support], at=b.center[:2], radius=0.1) for b in raw.free())}
    v_raw = verify_scene(raw)
    # tint only the bodies in object-object penetrations (sub-mm resting contact with the fixture is
    # the engine's rest offset)
    oo = [(x, y) for x, y, s in v_raw.pairs if s < 0 and not raw[x].fixed and not raw[y].fixed]
    pen_bodies = sorted({n for pr in oo for n in pr})

    # --- S4R, G5, route ----------------------------------------------------------------------------------
    text = layout_program if (a.layout or (a.pile and layout_program)) else program_for(raw, support, margin=0.01, inset=0.02)
    rep, res, states, t_rep, fails = repair(raw, text, tops, s_min=a.s_min)
    v_rep = verify_scene(rep)
    g5 = {m: settle_and_measure(rep, seconds=a.settle_seconds, timestep=1 / 600, support_tops=tops, decompose_free=(m == "coacd"), record_every=20)
          for m in ("hull", "coacd")}
    off_support = sorted(set(g5["hull"].left_support) | set(g5["coacd"].left_support))
    routed, g5r, t_route = None, None, 0.0
    if off_support:
        text2 = text.rstrip() + "".join(f"\n  within({c}, {support}.top, inset={a.inset})" for c in off_support) + "\n"
        routed, _, _, t_route, _ = repair(raw, text2, tops, record=False, s_min=a.s_min)
        g5r = {m: settle_and_measure(routed, seconds=a.settle_seconds, timestep=1 / 600, support_tops=tops, decompose_free=(m == "coacd"), record_every=20)
               for m in ("hull", "coacd")}
    g5_raw = {m: settle_and_measure(raw, seconds=a.settle_seconds, timestep=1 / 600, support_tops=tops, decompose_free=(m == "coacd"), record_every=20)
              for m in ("hull", "coacd")}
    summary = {"source": source, "n_free": len(raw.free()), "pen_raw": v_raw.pen_pairs, "max_pen_raw_mm": 1000 * v_raw.max_pen,
               "pen_raw_obj_obj": sum(1 for x, y, s in v_raw.pairs if s < 0 and not raw[x].fixed and not raw[y].fixed),
               "pen_rep": v_rep.pen_pairs, "rmsd": res.rmsd, "steps": res.steps, "repair_s": t_rep, "fails": fails,
               "off_support": off_support, "route_s": t_route,
               "g5": {k: {m: {"vmax": r.peak_speed, "dmax": r.peak_disp, "left": r.left_support} for m, r in d.items()}
                      for k, d in (("raw", g5_raw), ("rep", g5)) + ((("routed", g5r),) if g5r else ())}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1), flush=True)

    # --- shots ---------------------------------------------------------------------------------------------
    free = raw.free(); lo = np.min([b.world_aabb()[0] for b in free], 0); hi = np.max([b.world_aabb()[1] for b in free], 0)
    c = 0.5 * (lo + hi); ext = float(np.linalg.norm((hi - lo)[:2]))
    cam = {"target": [float(c[0]), float(c[1]), float(tops[support]) + 0.05], "dist": a.dist or max(1.1, 1.35 * ext), "elev_deg": a.elev, "azim_deg": a.azim}
    if a.orbit:
        cam["azim_deg_end"] = a.azim + a.orbit
    grounds = [b for b in raw.bodies if b.fixed and "ground" in b.tags]
    floor_z = float(grounds[0].world_aabb()[1][2]) if grounds else float(min(b.world_aabb()[0][2] for b in raw.bodies))
    ground_names = [b.name for b in grounds]     # rendered as the renderer's own floor plane instead
    assets = {}
    for b in raw.bodies:
        info = infos[b.name]
        assets[b.name] = {"obj": str((assets_dir / f"{b.name}.obj").resolve()), "c_model": list(map(float, info["c_model"])), "fixed": bool(b.fixed),
                          "pose": [b.center.tolist(), b.rotation.tolist()]}
    shots = {"assets": assets, "camera": cam, "camera_settle": dict(cam, dist=cam["dist"] * 1.15, target=[cam["target"][0], cam["target"][1], cam["target"][2] - 0.12]),
             "floor_z": floor_z, "always_hidden": ground_names, "gate": [], "settle": []}
    raw_frame = poses(raw)
    for _ in range(int(a.hold_seconds * FPS)):
        shots["gate"].append({"poses": ser(raw_frame), "red": pen_bodies, "hidden": []})
    spec_used = getattr(repair, "last_spec", None)
    placed = dict(spec_used.place) if spec_used is not None and spec_used.place else {}
    if placed:
        s_min = spec_used.s_min; top = tops[support]
        def scaled_frame(xy_of, yaw_of, s):
            fr = {}
            for b in raw.free():
                x, y = xy_of(b); yw = yaw_of(b)
                R = np.array([[np.cos(yw), -np.sin(yw), 0.0], [np.sin(yw), np.cos(yw), 0.0], [0.0, 0.0, 1.0]])
                b2 = copy.copy(b); b2.rotation = R
                z = top + b2.support_offset(raw.up, s)
                fr[b.name] = (np.array([x, y, z]), R, s)
            return fr
        yaw0 = {b.name: float(np.arctan2(b.rotation[1, 0], b.rotation[0, 0])) for b in raw.free()}
        xy0 = {b.name: b.center[:2].copy() for b in raw.free()}
        # (2) shrink at the original poses
        n_sh = int(2.5 * FPS)
        for f in range(n_sh):
            u = f / max(n_sh - 1, 1); sc_ = 1.0 + (s_min - 1.0) * u
            shots["gate"].append({"poses": ser(scaled_frame(lambda b: xy0[b.name], lambda b: yaw0[b.name], sc_)), "red": [], "hidden": []})
        # (3) the placement step in the shrunken scale-space
        n_dr = int(3.0 * FPS)
        for f in range(n_dr):
            u = f / max(n_dr - 1, 1); u = 0.5 - 0.5 * np.cos(np.pi * u)
            def xy_of(b, u=u):
                if b.name in placed:
                    tx, ty, _ = placed[b.name]; return (1 - u) * xy0[b.name] + u * np.array([tx, ty])
                return xy0[b.name]
            def yaw_of(b, u=u):
                if b.name in placed and placed[b.name][2] is not None:
                    t = np.radians(placed[b.name][2]); d = (t - yaw0[b.name] + np.pi) % (2 * np.pi) - np.pi
                    return yaw0[b.name] + u * d
                return yaw0[b.name]
            shots["gate"].append({"poses": ser(scaled_frame(xy_of, yaw_of, s_min)), "red": [], "hidden": []})
    n_states = len(states); total = int(a.repair_seconds * FPS)
    for f in range(total):
        u = f / max(total - 1, 1) * (n_states - 1); i0 = int(np.floor(u)); i1 = min(i0 + 1, n_states - 1); t = u - i0
        shots["gate"].append({"poses": ser(interpolate(states[i0][1], states[i1][1], t)), "red": [], "hidden": []})
    rep_frame = states[-1][1]
    for _ in range(int(a.hold_seconds * FPS)):
        shots["gate"].append({"poses": ser(rep_frame), "red": [], "hidden": []})
    if routed is not None:
        routed_frame = poses(routed); n_m = int(1.5 * FPS)
        for _ in range(int(1.5 * FPS)):
            shots["gate"].append({"poses": ser(rep_frame), "red": off_support, "hidden": []})
        for f in range(n_m):
            shots["gate"].append({"poses": ser(interpolate(rep_frame, routed_frame, f / max(n_m - 1, 1))), "red": off_support, "hidden": []})
        for _ in range(int(a.hold_seconds * FPS)):
            shots["gate"].append({"poses": ser(routed_frame), "red": [], "hidden": []})
    panels = [(0, g5_raw["coacd"]), (1, g5["coacd"])] + ([(2, g5r["coacd"])] if g5r else [])
    n_frames = min(len(r.trajectory) for _, r in panels)
    for k, r in panels:
        for f in range(n_frames):
            t, pose = r.trajectory[f]
            frame = {nm: (np.asarray(p[0]), quat_wxyz_to_R(p[1]), 1.0) for nm, p in pose.items()}
            shots["settle"].append({"panel": k, "frame": f, "poses": ser(frame), "red": list(r.left_support) if f > n_frames // 2 else [], "hidden": []})
    json.dump(shots, open(out / "shots.json", "w"))
    print("wrote", out / "shots.json", "gate frames", len(shots["gate"]), "settle shots", len(shots["settle"]), "panels", len(panels), flush=True)


if __name__ == "__main__":
    main()
