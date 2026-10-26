"""Shots for an example in docs/examples (a layout of SimReady packages and the program its agent wrote), rendered
later by viz/blender_render.py (no text overlays):

  gate = [the layout as written (penetrating bodies tinted) -> S4R scale continuation -> the repaired layout]

The repair is the one `simready.cli repair` runs, with the schedule of the example's program (the one its certificate
records), and its last frame must be the example's repaired.json. The packages must have been fetched with their textures
(python -m simready.cli assets fetch <package> --repo <repo> --textures).

    python viz/make_example_video.py docs/examples/shelf_books --out results/videos/shelf_books --azim 210 --elev 10 --dist 1.6 --target 0.65 0 0.62
    python viz/make_example_video.py docs/examples/shelf_pantry --out results/videos/shelf_pantry --azim 205 --elev 16 --dist 1.6 --target 0.72 0 0.70
    viz/run_blender.sh results/videos/shelf_books/shots.json gate -1 0 results/videos/shelf_books/blender_gate --width 1200 --height 900

(the README's shelf_fix clip takes the shelf_pantry camera)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from simready.cli import load_any, load_program, repair_scene  # noqa: E402
from simready.dsl.model import schedule_of  # noqa: E402
from simready.gates import verify_scene  # noqa: E402
from simready.io.nvidia_assets import package_dir, read_package  # noqa: E402
from render_frames import interpolate  # noqa: E402
from usd_to_obj import export  # noqa: E402

FPS = 30


def poses(scene):
    return {b.name: (b.center.copy(), b.rotation.copy(), 1.0) for b in scene.free()}


def ser(frame):
    return {k: [np.asarray(c).tolist(), np.asarray(R).tolist(), float(s)] for k, (c, R, s) in frame.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example"); ap.add_argument("--out", required=True)
    ap.add_argument("--hold-seconds", type=float, default=2.0); ap.add_argument("--shrink-seconds", type=float, default=1.0)
    ap.add_argument("--repair-seconds", type=float, default=7.0)
    ap.add_argument("--elev", type=float, default=20.0); ap.add_argument("--azim", type=float, default=180.0)
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--target", type=float, nargs=3, default=None, help="the point the camera looks at (default: the fixtures' centre)")
    a = ap.parse_args()
    ex, out = Path(a.example), Path(a.out)
    assets_dir = out / "assets"
    cert = json.loads((ex / "repaired.certificate.json").read_text())
    if Path(cert["scene"]).name != "layout.json":
        raise ValueError(f"the certificate repaired {cert['scene']}, not the example's layout.json")
    layout = json.loads((ex / "layout.json").read_text())
    sc = load_any(str(ex / "layout.json"))
    prog = load_program(str(ex / Path(cert["program"]).name), sc)[0]

    # the render mesh is the solver mesh in the same frame
    entries = {e["name"]: e for e in layout.get("fixtures", []) + layout["objects"]}
    for b in sc.bodies:                  # export refuses a material it cannot render as authored (check_renderable)
        e = entries[b.name]
        info = export(read_package(package_dir(e["package"], e["repo"]))["root_usd"], assets_dir, name=b.name,
                      rotation=b.meta.get("rest_rotation"))
        if not np.allclose(np.asarray(info["aabb"][0]) - b.meta["c_model"], b.verts.min(0), atol=1e-6):
            raise RuntimeError(f"render/solver mesh mismatch for {b.name}")

    v_raw = verify_scene(sc)
    tint = sorted({n for x, y, _ in v_raw.pairs for n in (x, y) if not sc[n].fixed})
    raw_frame = poses(sc)
    states = []

    def on_step(st):
        states.append({sc.bodies[k].name: (sc.bodies[k].center.copy(), sc.bodies[k].rotation.copy(), st.body_scale(k)) for k in st.free})

    if cert["provenance"]["schedule"] != schedule_of(prog):
        raise ValueError(f"the certificate's schedule {cert['provenance']['schedule']} is not the program's {schedule_of(prog)}")
    rep = repair_scene(sc, prog, on_step=on_step)
    states.append(poses(sc))
    done = {o["name"]: o for o in json.loads((ex / "repaired.json").read_text())["objects"]}
    dx = dyaw = 0.0
    for b in sc.free():
        o = done[b.name]
        dx = max(dx, float(np.abs(b.center - b.rotation @ b.meta["c_model"] - [o["x"], o["y"], o["z"]]).max()))
        yaw = math.degrees(math.atan2(b.rotation[1, 0], b.rotation[0, 0]))
        dyaw = max(dyaw, abs((yaw - o["yaw"] + 180.0) % 360.0 - 180.0))
    if dx > 1e-4 or dyaw > 1e-3:
        raise RuntimeError(f"the repair ends {dx:.2e} m / {dyaw:.2e} deg away from {ex / 'repaired.json'}")
    summary = {"example": str(ex), "pen_raw": v_raw.pen_pairs, "pen_rep": rep["after"].pen_pairs, "ok": rep["ok"],
               "steps": rep["res"].steps, "repair_s": rep["time_s"], "tinted": tint, "states": len(states),
               "max_offset_from_repaired_json_m": dx, "max_turn_from_repaired_json_deg": dyaw}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n"); print(json.dumps(summary, indent=1), flush=True)

    fixed = [b for b in sc.bodies if b.fixed]
    lo = np.min([b.world_aabb()[0] for b in fixed], 0); hi = np.max([b.world_aabb()[1] for b in fixed], 0)
    cam = {"target": a.target or (0.5 * (lo + hi)).tolist(), "dist": a.dist or 1.6 * float(np.linalg.norm(hi - lo)), "elev_deg": a.elev,
           "azim_deg": a.azim}
    assets = {b.name: {"obj": str((assets_dir / f"{b.name}.obj").resolve()), "c_model": np.asarray(b.meta["c_model"]).tolist(),
                       "fixed": bool(b.fixed), "pose": [b.center.tolist(), b.rotation.tolist()]} for b in sc.bodies}
    shots = {"assets": assets, "camera": cam, "camera_settle": cam, "floor_z": float(min(b.world_aabb()[0][2] for b in sc.bodies)),
             "gate": [], "settle": []}
    hold, shrink, total = int(a.hold_seconds * FPS), int(a.shrink_seconds * FPS), int(a.repair_seconds * FPS)
    shots["gate"] += [{"poses": ser(raw_frame), "red": tint, "hidden": []}] * hold
    for f in range(shrink):                  # down to the first accepted step at the smallest scale
        shots["gate"].append({"poses": ser(interpolate(raw_frame, states[0], f / shrink)), "red": [], "hidden": []})
    for f in range(total):
        u = f / max(total - 1, 1) * (len(states) - 1); i0 = int(u); i1 = min(i0 + 1, len(states) - 1)
        shots["gate"].append({"poses": ser(interpolate(states[i0], states[i1], u - i0)), "red": [], "hidden": []})
    shots["gate"] += [{"poses": ser(states[-1]), "red": [], "hidden": []}] * hold
    (out / "shots.json").write_text(json.dumps(shots))
    print("wrote", out / "shots.json", "gate frames", len(shots["gate"]), flush=True)


if __name__ == "__main__":
    main()
