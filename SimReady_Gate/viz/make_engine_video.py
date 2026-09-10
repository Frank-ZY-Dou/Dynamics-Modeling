"""Side-by-side clip of one exported scene running in MuJoCo, Genesis and Isaac Sim.

Stage 1 (``--shots``): turn the runners' pose recordings (``examples/run_*.py --record``) into a
shots file for ``viz/blender_render.py``, one panel per engine, reusing the assets and the settle
camera of an existing shots.json (the textured OBJs and the camera of the Example 2 videos).

    python viz/make_engine_video.py --base results/videos/pile_n20/shots.json \
        --record mujoco=poses_mujoco.json genesis=poses_genesis.json isaac=poses_isaac.json \
        --out results/videos/pile_n20_engines

Stage 2 (``--assemble``): after ``viz/run_blender_hq.sh <out>/shots.json settle <panel> <gpu>
<out>/blender_settle`` has rendered every panel, put the panels side by side (``--labels`` writes
the engine name and time into each panel) and encode ``engines.mp4`` and ``engines.gif``.

    python viz/make_engine_video.py --out results/videos/pile_n20_engines --assemble
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_frames import encode, hstack, overlay, quat_wxyz_to_R  # noqa: E402

INK = (0.12, 0.12, 0.12)
GREY = (0.45, 0.45, 0.45)
LABELS = {"mujoco": "MuJoCo", "genesis": "Genesis", "isaac": "Isaac Sim"}


def load_recording(path):
    rec = json.load(open(path))
    frames = []
    for f in rec["frames"]:
        frames.append((float(f["t"]), {n: (np.asarray(p[0], dtype=float), quat_wxyz_to_R(p[1])) for n, p in f["poses"].items()}))
    return rec, frames


def resample(frames, times):
    """Pose at each requested time by nearest recorded frame (recordings are at ~30 fps)."""
    ts = np.array([t for t, _ in frames])
    return [frames[int(np.argmin(np.abs(ts - t)))][1] for t in times]


def build_shots(base, records, out, seconds, fps):
    shots_base = json.load(open(base))
    shots = {"assets": shots_base["assets"], "camera_settle": dict(shots_base["camera_settle"]),
             "floor_z": shots_base.get("floor_z"), "always_hidden": shots_base.get("always_hidden", []),
             "engines": [], "settle": []}
    # A steady camera: the three panels are compared, not toured.
    shots["camera_settle"].pop("azim_deg_end", None); shots["camera_settle"].pop("dist_end", None)
    times = np.arange(0.0, seconds + 1e-9, 1.0 / fps)
    labels = []
    for k, (engine, path) in enumerate(records):
        rec, frames = load_recording(path)
        version = rec.get("version", "")
        labels.append({"panel": k, "engine": engine, "label": LABELS.get(engine, engine) + (f" {version}" if version else "")})
        for f, pose in enumerate(resample(frames, times)):
            poses = {n: [c.tolist(), R.tolist(), 1.0] for n, (c, R) in pose.items()}
            shots["settle"].append({"panel": k, "frame": f, "t": float(times[f]), "poses": poses,
                                    "red": [], "hidden": list(shots["always_hidden"])})
    shots["engines"] = labels
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    json.dump(shots, open(out / "shots.json", "w"))
    print("wrote", out / "shots.json", "panels", len(labels), "frames per panel", len(times), flush=True)


def assemble(out, fps, gif_width, gif_fps, labels_on=False):
    out = Path(out)
    shots = json.load(open(out / "shots.json"))
    labels = {e["panel"]: e["label"] for e in shots["engines"]}
    src = out / "blender_settle"
    panels = sorted(labels)
    n = min(len(glob.glob(str(src / f"p{k}_*.png"))) for k in panels)
    if n == 0:
        raise SystemExit("no rendered panels found under " + str(src))
    times = {(s["panel"], s["frame"]): s["t"] for s in shots["settle"]}
    tmp = out / "frames"; tmp.mkdir(exist_ok=True)
    for f in range(n):
        pngs = []
        for k in panels:
            png = tmp / f"p{k}_{f:05d}.png"
            subprocess.run(["cp", str(src / f"p{k}_{f:05d}.png"), str(png)], check=True)
            if labels_on:
                overlay(str(png), [(labels[k], (40, 32), 44, INK, True), (f"t = {times[(k, f)]:.2f} s", (40, 92), 30, GREY, False)])
            pngs.append(png)
        hstack(pngs, tmp / f"f_{f:05d}.png", gap=8)
    encode(tmp, out / "engines.mp4", fps=fps)
    gif = out / "engines.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "engines.mp4"), "-vf",
                    f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    print("wrote", out / "engines.mp4", "and", gif, "frames", n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="results/videos/pile_n20/shots.json", help="shots.json whose assets and settle camera are reused")
    ap.add_argument("--record", nargs="*", default=[], help="engine=poses.json, in panel order")
    ap.add_argument("--seconds", type=float, default=2.0); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--assemble", action="store_true"); ap.add_argument("--gif-width", type=int, default=1440); ap.add_argument("--gif-fps", type=float, default=12.5)
    ap.add_argument("--labels", action="store_true", help="write the engine name and time into each panel (off: the caption names the panels)")
    a = ap.parse_args()
    if a.assemble:
        assemble(a.out, a.fps, a.gif_width, a.gif_fps, labels_on=a.labels)
    else:
        records = [tuple(r.split("=", 1)) for r in a.record]
        if not records:
            raise SystemExit("give at least one --record engine=poses.json")
        build_shots(a.base, records, a.out, a.seconds, a.fps)


if __name__ == "__main__":
    main()
