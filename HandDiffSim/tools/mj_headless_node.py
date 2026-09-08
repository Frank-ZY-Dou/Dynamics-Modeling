#!/usr/bin/env python
"""Headless wrapper around the repo's dora simulation node (for machines without a display).

It replaces mujoco.viewer.launch_passive with a no-op "viewer" that reports it is running
for --seconds seconds, then runs the *unchanged* Client class from
AHSimulation.mj_mink_right / mj_mink_left.  When the fake viewer stops, Client.run() exits
normally; we then print the last motor command vector and render one EGL frame to PNG.

Used by tools/dataflow_angle_simu_headless.yml.
"""
import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402


class _FakeViewer:
    def __init__(self, seconds):
        self.t_end = time.time() + seconds
        self.n_sync = 0

    def is_running(self):
        return time.time() < self.t_end

    def sync(self):
        self.n_sync += 1


def install_fake_viewer(seconds):
    @contextlib.contextmanager
    def fake_launch_passive(model, data, **kwargs):
        yield _FakeViewer(seconds)

    mujoco.viewer.launch_passive = fake_launch_passive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", choices=["pos", "quat"], default="quat")
    ap.add_argument("-s", "--side", choices=["right", "left"], default="right")
    ap.add_argument("--seconds", type=float, default=5.0, help="wall-clock seconds to run before exiting")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "renders"))
    args = ap.parse_args()

    install_fake_viewer(args.seconds)
    if args.side == "right":
        from AHSimulation import mj_mink_right as sim
    else:
        from AHSimulation import mj_mink_left as sim

    client = sim.Client(args.mode)
    t0 = time.time()
    client.run()  # returns when the fake viewer "closes"
    elapsed = time.time() - t0

    motors = np.degrees(np.asarray(client.motor_pos, dtype=float)) if len(client.motor_pos) else None
    print(f"[headless {args.side}] ran {elapsed:.1f}s; last motor command [deg] = "
          f"{np.array2string(motors, precision=1) if motors is not None else 'none (no tick received)'}",
          flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    try:
        renderer = mujoco.Renderer(client.model, height=480, width=480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.03, 0.0, 0.10]
        cam.distance, cam.azimuth, cam.elevation = 0.32, 160, -20
        renderer.update_scene(client.data, camera=cam)
        rgb = renderer.render()
        png = Path(args.out) / f"dora_{args.side}_last_frame.png"
        try:
            import cv2
            cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            from PIL import Image
            Image.fromarray(rgb).save(png)
        print(f"[headless {args.side}] wrote {png}", flush=True)
        renderer.close()
    except Exception as e:  # rendering is best-effort
        print(f"[headless {args.side}] render skipped: {e}", flush=True)
    sys.exit(0 if motors is not None else 1)


if __name__ == "__main__":
    main()
