#!/usr/bin/env python
"""Side-by-side video: left = official closed-chain linkage model, right = simplified serial-joint model.

Both are driven by the same motor command trajectory theta(t). The linkage is settled quasi-statically by its
position servos (gravity/friction off); the serial model gets the joint angles from the identified closure model
(motors -> Newton on the two RSS chains -> mcp -> pip).

    python make_video.py --side right            # -> video/linkage_vs_serial_right.mp4
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from ah_serial.kinematics import SerialHandKinematics
from ah_serial.linkage import LinkageHand
from ah_serial.render import FFmpegWriter, Offscreen, compose_side_by_side, make_camera
from ah_serial.validate import SerialModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "AmazingHand"
SCENES = {"right": REPO / "Demo/AHSimulation/AHSimulation/AH_Right/mjcf/scene.xml",
          "left": REPO / "Demo/AHSimulation/AHSimulation/AH_Left/mjcf/scene.xml"}
DEG = 180 / np.pi


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# (time [s], theta1 [deg], theta2 [deg]) applied to all fingers; the wave phase uses per-finger offsets
KEYFRAMES = [(0.0, 0, 0), (1.6, 85, -85), (3.4, -85, 85), (4.4, 0, 0),
             (5.4, 35, 35), (7.0, -35, -35), (8.0, 0, 0),
             (9.0, 60, -20), (10.2, 20, -60), (11.2, 0, 0)]
WAVE_START, WAVE_END, T_END = 11.2, 15.4, 16.0
PHASE_LABEL = [(0.0, "flexion / extension  (θ1 = −θ2)"), (4.4, "abduction / adduction  (θ1 = θ2)"),
               (8.0, "mixed flexion + abduction"), (11.2, "finger wave"), (15.4, "return to zero")]


def motor_trajectory(t: float) -> np.ndarray:
    """(4,2) motor angles [rad] at time t."""
    th = np.zeros((4, 2))
    if t < WAVE_START:
        for (t0, a0, b0), (t1, a1, b1) in zip(KEYFRAMES[:-1], KEYFRAMES[1:]):
            if t0 <= t <= t1:
                s = smoothstep((t - t0) / (t1 - t0))
                th[:] = np.radians([a0 + s * (a1 - a0), b0 + s * (b1 - b0)])
                break
    elif t < WAVE_END:
        for n in range(4):
            phase = 2 * np.pi * (t - WAVE_START) / 2.1 - 0.9 * n
            env = smoothstep((t - WAVE_START) / 0.8) * smoothstep((WAVE_END - t) / 0.8)
            a = env * (35.0 + 35.0 * np.sin(phase - np.pi / 2))          # 0..70 deg flexion command
            b = env * 15.0 * np.sin(phase * 0.5)                          # gentle abduction, |a|+|b| <= 85 < 90 deg limit
            th[n] = np.radians([a + b, -a + b])
    return th


def phase_label(t):
    lab = PHASE_LABEL[0][1]
    for t0, s in PHASE_LABEL:
        if t >= t0:
            lab = s
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=720, help="panel size in pixels")
    ap.add_argument("--settle-steps", type=int, default=80, help="linkage servo steps per frame (2 ms each)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    side = args.side
    out = Path(args.out) if args.out else HERE / "video" / f"linkage_vs_serial_{side}.mp4"

    H = LinkageHand(SCENES[side])
    K = SerialHandKinematics.from_json(HERE / "params" / f"identified_{side}.json")
    serial = SerialModel(HERE / "models" / f"AH_{side.capitalize()}" / "scene.xml")
    m, d = H.model, H.data
    m.opt.gravity[:] = 0
    m.dof_frictionloss[:] = 0
    act = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}") for i in (1, 2)] for n in (1, 2, 3, 4)])
    qadr = np.array([[m.jnt_qposadr[j] for j in f.motor_j] for f in H.fingers])
    H.reset_zero()

    cam = make_camera(lookat=(0.03, 0.0, 0.10), distance=0.30, azimuth=160, elevation=-20)
    r_link = Offscreen(m, (args.size, args.size), cam)
    r_ser = Offscreen(serial.model, (args.size, args.size), cam)
    title_l = f"original closed-chain linkage ({side} hand)"
    title_r = "simplified serial model: abd → mcp → pip"

    n_frames = int(T_END * args.fps)
    writer = None
    x_prev = np.zeros((4, 2))            # (abd, P) warm start per finger
    max_tip_err, max_track_err = 0.0, 0.0
    t0 = time.time()
    for k in range(n_frames):
        t = k / args.fps
        th = motor_trajectory(t)
        # linkage: quasi-static servo settle, warm-started from the previous frame
        d.ctrl[act[:, 0]] = th[:, 0]
        d.ctrl[act[:, 1]] = th[:, 1]
        for _ in range(args.settle_steps):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        track_err = np.abs(d.qpos[qadr] - th).max() * DEG
        # serial: identified closure model
        q = np.zeros((4, 3))
        for n in range(4):
            Kf = K.fingers[n + 1]
            x, _ = Kf.rss_forward(th[n], x0=x_prev[n])
            x_prev[n] = x
            mcp = Kf.mcp_of_P(x[1])
            q[n] = [x[0], mcp, Kf.pip_of_mcp(mcp)]
        serial.set_joints(q)
        tip_err = np.array([1e3 * np.linalg.norm(d.site_xpos[f.tip_s] - serial.data.site_xpos[serial.tip[i]])
                            for i, f in enumerate(H.fingers)])
        max_tip_err, max_track_err = max(max_tip_err, tip_err.max()), max(max_track_err, track_err)

        left, right = r_link.render(d), r_ser.render(serial.data)
        thd, qd = th * DEG, q * DEG
        footer = [
            f"t = {t:5.2f} s   {phase_label(t)}",
            "motors θ1/θ2 [deg]     " + "  ".join(f"f{n + 1} {thd[n, 0]:6.1f}/{thd[n, 1]:6.1f}" for n in range(4)),
            "serial abd/mcp/pip     " + "  ".join(f"f{n + 1} {qd[n, 0]:5.1f}/{qd[n, 1]:5.1f}/{qd[n, 2]:5.1f}" for n in range(4)),
            f"fingertip error linkage vs serial: max {tip_err.max():.3f} mm   (linkage servo tracking {track_err:.2f} deg)",
        ]
        frame = compose_side_by_side(left, right, title_l, title_r, footer, footer_font=20)
        if writer is None:
            writer = FFmpegWriter(out, (frame.shape[1], frame.shape[0]), fps=args.fps)
        writer.write(frame)
        if k % (5 * args.fps) == 0:
            print(f"  frame {k}/{n_frames}  t={t:.1f}s  tip err {tip_err.max():.3f} mm", flush=True)
    writer.close()
    r_link.close()
    r_ser.close()
    print(f"wrote {out}  ({n_frames} frames, {T_END:.0f} s, {time.time() - t0:.1f}s render time)")
    print(f"max fingertip error over the video: {max_tip_err:.3f} mm; max servo tracking error {max_track_err:.2f} deg")


if __name__ == "__main__":
    main()
