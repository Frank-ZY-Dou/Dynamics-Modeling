#!/usr/bin/env python
"""Headless smoke test for the AmazingHand MuJoCo + mink simulation.

No display / dora needed.  It reproduces what Demo/AHSimulation/AHSimulation/mj_mink_right.py
does (same mink tasks, same "quat" control mode) and drives the fingertip orientation
targets with the same formulas as Demo/AHSimulation/examples/finger_angle_control.py.
Frames are rendered offscreen with EGL (MUJOCO_GL=egl) and written as PNG.

Usage (from the project root, with AmazingHand/Demo/.venv activated):
    python tools/check_sim_headless.py --side both --seconds 1.0 --out tools/renders
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import mink  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO = PROJECT_ROOT / "AmazingHand"
SCENE = {
    "right": REPO / "Demo/AHSimulation/AHSimulation/AH_Right/mjcf/scene.xml",
    "left": REPO / "Demo/AHSimulation/AHSimulation/AH_Left/mjcf/scene.xml",
}
MOTOR_JOINTS = [f"finger{f}_motor{m}" for f in range(1, 5) for m in (1, 2)]


def build_tasks(model, mode="quat"):
    """Same task set as Client.__init__ in mj_mink_right.py."""
    posture = mink.PostureTask(model, cost=1e-2)
    pos_cost, ori_cost = (1.0, 0.0) if mode == "pos" else (0.0, 1.0)
    tips = [
        mink.FrameTask(frame_name=f"tip{i}", frame_type="site",
                       position_cost=pos_cost, orientation_cost=ori_cost, lm_damping=1.0)
        for i in range(1, 5)
    ]
    eq_task = mink.EqualityConstraintTask(model, cost=1000.0)
    return posture, tips, [eq_task, posture, *tips]


def targets_at(t, side):
    """Fingertip orientation targets (w,x,y,z) exactly as finger_angle_control.py."""
    sgn = 1.0 if side == "right" else -1.0
    s1_pitch = np.sin(2 * np.pi * t) * np.radians(10.0) + np.radians(10.0)
    s1_roll = np.cos(2 * np.pi * t) * np.radians(10.0) * sgn
    s2_pitch = np.sin(2 * np.pi * t) * np.radians(140.0 / 2) + np.radians(140.0 / 2)
    s4_pitch = np.sin(2 * np.pi * t) * np.radians((90 + 53) / 2) + np.radians((90 - 53) / 2)
    q = lambda r: r.as_quat(scalar_first=True)  # noqa: E731
    return [
        q(Rotation.from_euler("XYZ", [s1_roll, s1_pitch, 0.0])),
        q(Rotation.from_euler("XYZ", [np.radians(10.0) * sgn, s2_pitch, 0.0])),
        q(Rotation.from_euler("XYZ", [np.radians(20.0) * sgn, s2_pitch, 0.0])),
        q(Rotation.from_euler("xyz", [0.0, -s4_pitch, np.radians(20.0) * sgn])),
    ]


def loop_closure_error(model, data):
    """Max distance (m) between the two sites of every <connect> equality = linkage closure error."""
    worst = 0.0
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        if model.eq_objtype[i] == mujoco.mjtObj.mjOBJ_SITE:
            p1 = data.site_xpos[model.eq_obj1id[i]]
            p2 = data.site_xpos[model.eq_obj2id[i]]
        else:  # body-anchored connect
            b1, b2 = model.eq_obj1id[i], model.eq_obj2id[i]
            p1 = data.xpos[b1] + data.xmat[b1].reshape(3, 3) @ model.eq_data[i, 0:3]
            p2 = data.xpos[b2] + data.xmat[b2].reshape(3, 3) @ model.eq_data[i, 3:6]
        worst = max(worst, float(np.linalg.norm(p1 - p2)))
    return worst


def make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.03, 0.0, 0.10]
    cam.distance = 0.32
    cam.azimuth = 160
    cam.elevation = -20
    return cam


def save_png(path, rgb):
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image
        Image.fromarray(rgb).save(path)


def run_side(side, seconds, out_dir, render):
    scene = SCENE[side]
    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(str(scene))
    print(f"[{side}] loaded {scene.relative_to(PROJECT_ROOT)} in {time.time() - t0:.2f}s")
    print(f"[{side}] nq={model.nq} nv={model.nv} nu={model.nu} neq={model.neq} "
          f"nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh}")
    n_connect = sum(int(model.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT) for i in range(model.neq))
    print(f"[{side}] closed-loop <connect> constraints: {n_connect}  (linkage loops)")

    configuration = mink.Configuration(model)
    posture, tips, tasks = build_tasks(model, "quat")
    model, data = configuration.model, configuration.data

    configuration.update_from_keyframe("zero")
    posture.set_target_from_configuration(configuration)
    for i in range(1, 5):
        mink.move_mocap_to_frame(model, data, f"finger{i}_target", f"tip{i}", "site")
    mocap_ids = [model.body(f"finger{i}_target").mocapid[0] for i in range(1, 5)]
    motor_qadr = [model.joint(n).qposadr[0] for n in MOTOR_JOINTS]

    renderer = cam = None
    if render:
        renderer = mujoco.Renderer(model, height=480, width=480)
        cam = make_camera()

    dt = 1.0 / 1000.0  # RateLimiter(frequency=1000.0) in the repo node
    n_steps = int(round(seconds / dt))
    sample_times = [0.0, 0.25, 0.5, 0.75]  # open / fully flexed / mid / extended
    frames, rows = [], []
    solve_times = []
    print(f"[{side}] {'t[s]':>5} | {'closure err [mm]':>16} | motor angles [deg] (f1m1 f1m2 f2m1 f2m2 f3m1 f3m2 f4m1 f4m2)")
    for k in range(n_steps + 1):
        t = k * dt
        for q, mid in zip(targets_at(t, side), mocap_ids):
            data.mocap_quat[mid] = q
        for i, task in enumerate(tips, start=1):
            task.set_target(mink.SE3.from_mocap_name(model, data, f"finger{i}_target"))
        ts = time.perf_counter()
        vel = mink.solve_ik(configuration, tasks, dt, "quadprog", 1e-5)
        configuration.integrate_inplace(vel, dt)
        solve_times.append(time.perf_counter() - ts)

        if any(abs(t - s) < dt / 2 for s in sample_times):
            err_mm = 1e3 * loop_closure_error(model, data)
            motors = np.degrees(data.qpos[motor_qadr])
            rows.append((t, err_mm, motors))
            print(f"[{side}] {t:5.2f} | {err_mm:16.3f} | " + " ".join(f"{a:7.1f}" for a in motors))
            if renderer is not None:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

    st = np.array(solve_times) * 1e3
    print(f"[{side}] IK solve time: mean {st.mean():.2f} ms, max {st.max():.2f} ms over {len(st)} steps")
    motors_all = np.array([r[2] for r in rows])
    print(f"[{side}] motor angle span over the cycle: min {motors_all.min():.1f} deg, max {motors_all.max():.1f} deg "
          f"(joint range is +-90 deg)")
    worst = max(r[1] for r in rows)
    ok = worst < 1.0
    print(f"[{side}] worst loop-closure error {worst:.3f} mm -> {'OK' if ok else 'TOO LARGE'}")

    if frames:
        strip = np.concatenate(frames, axis=1)
        png = Path(out_dir) / f"{side}_hand_ik_frames.png"
        save_png(png, strip)
        print(f"[{side}] wrote {png}  (t = {', '.join(f'{s:.2f}s' for s in sample_times)} left->right)")
    if renderer is not None:
        renderer.close()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["right", "left", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=1.0, help="simulated seconds (targets are a 1 Hz cycle)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "tools" / "renders"))
    ap.add_argument("--no-render", action="store_true", help="skip offscreen rendering")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"mujoco {mujoco.__version__}, mink {getattr(mink, '__version__', '?')}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    sides = ["right", "left"] if args.side == "both" else [args.side]
    results = {s: run_side(s, args.seconds, args.out, not args.no_render) for s in sides}
    print("SUMMARY:", ", ".join(f"{s}={'OK' if ok else 'FAIL'}" for s, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
