#!/usr/bin/env python
"""Reference trajectories from the original closed-chain hand, standing in for real telemetry.

The upstream linkage model (gravity on, its own joint friction, its own position servos with kp = 50) is driven
with randomized servo-command programs at the control rate. For every control step we record the commanded
servo angles, the servo angles actually reached, the equivalent hinge angles of the simplified hand
(abd, mcp, pip per finger, extracted the same way as in serial_hand) and the fingertip positions.

    python -m diffsim.synth_data --side right --n-train 12 --n-val 4                           # -> diffsim/data/right/
    python -m diffsim.synth_data --side right --servo-kp 25 --motor-friction 0.05 --tag _kp25  # mismatched reference
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from .paths import SERIAL_DIR, linkage_scene_xml

sys.path.insert(0, str(SERIAL_DIR))
from ah_serial.linkage import LinkageHand, MotionSamples  # noqa: E402
from ah_serial.kinematics import SerialHandKinematics  # noqa: E402
from ah_serial.mathutil import mat_to_quat  # noqa: E402

HERE = Path(__file__).resolve().parent
LIMIT = np.radians(80.0)          # stay inside the monotonic servo range (dead centre at ~87 deg)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def command_program(T: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """(T, 8) servo angles: piecewise-smooth hold/move segments plus low-frequency sinusoids, per finger
    random mix of flexion (theta1 = -theta2) and abduction (theta1 = theta2) content."""
    t = np.arange(T) * dt
    cmd = np.zeros((T, 8))
    for n in range(4):
        flex = np.zeros(T)
        abd = np.zeros(T)
        # segments: random targets held, moved with smoothstep transitions
        k = 0
        cur_f, cur_a = 0.0, 0.0
        while k < T:
            dur = int(rng.uniform(0.4, 1.6) / dt)
            move = int(rng.uniform(0.15, 0.6) / dt)
            tgt_f = rng.uniform(-0.75, 1.0) * LIMIT      # more flexion than extension
            tgt_a = rng.uniform(-0.35, 0.35) * LIMIT
            seg = slice(k, min(T, k + dur))
            s = smoothstep(np.arange(seg.stop - seg.start) / max(move, 1))
            flex[seg] = cur_f + s * (tgt_f - cur_f)
            abd[seg] = cur_a + s * (tgt_a - cur_a)
            cur_f, cur_a = tgt_f, tgt_a
            k = seg.stop
        f1, f2 = rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.4)
        flex += 0.12 * LIMIT * np.sin(2 * np.pi * f1 * t + rng.uniform(0, 6.28))
        abd += 0.06 * LIMIT * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 6.28))
        th1, th2 = flex + abd, -flex + abd
        scale = np.maximum(1.0, np.maximum(np.abs(th1), np.abs(th2)) / LIMIT)
        cmd[:, 2 * n] = th1 / scale
        cmd[:, 2 * n + 1] = th2 / scale
    return cmd


def record(H: LinkageHand, K: SerialHandKinematics, cmd: np.ndarray, dt: float, n_sub: int, warm_steps: int = 100):
    m, d = H.model, H.data
    act = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}") for i in (1, 2)]
                    for n in (1, 2, 3, 4)]).reshape(-1)
    qadr = np.array([m.jnt_qposadr[j] for f in H.fingers for j in f.motor_j])
    T = len(cmd)
    servo = np.zeros((T, 8))
    tips = np.zeros((T, 4, 3))
    pose = {k: np.zeros((4, T, 7)) for k in "GPLD"}
    H.reset_zero()
    d.ctrl[act] = cmd[0]
    for _ in range(warm_steps):                     # settle on the first command
        mujoco.mj_step(m, d)
    for k in range(T):
        d.ctrl[act] = cmd[k]
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        servo[k] = d.qpos[qadr]
        for i, f in enumerate(H.fingers):
            tips[k, i] = d.site_xpos[f.tip_s]
            for key, b in (("G", f.G), ("P", f.P), ("L", f.L), ("D", f.D)):
                pose[key][i, k, :3] = d.xpos[b]
                pose[key][i, k, 3:] = d.xquat[b]
    S = MotionSamples(np.zeros((T, 2)), np.ones((4, T), bool), np.zeros((4, T)), np.zeros((4, T)), pose,
                      np.concatenate([np.transpose(tips, (1, 0, 2)), np.tile([1.0, 0, 0, 0], (4, T, 1))], axis=2))
    H.extract_joint_angles(S)
    q = np.transpose(S.q[:, :, :3], (1, 0, 2)).reshape(T, 12)          # [abd, mcp, pip] per finger
    # joint-space equivalent of the servo command (exact closure model), for the PD baseline and as a feature
    cmd_joint = np.zeros((T, 8))
    for k in range(T):
        qj = K.motors_to_joints(cmd[k], exact=True)
        cmd_joint[k] = qj[:, :2].reshape(-1)
    return servo, q, tips, cmd_joint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="right")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.02, help="control period (50 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--servo-kp", type=float, default=None,
                    help="override the servo gain of the reference hand (upstream: 50); a mismatch the simplified model does not know")
    ap.add_argument("--motor-friction", type=float, default=None,
                    help="extra Coulomb friction [N m] on the reference hand's servo joints")
    ap.add_argument("--tag", default="", help="suffix of the output folder, e.g. _kp25")
    args = ap.parse_args()
    H = LinkageHand(linkage_scene_xml(args.side))
    if args.servo_kp is not None:
        for n in (1, 2, 3, 4):
            for i in (1, 2):
                a = mujoco.mj_name2id(H.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}")
                H.model.actuator_gainprm[a, 0] = args.servo_kp
                H.model.actuator_biasprm[a, 1] = -args.servo_kp
    if args.motor_friction is not None:
        for f in H.fingers:
            for j in f.motor_j:
                H.model.dof_frictionloss[H.model.jnt_dofadr[j]] += args.motor_friction
    n_sub = int(round(args.dt / H.model.opt.timestep))
    K = SerialHandKinematics.from_json(SERIAL_DIR / "params" / f"identified_{args.side}.json")
    out = HERE / "data" / (args.side + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    T = int(round(args.seconds / args.dt))
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        for i in range(n):
            cmd = command_program(T, args.dt, rng)
            servo, q, tips, cmd_joint = record(H, K, cmd, args.dt, n_sub)
            path = out / f"{split}_{i:03d}.npz"
            np.savez_compressed(path, dt=args.dt, cmd=cmd.astype(np.float32), servo=servo.astype(np.float32),
                                q=q.astype(np.float32), tips=tips.astype(np.float32), cmd_joint=cmd_joint.astype(np.float32))
            err = np.degrees(np.abs(servo - cmd).mean())
            print(f"{path.name}: {T} steps, mean |servo - cmd| {err:.2f} deg, mcp range "
                  f"[{np.degrees(q[:, 1::3].min()):.0f}, {np.degrees(q[:, 1::3].max()):.0f}] deg")


if __name__ == "__main__":
    main()
