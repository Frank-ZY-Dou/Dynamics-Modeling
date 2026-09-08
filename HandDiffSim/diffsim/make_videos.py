#!/usr/bin/env python
"""Videos of the differentiable-simulation results.

    python -m diffsim.make_videos ik        # IK through the simulator: the optimization converging, then the
                                            # optimized torques moving the hand from rest to the targets
    python -m diffsim.make_videos rollout   # left: original linkage replaying a held-out recording,
                                            # right: simplified hand driven open loop by the learned surrogate
    python -m diffsim.make_videos track     # reference motion | kinematic-IK commands | commands optimized
                                            # through the simulator (examples/track_reference.py)
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from .backend import HandWarpBackend
from .data import Normalizer, load_trajectories, rebuild_features
from .kinematics_torch import HandKinematicsTorch
from .models import create_model
from .paths import SERIAL_DIR, linkage_scene_xml
from .servo import driven_joints, with_pip

sys.path.insert(0, str(SERIAL_DIR))
from ah_serial.linkage import LinkageHand  # noqa: E402
from ah_serial.render import FFmpegWriter, Offscreen, compose_side_by_side, make_camera  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEG = 180 / np.pi
CAM = dict(lookat=(0.03, 0.0, 0.10), distance=0.30, azimuth=160, elevation=-20)


def serial_scene():
    mjm = mujoco.MjModel.from_xml_path(str(SERIAL_DIR / "models" / "AH_Right" / "scene.xml"))
    return mjm, mujoco.MjData(mjm)


def video_ik(args):
    from .examples.ik_through_sim import Q_TRUE_DEG, motion_under, solve_torque, targets_from
    dev = torch.device(args.device)
    fk = HandKinematicsTorch("right", device=dev)
    targets, _ = targets_from(fk, Q_TRUE_DEG, dev)
    be = HandWarpBackend(args.starts, 0.02, 10, device=str(dev), actuation="torque")
    trace = []
    print("solving (torque mode, tracing the best start) ...", flush=True)
    q_end, tips, hist, best_tau, conv = solve_torque(be, fk, targets, dev, trace=trace)
    err_final = float(((tips - targets).norm(dim=-1) * 1e3).mean())
    motion = motion_under(be, best_tau, best_tau.shape[0])                 # (K+1, 12)
    print(f"final mean fingertip error {err_final:.3f} mm; {len(trace)} iterations traced", flush=True)

    mjm, d = serial_scene()
    for n in range(4):
        b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"finger{n + 1}_target")
        d.mocap_pos[mjm.body_mocapid[b]] = targets[0, n].cpu().numpy()
    r = Offscreen(mjm, (args.size, args.size), make_camera(**CAM))
    out = HERE / "video" / "ik_through_sim.mp4"
    writer = None
    fps = args.fps

    def emit(q12, lines):
        nonlocal writer
        d.qpos[:] = q12
        mujoco.mj_forward(mjm, d)
        frame = compose_side_by_side(r.render(d), r.render(d), "", "", lines, header_h=0, footer_font=20)
        # one panel only: crop the duplicated right half
        frame = frame[:, : args.size]
        frame = np.ascontiguousarray(frame)
        if writer is None:
            writer = FFmpegWriter(out, (frame.shape[1], frame.shape[0]), fps=fps)
        writer.write(frame)

    # part 1: the optimization, pose at the end of the horizon of the best start, every `stride` iterations
    stride = max(1, len(trace) // (fps * args.opt_seconds))
    for it in range(0, len(trace), stride):
        q12, e = trace[it]
        emit(q12, [f"IK through the simulator, torque mode, {args.starts} starts",
                   f"iteration {it + 1:3d}/{len(trace)}   mean tip error {e:7.3f} mm"])
    for _ in range(fps):                                                   # hold the converged pose for 1 s
        emit(trace[-1][0], [f"IK through the simulator, torque mode, {args.starts} starts",
                            f"iteration {len(trace):3d}/{len(trace)}   mean tip error {trace[-1][1]:7.3f} mm"])
    # part 2: the optimized torque sequence played from rest, slowed down
    rep = max(1, int(round(fps * args.motion_seconds / len(motion))))
    for k, q12 in enumerate(motion):
        for _ in range(rep):
            emit(q12, [f"optimized torques from rest, {rep * 50 // fps:d}x slower",
                       f"t = {k * 0.02:5.2f} s   final mean tip error {err_final:.2f} mm"])
    for _ in range(fps):
        emit(motion[-1], [f"optimized torques from rest, {rep * 50 // fps:d}x slower",
                          f"t = {(len(motion) - 1) * 0.02:5.2f} s   final mean tip error {err_final:.2f} mm"])
    writer.close()
    r.close()
    print(f"wrote {out}")


def video_rollout(args):
    dev = torch.device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    H = int(cfg["history_length"])
    traj = load_trajectories(str(ROOT / cfg["val_glob"]))[args.traj_index]
    norm = Normalizer(payload["feature_mean"], payload["feature_std"], dev)
    model = create_model(cfg.get("model", "mlp"), traj["feats"].shape[1], H, **cfg.get("model_kwargs", {})).to(dev)
    model.load_state_dict(payload.get("ema_state_dict", payload["state_dict"]))
    model.eval()
    dt = traj["dt"]
    n_sub = int(cfg.get("sim_substeps", round(dt / 0.002)))
    be = HandWarpBackend(1, dt, n_sub, side=cfg.get("side", "right"), device=str(dev))
    fk = HandKinematicsTorch("right", device=dev)
    tau_max = float(cfg.get("tau_max", 2.0))

    # surrogate rollout, open loop over the whole recording (same loop as evaluate.rollout_full)
    T = len(traj["feats"])
    feats = torch.from_numpy(traj["feats"]).to(dev)
    q = torch.from_numpy(traj["q12"][H:H + 1]).to(dev)
    v = torch.from_numpy(traj["v12"][H:H + 1]).to(dev)
    hist = norm(feats[0:H][None])
    Q12 = np.zeros((T, 12), np.float32)
    Q12[:H + 1] = traj["q12"][:H + 1]
    with torch.no_grad():
        for k in range(H, T - 1):
            q8, v8 = driven_joints(q), driven_joints(v)
            feat_n = norm(rebuild_features(feats[k:k + 1], q8, v8))
            tau = model(hist.reshape(1, -1), feat_n).clamp(-tau_max, tau_max)
            q, v = be.step(q, v, tau)
            v = v.clamp(-30, 30)
            hist = torch.cat([hist[:, 1:], feat_n[:, None, :]], dim=1)
            Q12[k + 1] = q[0].cpu().numpy()
    err = np.abs(Q12.reshape(T, 4, 3)[:, :, :2] - traj["q8"].reshape(T, 4, 2)) * DEG      # (T,4,2)

    # the reference: the original linkage replaying the recorded commands with the settings of the recording
    Hl = LinkageHand(linkage_scene_xml("right"))
    ml, dl = Hl.model, Hl.data
    if args.servo_kp is not None:
        for n in (1, 2, 3, 4):
            for i in (1, 2):
                a = mujoco.mj_name2id(ml, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}")
                ml.actuator_gainprm[a, 0] = args.servo_kp
                ml.actuator_biasprm[a, 1] = -args.servo_kp
    if args.motor_friction is not None:
        for f in Hl.fingers:
            for j in f.motor_j:
                ml.dof_frictionloss[ml.jnt_dofadr[j]] += args.motor_friction
    act = np.array([[mujoco.mj_name2id(ml, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}") for i in (1, 2)]
                    for n in (1, 2, 3, 4)]).reshape(-1)
    cmd = traj["cmd_servo"]
    Hl.reset_zero()
    dl.ctrl[act] = cmd[0]
    for _ in range(100):
        mujoco.mj_step(ml, dl)

    mjs, ds = serial_scene()
    ds.mocap_pos[:, 2] = -1.0                       # the fingertip target markers are meaningless here: hide them
    dl.mocap_pos[:, 2] = -1.0
    r_l = Offscreen(ml, (args.size, args.size), make_camera(**CAM))
    r_s = Offscreen(mjs, (args.size, args.size), make_camera(**CAM))
    out = HERE / "video" / f"surrogate_rollout_{Path(args.ckpt).parent.name}.mp4"
    writer = None
    every = max(1, int(round(1.0 / (dt * args.fps))))
    label = "original linkage" + (f", servo kp {args.servo_kp:g} + friction" if args.servo_kp else "")
    for k in range(T):
        dl.ctrl[act] = cmd[k]
        for _ in range(n_sub):
            mujoco.mj_step(ml, dl)
        if k % every:
            continue
        ds.qpos[:] = Q12[k]
        mujoco.mj_forward(mjs, ds)
        mae_so_far = err[H + 1:k + 1].mean() if k > H else 0.0
        lines = [f"t = {k * dt:5.2f} s   {traj['name']}   open loop from the recorded initial state",
                 f"surrogate joint error so far {mae_so_far:5.2f} deg   |   physics-only servo baseline over the clip {args.baseline_mae:.2f} deg"]
        frame = compose_side_by_side(r_l.render(dl), r_s.render(ds), label,
                                     f"simplified hand + learned surrogate ({cfg.get('model', 'mlp')})", lines,
                                     footer_font=20, title_font=24)
        if writer is None:
            writer = FFmpegWriter(out, (frame.shape[1], frame.shape[0]), fps=args.fps)
        writer.write(frame)
    writer.close()
    r_l.close()
    r_s.close()
    print(f"wrote {out}; surrogate joint MAE over the clip {err[H + 1:].mean():.3f} deg")


def compose_panels(images, titles, footer_lines, header_h=56, title_font=22, footer_font=20):
    """n panels side by side with titles above and status lines below (generalizes compose_side_by_side)."""
    from PIL import Image, ImageDraw
    from ah_serial.render import draw_text, text_width
    h, w = images[0].shape[:2]
    n = len(images)
    footer_h = int(16 + footer_font * 1.35 * max(1, len(footer_lines)) + 10)
    canvas = Image.new("RGB", (n * w, h + header_h + footer_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)
    for i, (img, title) in enumerate(zip(images, titles)):
        canvas.paste(Image.fromarray(img), (i * w, header_h))
        tw = text_width(draw, title, title_font)
        draw_text(draw, (i * w + (w - tw) / 2, (header_h - title_font) / 2), title, title_font)
        if i:
            draw.line([(i * w, 0), (i * w, header_h + h)], fill=(70, 74, 84), width=2)
    y = header_h + h + 10
    for line in footer_lines:
        draw_text(draw, (18, y), line, footer_font, fill=(220, 220, 220), latin="mono")
        y += int(footer_font * 1.35)
    return np.asarray(canvas)


def video_track(args):
    """Reference motion (original linkage) | simplified hand under kinematic-IK commands | under optimized commands."""
    tr = np.load(args.npz or (HERE / "outputs" / "track" / "track_commands.npz"))
    rec = np.load(str(tr["data"]))
    dt, N = float(tr["dt"]), int(tr["steps"])
    idx, speed = tr["idx"], int(tr["speed"])                 # recording frames used as reference, playback factor
    q_kin, q_opt, tips_ref = tr["q_kin"], tr["q_opt"], tr["tips_ref"]
    err_kin, err_opt = tr["err_kin"].mean(axis=1), tr["err_opt"].mean(axis=1)
    cmd = rec["cmd"]
    n_sub = int(round(dt / 0.002))

    # the reference panel replays the recording itself (nominal linkage, its own servos), advanced to the
    # recording frame each reference step uses, so it stays in sync with the sped-up reference
    Hl = LinkageHand(linkage_scene_xml("right"))
    ml, dl = Hl.model, Hl.data
    act = np.array([[mujoco.mj_name2id(ml, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}") for i in (1, 2)]
                    for n in (1, 2, 3, 4)]).reshape(-1)
    Hl.reset_zero()
    dl.ctrl[act] = cmd[0]
    for _ in range(100):
        mujoco.mj_step(ml, dl)
    rec_k = 0

    def replay_to(frame):
        nonlocal rec_k
        while rec_k < frame:
            dl.ctrl[act] = cmd[rec_k]
            for _ in range(n_sub):
                mujoco.mj_step(ml, dl)
            rec_k += 1

    dl.mocap_pos[:, 2] = -1.0
    mjs, ds = serial_scene()
    mjs2, ds2 = serial_scene()
    mocap = [mjs.body_mocapid[mujoco.mj_name2id(mjs, mujoco.mjtObj.mjOBJ_BODY, f"finger{n + 1}_target")] for n in range(4)]
    r_l = Offscreen(ml, (args.size, args.size), make_camera(**CAM))
    r_a = Offscreen(mjs, (args.size, args.size), make_camera(**CAM))
    r_b = Offscreen(mjs2, (args.size, args.size), make_camera(**CAM))
    out = HERE / "video" / (args.out or "track_reference.mp4")
    writer = None
    every = max(1, int(round(1.0 / (dt * args.fps))))
    titles = [f"reference motion ({speed}x recording)", "kinematic IK commands", "optimized through the simulator"]
    for k in range(N):
        replay_to(int(idx[k]))
        if k % every:
            continue
        for d_, q_ in ((ds, q_kin[k]), (ds2, q_opt[k])):
            d_.qpos[:] = q_
            for n in range(4):
                d_.mocap_pos[mocap[n]] = tips_ref[k, n]
        mujoco.mj_forward(mjs, ds)
        mujoco.mj_forward(mjs2, ds2)
        lines = [f"t = {k * dt:5.2f} s   red spheres: reference fingertip positions at this instant",
                 f"fingertip tracking error, mean so far:   kinematic IK {err_kin[:k + 1].mean():5.2f} mm   |   optimized {err_opt[:k + 1].mean():5.2f} mm"]
        frame = compose_panels([r_l.render(dl), r_a.render(ds), r_b.render(ds2)], titles, lines)
        if writer is None:
            writer = FFmpegWriter(out, (frame.shape[1], frame.shape[0]), fps=args.fps)
        writer.write(frame)
    writer.close()
    for r in (r_l, r_a, r_b):
        r.close()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    a = sub.add_parser("ik")
    a.add_argument("--starts", type=int, default=12)
    a.add_argument("--fps", type=int, default=30)
    a.add_argument("--size", type=int, default=640)
    a.add_argument("--opt-seconds", type=float, default=6.0, help="length of the optimization part")
    a.add_argument("--motion-seconds", type=float, default=4.0, help="length of the slowed-down motion part")
    a.add_argument("--device", default="cuda")
    b = sub.add_parser("rollout")
    b.add_argument("--ckpt", default=str(HERE / "outputs" / "transformer_kp25" / "ckpt_final.pt"))
    b.add_argument("--traj-index", type=int, default=0)
    b.add_argument("--servo-kp", type=float, default=25.0, help="servo gain used when the recording was made")
    b.add_argument("--motor-friction", type=float, default=0.05)
    b.add_argument("--baseline-mae", type=float, default=0.625, help="physics-only baseline MAE, from evaluate.py")
    b.add_argument("--fps", type=int, default=25)
    b.add_argument("--size", type=int, default=640)
    b.add_argument("--device", default="cuda")
    c = sub.add_parser("track")
    c.add_argument("--npz", default=None, help="result file of examples/track_reference.py (default: outputs/track)")
    c.add_argument("--out", default=None, help="file name under diffsim/video (default: track_reference.mp4)")
    c.add_argument("--fps", type=int, default=25)
    c.add_argument("--size", type=int, default=480)
    args = ap.parse_args()
    (HERE / "video").mkdir(exist_ok=True)
    {"ik": video_ik, "rollout": video_rollout, "track": video_track}[args.what](args)


if __name__ == "__main__":
    main()
