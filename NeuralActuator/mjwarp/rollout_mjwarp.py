"""Roll out a task with the mjwarp backend (differentiable MuJoCo Warp dynamics) and dump a
trajectory. The predicted joint path sim_q is advanced by MJWarpBackendTorch,
not MJX; the step logic mirrors the public evaluate_actuator.py single_step
(feature update from sim state, torque clip, gripper clamp, qvel clip, history
roll). Dumps npz compatible with the paper renderer and with the mjwarp renderer.

Dynamics-rollout mode: sim_q comes from Newton.
Force-sensor mode (--force_only): no simulator; both panels replay the recorded
motion and only the force arrow (predicted vs ground truth) differs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "newton")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from torch_native.model_torch import TransformerActuatorTorch
from mjwarp_backend_torch import MJWarpBackendTorch
from backends.base import GRIPPER_MIN, GRIPPER_MAX

GRIP_TORQUE_CLIP = 1.5
QVEL_CLIP = 10.0


def load_csv(path):
    df = pd.read_csv(path)
    g = lambda cols, d: (df[cols].values.astype(np.float32)
                         if all(c in df.columns for c in cols) else d)
    n = len(df)
    pos = df[[f"pos{i}" for i in range(1, 6)]].values.astype(np.float32)
    goal = g([f"goal_pos{i}" for i in range(1, 6)], pos.copy())
    aperture = (df["aperture"].values / 1000.0).astype(np.float32) if "aperture" in df else np.zeros(n, np.float32)
    goal_ap = df["goal_aperture"].values.astype(np.float32) if "goal_aperture" in df else (
        df["aperture"].values.astype(np.float32) if "aperture" in df else np.zeros(n, np.float32))
    cur = g([f"current{i}" for i in range(1, 6)], np.zeros((n, 5), np.float32))
    vel = g([f"vel{i}" for i in range(1, 6)], np.zeros((n, 5), np.float32))
    volts = g([f"volts{i}" for i in range(1, 6)], np.zeros((n, 5), np.float32))
    temp = g([f"temp{i}" for i in range(1, 6)], np.zeros((n, 5), np.float32))
    force = np.zeros((n, 3), np.float32)
    if all(c in df.columns for c in ("force_x", "force_y", "force_z")):
        force = df[["force_x", "force_y", "force_z"]].values.astype(np.float32).copy()
        force[force == -999] = 0.0
    return dict(pos=pos, goal=goal, aperture=aperture, goal_ap=goal_ap, cur=cur,
                vel=vel, volts=volts, temp=temp, force=force, n=n)


def csv_features(d):
    n = d["n"]
    f = np.zeros((n, 36), np.float32)
    f[:, 0:5] = d["goal"]
    f[:, 5:9] = d["pos"][:, :4]
    f[:, 9] = d["aperture"] * 1000.0
    f[:, 10:15] = d["cur"]
    f[:, 15:20] = d["vel"]
    f[:, 20:25] = d["volts"]
    f[:, 25:30] = d["temp"]
    f[:, 30] = d["goal_ap"]
    f[:, 31:35] = d["goal"][:, :4] - d["pos"][:, :4]
    f[:, 35] = d["goal_ap"] - d["aperture"] * 1000.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task_name", default="task")
    ap.add_argument("--data_dt", type=float, default=0.017)
    ap.add_argument("--sim_step_size", type=int, default=4)
    ap.add_argument("--history_length", type=int, default=8)
    ap.add_argument("--force_only", action="store_true",
                    help="virtual force sensor: no simulator, replay recorded motion")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = payload.get("ema_state_dict", payload["state_dict"])
    mean = torch.from_numpy(np.asarray(payload["feature_mean"], np.float32)).to(dev)
    std = torch.from_numpy(np.asarray(payload["feature_std"], np.float32)).to(dev)
    model = TransformerActuatorTorch(feature_dim=36, hidden_dim=192, latent_dim=96,
                                     num_heads=4, num_layers=4, d_ff=384).to(dev)
    model.load_state_dict(sd)
    model.eval()

    d = load_csv(args.csv)
    csv_feat = csv_features(d)
    n = d["n"]
    H = args.history_length

    def norm(x):
        return ((x - mean) / std).clamp(-10.0, 10.0)

    be = None if args.force_only else MJWarpBackendTorch(1, args.data_dt, args.sim_step_size,
                                                         device=str(dev))
    # OMX renderer schema: gt_q (T,7) = [pos1-4 rad, pos5, aperture m, aperture m]
    gt_q = np.zeros((n, 7), np.float32)
    gt_q[:, :4] = d["pos"][:, :4]
    gt_q[:, 4] = d["pos"][:, 4]
    gt_q[:, 5] = d["aperture"]
    gt_q[:, 6] = d["aperture"]
    # sim_q (T,5) = [arm1-4 rad, gripper m]
    sim_q = np.zeros((n, 5), np.float32)
    force_pred = np.zeros((n, 3), np.float32)
    gate_rec = np.zeros(n, np.float32)

    q = torch.zeros(1, 6, device=dev)
    q[0, :4] = torch.from_numpy(d["pos"][0, :4]).to(dev)
    q[0, 4] = q[0, 5] = float(d["aperture"][0])
    v = torch.zeros(1, 6, device=dev)
    hist = torch.zeros(1, H, 36, device=dev)

    with torch.no_grad():
        for i in range(n):
            feat = torch.from_numpy(csv_feat[i]).to(dev).clone()[None, :]
            if not args.force_only:
                feat[0, 5:9] = q[0, :4]
                feat[0, 9] = q[0, 4] * 1000.0
                feat[0, 15:20] = v[0, :5]
                feat[0, 31:35] = feat[0, 0:4] - q[0, :4]
                feat[0, 35] = feat[0, 30] - q[0, 4] * 1000.0
            feat_n = norm(feat)
            tau_p, f_final, _, gate, _, _ = model(hist.reshape(1, -1), feat_n)
            force_pred[i] = f_final[0].cpu().numpy()
            gate_rec[i] = float(gate[0, 0])

            if args.force_only:
                sim_q[i] = np.concatenate([gt_q[i, :4], gt_q[i, 5:6]])
            else:
                tau_limit = torch.tensor([5.0, 5.0, 5.0, 5.0, GRIP_TORQUE_CLIP], device=dev)
                tau = torch.minimum(torch.maximum(tau_p, -tau_limit), tau_limit)
                q2, v2 = be.step(q, v, tau, torch.zeros(1, 3, device=dev), route_force=False)
                q2 = torch.cat([q2[:, :4], q2[:, 4:6].clamp(GRIPPER_MIN, GRIPPER_MAX)], dim=1)
                v2 = torch.nan_to_num(v2.clamp(-QVEL_CLIP, QVEL_CLIP), nan=0.0)
                if i < n - 1:
                    q, v = q2, v2
                sim_q[i, :4] = q[0, :4].cpu().numpy()
                sim_q[i, 4] = float(q[0, 4])
            hist = torch.cat([hist[:, 1:], feat_n[None, :] if feat_n.dim() == 1 else feat_n[:, None, :]], dim=1)

    np.savez(args.out, robot="omx", task=args.task_name,
             sim_q=sim_q, gt_q=gt_q, force_pred=force_pred, force_gt=d["force"],
             gate=gate_rec, mode="force_only" if args.force_only else "dynamics")
    err = np.abs(sim_q[:, :4] - gt_q[:, :4]).mean() * 180 / np.pi
    print(f"{args.task_name}: {'force-only' if args.force_only else 'mjwarp dynamics'} "
          f"rollout, {n} steps, arm MAE {err:.3f} deg -> {args.out}")


if __name__ == "__main__":
    main()
