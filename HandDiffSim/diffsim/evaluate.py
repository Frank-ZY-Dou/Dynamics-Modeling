#!/usr/bin/env python
"""Open-loop evaluation of a trained torque surrogate against the reference trajectories.

For every validation trajectory the simplified hand is rolled out from the recorded initial state for the whole
recording, driven by (a) the learned torque surrogate (torque held over each 20 ms control step) and (b) the
model's own position servos with the upstream gain (kp = 50, updated every 2 ms substep) tracking the
joint-space command. Reports the joint-angle MAE of both against the recording and plots one finger.

    python -m diffsim.evaluate --ckpt diffsim/outputs/mlp/ckpt_final.pt
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .backend import HandWarpBackend  # noqa: E402
from .data import F_DIM, Normalizer, load_trajectories, rebuild_features  # noqa: E402
from .models import create_model  # noqa: E402
from .servo import driven_joints  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEG = 180 / np.pi


@torch.no_grad()
def rollout_full(traj, be, dev, torque_fn, history, norm):
    """Whole-trajectory open-loop rollout; torque_fn(hist_norm, feat_norm, q8, v8, k) -> tau (1,8)."""
    T = len(traj["feats"])
    feats = torch.from_numpy(traj["feats"]).to(dev)
    q = torch.from_numpy(traj["q12"][history:history + 1]).to(dev)
    v = torch.from_numpy(traj["v12"][history:history + 1]).to(dev)
    hist = norm(feats[0:history][None])
    Q = np.zeros((T, 8), np.float32)
    Q[:history + 1] = traj["q8"][:history + 1]
    for k in range(history, T - 1):
        q8, v8 = driven_joints(q), driven_joints(v)
        feat_n = norm(rebuild_features(feats[k:k + 1], q8, v8))
        tau = torque_fn(hist, feat_n, q8, v8, k)
        q, v = be.step(q, v, tau)
        v = v.clamp(-30, 30)
        hist = torch.cat([hist[:, 1:], feat_n[:, None, :]], dim=1)
        Q[k + 1] = driven_joints(q)[0].cpu().numpy()
    return Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-glob", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    H = int(cfg["history_length"])
    trajs = load_trajectories(str(ROOT / (args.val_glob or cfg["val_glob"])))
    norm = Normalizer(payload["feature_mean"], payload["feature_std"], dev)
    model = create_model(cfg.get("model", "mlp"), F_DIM, H, **cfg.get("model_kwargs", {})).to(dev)
    model.load_state_dict(payload.get("ema_state_dict", payload["state_dict"]))
    model.eval()
    dt = trajs[0]["dt"]
    n_sub = int(cfg.get("sim_substeps", round(dt / 0.002)))
    be = HandWarpBackend(1, dt, n_sub, side=cfg.get("side", "right"), device=str(dev))
    be_pos = HandWarpBackend(1, dt, n_sub, side=cfg.get("side", "right"), device=str(dev), actuation="position")
    tau_max = float(cfg.get("tau_max", 2.0))

    def learned(hist, feat_n, q8, v8, k):
        return model(hist.reshape(1, -1), feat_n).clamp(-tau_max, tau_max)

    results = {}
    out_dir = Path(args.ckpt).parent
    for traj in trajs:
        cmd_joint = torch.from_numpy(traj["cmd_joint"]).to(dev)

        def pd(hist, feat_n, q8, v8, k):
            return cmd_joint[k:k + 1]                   # servo targets for the position-mode backend

        Q_pd = rollout_full(traj, be_pos, dev, pd, H, norm)
        Q_nn = rollout_full(traj, be, dev, learned, H, norm)
        ref = traj["q8"]
        sl = slice(H + 1, None)
        mae_pd = np.abs(Q_pd[sl] - ref[sl]).mean(axis=0) * DEG
        mae_nn = np.abs(Q_nn[sl] - ref[sl]).mean(axis=0) * DEG
        results[traj["name"]] = dict(pd_mae_deg=float(mae_pd.mean()), learned_mae_deg=float(mae_nn.mean()),
                                     pd_per_joint=mae_pd.round(3).tolist(), learned_per_joint=mae_nn.round(3).tolist())
        print(f"{traj['name']}: joint MAE  position servo {mae_pd.mean():.3f} deg  |  learned surrogate {mae_nn.mean():.3f} deg")
        if traj is trajs[0]:
            t = np.arange(len(ref)) * dt
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            for ax, j, name in ((axes[0], 1, "finger 1 mcp (knuckle)"), (axes[1], 0, "finger 1 abd (sideways)")):
                ax.plot(t, ref[:, j] * DEG, color="#0b0b0b", lw=1.6, label="reference (linkage model)")
                ax.plot(t, Q_pd[:, j] * DEG, color="#eb6834", lw=1.2, label="simplified hand + its position servos (kp = 50)")
                ax.plot(t, Q_nn[:, j] * DEG, color="#2a78d6", lw=1.2, label="simplified hand + learned torque surrogate")
                ax.set_ylabel(f"{name} [deg]")
                ax.grid(True, color="#e1e0d9")
            axes[0].legend(loc="upper right", fontsize=8, frameon=False)
            axes[1].set_xlabel("time [s]")
            axes[0].set_title(f"Open-loop rollout on {traj['name']}: joint MAE servo {mae_pd.mean():.2f}° vs learned {mae_nn.mean():.2f}°", loc="left", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / "eval_rollout.png", dpi=140)
            plt.close(fig)
    summary = dict(pd_mae_deg=float(np.mean([r["pd_mae_deg"] for r in results.values()])),
                   learned_mae_deg=float(np.mean([r["learned_mae_deg"] for r in results.values()])), per_trajectory=results)
    with open(out_dir / "eval.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"mean over {len(trajs)} validation trajectories: position servo {summary['pd_mae_deg']:.3f} deg, "
          f"learned surrogate {summary['learned_mae_deg']:.3f} deg -> {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
