"""Trajectory files -> arrays, features and rollout windows (shared by training and evaluation).

Per step the feature vector is
    [cmd_joint (8), q (8), qd (8), cmd_joint - q (8), cmd_servo (8)]          F = 40
where q and qd are the driven hinges (abd, mcp per finger) and cmd_servo the raw servo command. Inside a
rollout the q, qd and (cmd_joint - q) columns are rebuilt from the simulated state, as in the reference trainer.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

F_DIM = 40
STD_FLOOR = np.array([0.02] * 8 + [0.02] * 8 + [0.2] * 8 + [0.01] * 8 + [0.02] * 8, np.float32)


def load_trajectories(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    out = []
    for f in files:
        z = np.load(f)
        q12 = z["q"].astype(np.float32)                      # (T,12) abd, mcp, pip per finger
        dt = float(z["dt"])
        v12 = np.gradient(q12, dt, axis=0).astype(np.float32)
        q8 = q12.reshape(-1, 4, 3)[:, :, :2].reshape(-1, 8)
        v8 = v12.reshape(-1, 4, 3)[:, :, :2].reshape(-1, 8)
        cj, cs = z["cmd_joint"].astype(np.float32), z["cmd"].astype(np.float32)
        feats = np.concatenate([cj, q8, v8, cj - q8, cs], axis=1).astype(np.float32)
        out.append(dict(name=Path(f).stem, dt=dt, q12=q12, v12=v12, q8=q8, v8=v8, cmd_joint=cj, cmd_servo=cs,
                        feats=feats, tips=z["tips"].astype(np.float32)))
    return out


def feature_stats(trajs: list[dict]):
    allf = np.concatenate([t["feats"] for t in trajs], axis=0)
    mean = allf.mean(axis=0).astype(np.float32)
    std = np.maximum(allf.std(axis=0).astype(np.float32), STD_FLOOR)
    return mean, std


class Normalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std).clamp(-10.0, 10.0)


def rebuild_features(feats: torch.Tensor, q8: torch.Tensor, v8: torch.Tensor) -> torch.Tensor:
    """Replace the state-dependent columns of the recorded features with the simulated state."""
    cj = feats[:, 0:8]
    return torch.cat([cj, q8, v8, cj - q8, feats[:, 32:40]], dim=1)


def sample_windows(trajs: list[dict], history: int, steps: int, batch: int, rng: np.random.Generator):
    """Random (trajectory, start) pairs with `history` steps before and `steps` after."""
    lens = np.array([len(t["feats"]) for t in trajs])
    idx_t = rng.integers(0, len(trajs), batch)
    starts = np.array([rng.integers(history, lens[i] - steps - 1) for i in idx_t])
    return idx_t, starts


def gather_batch(trajs, idx_t, starts, history, steps, device):
    """Tensors for a rollout batch: initial state, normalized-later history, per-step features and targets."""
    B = len(idx_t)
    q0 = np.stack([trajs[i]["q12"][s] for i, s in zip(idx_t, starts)])
    v0 = np.stack([trajs[i]["v12"][s] for i, s in zip(idx_t, starts)])
    hist = np.stack([trajs[i]["feats"][s - history:s] for i, s in zip(idx_t, starts)])            # (B,H,F)
    feat_seq = np.stack([trajs[i]["feats"][s:s + steps] for i, s in zip(idx_t, starts)], axis=1)   # (K,B,F)
    tgt_seq = np.stack([trajs[i]["q8"][s + 1:s + steps + 1] for i, s in zip(idx_t, starts)], axis=1)  # (K,B,8)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return t(q0), t(v0), t(hist), t(feat_seq), t(tgt_seq)
