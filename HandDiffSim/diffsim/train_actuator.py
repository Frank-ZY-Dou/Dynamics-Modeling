#!/usr/bin/env python
"""Learn a torque surrogate for the simplified hand by backpropagating pose error through MuJoCo Warp.

Port of the NeuralActuator trainer to the hand: a network maps a window of commands and proprioception to the
eight joint torques, the differentiable backend advances the simplified hand, and the loss is the joint-angle
error against the reference trajectories (here recorded from the original linkage model, later from the real
hand). Curriculum on the rollout length, AdamW with warm-up/cosine schedule, gradient clipping, non-finite
update skipping and EMA follow the reference.

    python -m diffsim.train_actuator --config diffsim/configs/hand_mlp.yaml
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .backend import HandWarpBackend
from .data import F_DIM, Normalizer, feature_stats, gather_batch, load_trajectories, rebuild_features, sample_windows
from .models import create_model
from .servo import driven_joints

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def smooth_l1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


def rollout(model, be, norm, hp, q0, v0, hist0, feat_seq, training: bool):
    """Batched rollout. Returns stacked driven joint angles (K,B,8) and torques (K,B,8)."""
    model.train(training)
    B = q0.shape[0]
    q, v, hist = q0, v0, norm(hist0)
    qs, taus = [], []
    for k in range(feat_seq.shape[0]):
        feat_n = norm(rebuild_features(feat_seq[k], driven_joints(q), driven_joints(v)))
        tau = model(hist.reshape(B, -1), feat_n).clamp(-hp["tau_max"], hp["tau_max"])
        q2, v2 = be.step(q, v, tau)
        q2 = torch.nan_to_num(q2, nan=0.0)
        v2 = torch.nan_to_num(v2.clamp(-hp["qvel_clip"], hp["qvel_clip"]), nan=0.0)
        hist = torch.cat([hist[:, 1:], feat_n[:, None, :]], dim=1)
        qs.append(driven_joints(q2))
        taus.append(tau)
        q, v = q2, v2
    return torch.stack(qs), torch.stack(taus)


def lr_at(count, lr, warm, decay_steps):
    init, peak, end = lr * 0.01, lr, lr * 0.1
    if count < warm:
        return init + (peak - init) * (count / warm)
    t = min((count - warm) / max(decay_steps - warm, 1), 1.0)
    alpha = end / peak
    return peak * (alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * t)))


class Optim:
    """AdamW + global-norm clipping + warm-up/cosine schedule; non-finite gradients skip the update."""

    def __init__(self, model, lr, warm, decay_steps, grad_clip, weight_decay, max_consecutive_errors=50):
        self.model, self.lr, self.warm, self.decay_steps = model, lr, warm, decay_steps
        self.grad_clip, self.max_err = grad_clip, max_consecutive_errors
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        self.count, self.nonfinite_consec = 0, 0

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def step(self):
        params = list(self.model.parameters())
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        finite = all(torch.isfinite(p.grad).all() for p in params)
        applied = finite or self.nonfinite_consec >= self.max_err
        if applied:
            for g in self.opt.param_groups:
                g["lr"] = lr_at(self.count, self.lr, self.warm, self.decay_steps)
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            self.opt.step()
            self.count += 1
        self.nonfinite_consec = 0 if finite else self.nonfinite_consec + 1
        return applied, finite


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None, help="override the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    trajs = load_trajectories(str(ROOT / cfg["data_glob"]))
    mean, std = feature_stats(trajs)
    norm = Normalizer(mean, std, dev)
    B, H = int(cfg["batch_size"]), int(cfg["history_length"])
    epochs = int(args.epochs or cfg["epochs"])
    dt = trajs[0]["dt"]
    n_sub = int(cfg.get("sim_substeps", round(dt / 0.002)))
    hp = dict(tau_max=float(cfg.get("tau_max", 2.0)), qvel_clip=float(cfg.get("qvel_clip", 30.0)),
              w_pos=float(cfg.get("pos_loss_weight", 100.0)), w_tau=float(cfg.get("torque_reg_weight", 0.0)))
    print(f"data: {len(trajs)} trajectories, {sum(len(t['feats']) for t in trajs)} steps, dt={dt}, F={F_DIM}")

    model = create_model(cfg.get("model", "mlp"), F_DIM, H, **cfg.get("model_kwargs", {})).to(dev)
    print(f"model: {cfg.get('model', 'mlp')} with {sum(p.numel() for p in model.parameters()):,} parameters")
    be = HandWarpBackend(B, dt, n_sub, side=cfg.get("side", "right"), device=str(dev))
    print(f"backend: {be.name}, B={B}, n_sub={n_sub}, Jacobian worlds={be.W}")

    lr = float(cfg["lr"])
    warm = max(int(cfg.get("lr_warmup_epochs", 0)), 1)
    optim = Optim(model, lr, warm, max(int(cfg.get("lr_decay_epochs", epochs)), warm + 1),
                  float(cfg.get("grad_clip", 1.0)), float(cfg.get("weight_decay", 1e-4)))
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = {k: p.detach().clone() for k, p in model.state_dict().items()} if ema_decay > 0 else None
    cur_eps, cur_steps = list(cfg.get("curriculum_epochs", [])), list(cfg.get("curriculum_steps", []))

    def steps_for(ep):
        for e, s in zip(cur_eps, cur_steps):
            if ep < e:
                return int(s)
        return int(cfg["rollout_steps"])

    out_dir = ROOT / cfg.get("out_dir", "diffsim/outputs/run")
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.jsonl", "w")

    def save(tag):
        payload = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, feature_mean=mean, feature_std=std,
                       config=cfg, epoch=ep + 1)
        if ema is not None:
            payload["ema_state_dict"] = {k: v.cpu() for k, v in ema.items()}
        torch.save(payload, out_dir / f"ckpt_{tag}.pt")

    t0 = time.time()
    nonfinite = 0
    for ep in range(epochs):
        K = steps_for(ep)
        idx_t, starts = sample_windows(trajs, H, K, B, np_rng)
        q0, v0, hist0, feat_seq, tgt_seq = gather_batch(trajs, idx_t, starts, H, K, dev)
        Q, TAU = rollout(model, be, norm, hp, q0, v0, hist0, feat_seq, training=True)
        pos = smooth_l1(Q - tgt_seq).mean()
        loss = hp["w_pos"] * pos + hp["w_tau"] * (TAU ** 2).mean()
        optim.zero_grad()
        loss.backward()
        applied, finite = optim.step()
        if not finite:
            nonfinite += 1
        if ema is not None:
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    ema[k].mul_(ema_decay).add_(sd[k], alpha=1.0 - ema_decay)
        if ep % 10 == 0 or ep == epochs - 1:
            mae = (Q - tgt_seq).abs().mean(dim=(0, 1)).detach().cpu().numpy() * 180 / np.pi
            rec = dict(epoch=ep, steps=K, loss=float(loss), pos_mae_deg=float(mae.mean()),
                       per_joint_deg=[round(float(x), 3) for x in mae], tau_rms=float(TAU.detach().pow(2).mean().sqrt()),
                       lr=lr_at(optim.count, lr, warm, optim.decay_steps), nonfinite=nonfinite, sec=round(time.time() - t0, 1))
            print(f"ep {ep:4d} K={K:3d} loss={rec['loss']:.4f} pose MAE={rec['pos_mae_deg']:.3f} deg "
                  f"tau_rms={rec['tau_rms']:.3f} Nm lr={rec['lr']:.2e} [{rec['sec']}s]", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if int(cfg.get("ckpt_interval", 0)) and (ep + 1) % int(cfg["ckpt_interval"]) == 0:
            save(f"ep{ep + 1:05d}")
    save("final")
    logf.close()
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
