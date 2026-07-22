"""OMX actuator training in PyTorch + Newton, no JAX in the training loop.

Port of train_newton_implicit.py (itself a port of the public
train_actuator_diffsim.py); same network, data pipeline, losses, curriculum.

optax mapping:
- warmup_cosine_decay_schedule -> lr_at()
- chain(clip_by_global_norm, adamw) + apply_if_finite -> OptaxSemantics
- batch sampling: public sample_valid_indices with the same numpy seed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["MUJOCO_GL"] = "egl"          # force EGL even if MUJOCO_GL was already set
os.environ["PYOPENGL_PLATFORM"] = "egl"

import yaml
import mujoco

from public_import import load_dataset, sample_valid_indices, validate_mujoco_joint_limits
from backends.base import GRIPPER_MIN, GRIPPER_MAX, ROBOT_XML
from torch_native.model_torch import TransformerActuatorTorch, load_flax_params
from torch_native.newton_backend_torch import NewtonBackendTorch


def smooth_l1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


def build_feat(feats, q, v):
    """Rebuild the sim-state-dependent feature columns. Layout: [0:5) tgt,
    [5:9) arm q, 9 aperture(mm), [10:15) pass, [15:20) vel, [20:31) pass,
    [31:35) tgt-q, 35 tgt_ap-ap."""
    ap_mm = q[:, 4] * 1000.0
    return torch.cat([
        feats[:, 0:5],
        q[:, :4],
        ap_mm[:, None],
        feats[:, 10:15],
        v[:, :5],
        feats[:, 20:31],
        feats[:, 0:4] - q[:, :4],
        (feats[:, 30] - ap_mm)[:, None],
    ], dim=1)


def rollout_loss(model, sim_step, norm, hp, q0, v0, hist0,
                 tgt_seq, feat_seq, gtf_seq, training: bool):
    """Batched rollout + losses (port of train_newton_implicit.py loss_fn).
    Sequences are (steps, B, ...). Returns (total_loss, aux dict of tensors)."""
    B = q0.shape[0]
    dev = q0.device
    tau_limit = torch.tensor([5.0, 5.0, 5.0, 5.0, hp["gripper_torque_clip"]], device=dev)
    zeros_f = torch.zeros(B, 3, device=dev)
    model.train(training)

    q, v, hist = q0, v0, hist0
    qs, ffs, gates = [], [], []
    for k in range(tgt_seq.shape[0]):
        tgt, feats = tgt_seq[k], feat_seq[k]

        feat_n = norm(build_feat(feats, q, v))
        tau_p, f_final, _f_raw, gate, _cond, _ = model(hist.reshape(B, -1), feat_n)
        tau = torch.minimum(torch.maximum(tau_p, -tau_limit), tau_limit)

        xfrc = f_final if hp["explicit"] else zeros_f
        q2, v2 = sim_step(q, v, tau, xfrc)

        # state surgery (train_newton_implicit.py:151-159)
        q2 = torch.cat([q2[:, :4], q2[:, 4:6].clamp(GRIPPER_MIN, GRIPPER_MAX)], dim=1)
        q_safe = torch.nan_to_num(q2, nan=0.0)
        nan_mask = torch.isnan(q2[:, :5])
        q2 = torch.cat([torch.where(nan_mask, tgt, q_safe[:, :5]), q_safe[:, 5:6]], dim=1)
        v2 = torch.nan_to_num(v2.clamp(-hp["qvel_clip"], hp["qvel_clip"]), nan=0.0,
                              posinf=hp["qvel_clip"], neginf=-hp["qvel_clip"])

        hist = torch.cat([hist[:, 1:], feat_n[:, None, :]], dim=1)
        qs.append(q2); ffs.append(f_final); gates.append(gate)
        q, v = q2, v2

    # losses over the stacked rollout (train_newton_implicit.py:163-186),
    # with the per-step reductions collapsed into one (K,B,...) pass
    Q = torch.stack(qs)                                  # (K,B,6)
    Ff = torch.stack(ffs)                                # (K,B,3)
    G = torch.stack(gates)                               # (K,B,1)
    arm = smooth_l1(Q[:, :, :4] - tgt_seq[:, :, :4]).mean()
    grip = smooth_l1((Q[:, :, 4] - tgt_seq[:, :, 4]) * 1000.0).mean()
    fmag = (gtf_seq ** 2).sum(dim=2).sqrt()
    has_f = (fmag > 0.01).float()
    focal = has_f * (hp["force_focal"] - 1.0) + 1.0
    force = (smooth_l1(Ff - gtf_seq).mean(dim=2) * focal).mean()
    gate_p = G[:, :, 0].clamp(1e-7, 1 - 1e-7)
    gate_e = (-(has_f * gate_p.log() + (1 - has_f) * (1 - gate_p).log())).mean()
    pj = (Q[:, :, :5] - tgt_seq).abs()                   # (K,B,5)
    total = (hp["w_pos"] * (arm + hp["w_grip"] * grip)
             + hp["w_force"] * force + hp["w_gate"] * gate_e)
    aux = dict(arm=arm, grip=grip, force=force, gate=gate_e,
               pos_mae=pj[:, :, :4].mean(), force_mae=(Ff - gtf_seq).abs().mean(),
               per_joint=pj.mean(dim=(0, 1)))
    return total, aux


def lr_at(count, lr, warm, decay_steps):
    """`count` is the number of previously applied updates (optax's
    scale_by_schedule count), not the epoch."""
    init, peak, end = lr * 0.01, lr, lr * 0.1
    if count < warm:
        return init + (peak - init) * (count / warm)
    t = min((count - warm) / max(decay_steps - warm, 1), 1.0)
    alpha = end / peak
    return peak * (alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * t)))


class OptaxSemantics:
    """torch stand-in for optax.apply_if_finite(chain(clip_by_global_norm, adamw)).
    Nonfinite grads skip the update and freeze the schedule count; past
    max_consecutive_errors consecutive skips the update goes through anyway.
    None grads get zeroed first so weight decay still reaches unused heads.
    """

    def __init__(self, model, lr, warm, decay_steps, grad_clip, weight_decay,
                 max_consecutive_errors=200):
        self.model = model
        self.lr, self.warm, self.decay_steps = lr, warm, decay_steps
        self.grad_clip = grad_clip
        self.max_err = max_consecutive_errors
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=weight_decay)
        self.sched_count = 0
        self.nonfinite_consec = 0

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def step(self):
        """Apply-or-reject one update from the grads currently on the model.
        Returns (applied, finite)."""
        params = list(self.model.parameters())
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        finite = all(torch.isfinite(p.grad).all() for p in params)
        applied = finite or self.nonfinite_consec >= self.max_err
        if applied:
            for g in self.opt.param_groups:
                g["lr"] = lr_at(self.sched_count, self.lr, self.warm, self.decay_steps)
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            self.opt.step()
            self.sched_count += 1
        if finite:
            self.nonfinite_consec = 0
        else:
            self.nonfinite_consec += 1
        return applied, finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log_json", default=None)
    ap.add_argument("--flax_init", action="store_true",
                    help="initialize from the flax init at the same seed (parity runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .pt to resume from"
                         "schedule/rng states for an exact continuation")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    B = int(cfg["batch_size"])
    rollout_steps = int(cfg["rollout_steps"])
    history_length = int(cfg["history_length"])
    data_dt = float(cfg["data_dt"])
    epochs = int(cfg["epochs"])
    hp = dict(
        gripper_torque_clip=float(cfg.get("gripper_torque_clip", 1.5)),
        qvel_clip=float(cfg.get("qvel_clip", 10.0)),
        w_pos=float(cfg["pos_loss_weight"]), w_grip=float(cfg["gripper_loss_weight"]),
        w_force=float(cfg["force_loss_weight"]), w_gate=float(cfg["gate_loss_weight"]),
        force_focal=float(cfg.get("force_focal_weight", 5.0)),
        explicit=bool(cfg.get("apply_external_force", False)),
    )
    init_pos_noise_std = float(cfg.get("init_pos_noise_std", 0.0))

    xml = os.path.join(os.path.dirname(__file__), "..", ROBOT_XML)
    mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(xml))
    result = load_dataset(cfg["datasets"], mj_model, int(cfg.get("downsample_factor", 1)),
                          return_boundaries=True, cfg=cfg)
    data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries = result
    data_values = np.asarray(data_values); q_traj = np.asarray(q_traj)
    v_traj = np.asarray(v_traj); gt_pos = np.asarray(gt_pos); gt_force = np.asarray(gt_force)
    validate_mujoco_joint_limits(mj_model, data_values)
    F = data_values.shape[1]
    print(f"data: {len(data_values)} samples, {len(boundaries)} trajs, feature_dim={F}")

    # feature normalization
    feat_mean = data_values.mean(axis=0).astype(np.float32)
    feat_std = data_values.std(axis=0).astype(np.float32)
    std_floor = np.array([0.05] * 5 + [0.05] * 4 + [1.0] + [10.0] * 5 + [0.1] * 5
                         + [0.1] * 5 + [1.0] * 5 + [1.0] + [0.05] * 4 + [1.0], dtype=np.float32)
    feat_std = np.maximum(feat_std, std_floor)
    mean_t = torch.from_numpy(feat_mean).to(dev)
    std_t = torch.from_numpy(feat_std).to(dev)

    def norm(x):
        return ((x - mean_t) / std_t).clamp(-10.0, 10.0)

    model = TransformerActuatorTorch(
        feature_dim=F, hidden_dim=int(cfg["hidden_dim"]), latent_dim=int(cfg["latent_dim"]),
        num_heads=int(cfg["num_heads"]), num_layers=int(cfg["num_layers"]),
        d_ff=int(cfg["d_ff"]), dropout=float(cfg.get("dropout_rate", 0.1)),
        gated=bool(cfg.get("use_gated_attention", True)), n_joints=5,
        pool_type=cfg.get("pool_type", "mean"),
        zero_init_head=bool(cfg.get("zero_init_torque_head", False))).to(dev)
    if args.flax_init:
        # same starting point as the JAX trainer: flax init at this seed, ported
        import jax
        import jax.numpy as jnp
        from public_import import create_model
        fm = create_model(model_type=cfg.get("model_type", "transformer"),
                          hidden_dim=int(cfg["hidden_dim"]), latent_dim=int(cfg["latent_dim"]),
                          dropout_rate=float(cfg.get("dropout_rate", 0.1)),
                          backbone_activation=cfg.get("backbone_activation", "silu"),
                          num_heads=int(cfg["num_heads"]), num_layers=int(cfg["num_layers"]),
                          d_ff=int(cfg["d_ff"]), pool_type=cfg.get("pool_type", "mean"),
                          use_gated_attention=bool(cfg.get("use_gated_attention", True)),
                          zero_init_head=bool(cfg.get("zero_init_torque_head", False)))
        _, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        fparams = fm.init({"params": init_rng, "dropout": init_rng},
                          jnp.ones((1, history_length * F)), jnp.ones((1, F)), None, ts=data_dt)
        load_flax_params(model, jax.device_get(fparams))
        model.to(dev)
        print("initialized from flax init (seed-matched)")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_par:,}")

    be = NewtonBackendTorch(B, data_dt, int(cfg["sim_step_size"]), device=str(dev))
    print(f"backend: newton_torch | coupling: {'EXPLICIT' if hp['explicit'] else 'implicit'}")

    lr = float(cfg["lr"])
    warm = min(max(int(cfg.get("lr_warmup_epochs", 0)), 1), max(epochs // 2, 1))
    decay_steps = max(int(cfg.get("lr_decay_epochs", epochs)), warm + 1)
    optim = OptaxSemantics(model, lr, warm, decay_steps,
                           float(cfg.get("grad_clip", 1.0)),
                           float(cfg.get("weight_decay", 1e-4)))

    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = ({k: p.detach().clone() for k, p in model.state_dict().items()}
           if ema_decay > 0 else None)

    cur_eps = list(cfg.get("curriculum_epochs", []) or [])
    cur_steps = list(cfg.get("curriculum_steps", []) or [])

    def steps_for_epoch(ep):
        for e, st in zip(cur_eps, cur_steps):
            if ep < e:
                return int(st)
        return rollout_steps

    ckpt_dir = cfg.get("ckpt_dir")
    ckpt_interval = int(cfg.get("ckpt_interval", 0))
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    np_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 100003)   # distinct stream, keeps np_rng aligned

    def save_ckpt(tag, ep):
        """Save full training state so a resumed run continues where it left off."""
        payload = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                       feature_mean=feat_mean, feature_std=feat_std, epoch=ep,
                       opt_state=optim.opt.state_dict(),
                       sched_count=optim.sched_count,
                       nonfinite_consec=optim.nonfinite_consec,
                       np_rng_state=np_rng.bit_generator.state,
                       noise_rng_state=noise_rng.bit_generator.state,
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state(dev))
        if ema is not None:
            payload["ema_state_dict"] = {k: v.cpu() for k, v in ema.items()}
        tmp = os.path.join(ckpt_dir, f".ckpt_{tag}.pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, os.path.join(ckpt_dir, f"ckpt_{tag}.pt"))  # atomic

    start_ep = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.to(dev)
        optim.opt.load_state_dict(payload["opt_state"])
        optim.sched_count = payload["sched_count"]
        optim.nonfinite_consec = payload["nonfinite_consec"]
        np_rng.bit_generator.state = payload["np_rng_state"]
        noise_rng.bit_generator.state = payload["noise_rng_state"]
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"], dev)
        if ema is not None:
            assert "ema_state_dict" in payload, "resume ckpt lacks EMA state"
            for k in ema:
                ema[k].copy_(payload["ema_state_dict"][k].to(dev))
        start_ep = payload["epoch"]
        print(f"resumed from {args.resume} at epoch {start_ep}")

    logf = open(args.log_json, "a" if args.resume else "w") if args.log_json else None
    nonfinite = 0
    t0 = time.time()
    for ep in range(start_ep, epochs):
        cur = steps_for_epoch(ep)
        starts, traj_starts = sample_valid_indices(boundaries, history_length, cur, B, rng=np_rng)
        starts = np.asarray(starts); traj_starts = np.asarray(traj_starts)

        q0 = np.array(q_traj[starts], np.float32); v0 = np.array(v_traj[starts], np.float32)
        if init_pos_noise_std > 0:
            q0[:, :4] += noise_rng.normal(0.0, init_pos_noise_std, (B, 4)).astype(np.float32)
        hist_idx = starts[:, None] - history_length + np.arange(history_length)[None, :]
        valid = hist_idx >= traj_starts[:, None]
        hist_np = data_values[np.maximum(hist_idx, traj_starts[:, None])].astype(np.float32)
        idx = starts[:, None] + np.arange(cur)[None, :]
        to_t = lambda a: torch.from_numpy(
            np.ascontiguousarray(np.swapaxes(a, 0, 1))).float().to(dev)

        hist0 = norm(torch.from_numpy(hist_np).to(dev))
        hist0 = hist0 * torch.from_numpy(valid[:, :, None].astype(np.float32)).to(dev)
        sim_step = (be.step if hp["explicit"]
                    else (lambda q, v, c, f: be.step(q, v, c, f, route_force=False)))
        loss, aux = rollout_loss(
            model, sim_step, norm, hp,
            torch.from_numpy(q0).to(dev), torch.from_numpy(v0).to(dev), hist0,
            to_t(gt_pos[idx + 1]), to_t(data_values[idx]), to_t(gt_force[idx + 1]),
            training=True)

        optim.zero_grad()
        loss.backward()
        applied, finite = optim.step()
        if not finite:
            nonfinite += 1
            print(f"[warn] non-finite grads at ep {ep} (total {nonfinite}); "
                  f"{'applied anyway (apply_if_finite limit hit)' if applied else 'step skipped'}")
        # EMA updates every epoch even when the step was skipped, same as the JAX trainer
        if ema is not None:
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    ema[k].mul_(ema_decay).add_(sd[k], alpha=1.0 - ema_decay)

        if ckpt_dir and ckpt_interval and (ep + 1) % ckpt_interval == 0:
            save_ckpt(f"ep{ep+1:06d}", ep + 1)
        if ep % 10 == 0 or ep == epochs - 1:
            pj = aux["per_joint"].detach().cpu().numpy()
            rec = dict(epoch=ep, loss=float(loss), arm=float(aux["arm"]), grip=float(aux["grip"]),
                       force=float(aux["force"]), gate=float(aux["gate"]),
                       pos_mae=float(aux["pos_mae"]), force_mae=float(aux["force_mae"]),
                       j_deg=[round(float(x), 3) for x in pj[:4] * 180 / np.pi],
                       grip_mm=round(float(pj[4] * 1000), 3),
                       sec=round(time.time() - t0, 1))
            print(f"[newton_torch] ep {ep}: loss={rec['loss']:.4f} arm={rec['arm']:.5f} "
                  f"grip={rec['grip']:.4f} force={rec['force']:.4f} gate={rec['gate']:.4f} "
                  f"J(deg)={rec['j_deg']} grip={rec['grip_mm']}mm  [{rec['sec']}s]")
            if logf:
                logf.write(json.dumps(rec) + "\n"); logf.flush()
    if ckpt_dir:
        save_ckpt("final", epochs)
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
