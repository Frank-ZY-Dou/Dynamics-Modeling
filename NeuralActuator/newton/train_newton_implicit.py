"""OMX actuator training on the Newton or MJX backend, implicit coupling.

Rollout and losses follow the public train_actuator_diffsim.py. Implicit means
the predicted force is supervised but never applied to the body (xfrc = 0).
Gradient agreement between the backends is covered by the tests in tests/.

Usage:
    python train_newton_implicit.py --config configs/omx_newton.yaml --backend newton
    python train_newton_implicit.py --config configs/omx_newton.yaml --backend mjx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax
import yaml
from flax.training.train_state import TrainState

import mujoco
from public_import import load_dataset, sample_valid_indices, validate_mujoco_joint_limits, create_model
from backends.base import GRIPPER_MIN, GRIPPER_MAX, ROBOT_XML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", choices=["newton", "mjx"], default="newton")
    ap.add_argument("--log_json", default=None, help="write per-epoch metrics as jsonl")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .pkl to resume from"
                         "rng states for an exact continuation")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed = int(cfg.get("seed", 0))
    B = int(cfg["batch_size"])
    rollout_steps = int(cfg["rollout_steps"])
    history_length = int(cfg["history_length"])
    data_dt = float(cfg["data_dt"])
    sim_step_size = int(cfg["sim_step_size"])
    epochs = int(cfg["epochs"])
    gripper_torque_clip = float(cfg.get("gripper_torque_clip", 1.5))
    init_pos_noise_std = float(cfg.get("init_pos_noise_std", 0.0))
    qvel_clip = float(cfg.get("qvel_clip", 10.0))
    w_pos = float(cfg["pos_loss_weight"]); w_grip = float(cfg["gripper_loss_weight"])
    w_force = float(cfg["force_loss_weight"]); w_gate = float(cfg["gate_loss_weight"])
    force_focal = float(cfg.get("force_focal_weight", 5.0))
    explicit = bool(cfg.get("apply_external_force", False))

    xml = os.path.join(os.path.dirname(__file__), ROBOT_XML)
    mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(xml))

    # data comes through the public pipeline, trajectory boundaries preserved
    result = load_dataset(cfg["datasets"], mj_model, int(cfg.get("downsample_factor", 1)),
                          return_boundaries=True, cfg=cfg)
    data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries = result
    validate_mujoco_joint_limits(mj_model, data_values)
    F = data_values.shape[1]
    print(f"data: {len(data_values)} samples, {len(boundaries)} trajs, feature_dim={F}")

    # normalization, matches train_actuator_diffsim.py:849-880
    feat_mean = data_values.mean(axis=0).astype(np.float32)
    feat_std = data_values.std(axis=0).astype(np.float32)
    std_floor = np.array([0.05] * 5 + [0.05] * 4 + [1.0] + [10.0] * 5 + [0.1] * 5
                         + [0.1] * 5 + [1.0] * 5 + [1.0] + [0.05] * 4 + [1.0], dtype=np.float32)
    feat_std = np.maximum(feat_std, std_floor)
    mean_j, std_j = jnp.asarray(feat_mean), jnp.asarray(feat_std)

    def normalize(x):
        return jnp.clip((x - mean_j) / std_j, -10.0, 10.0)

    model = create_model(
        model_type=cfg.get("model_type", "transformer"),
        hidden_dim=int(cfg["hidden_dim"]), latent_dim=int(cfg["latent_dim"]),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        backbone_activation=cfg.get("backbone_activation", "silu"),
        num_heads=int(cfg["num_heads"]), num_layers=int(cfg["num_layers"]),
        d_ff=int(cfg["d_ff"]), pool_type=cfg.get("pool_type", "mean"),
        use_gated_attention=bool(cfg.get("use_gated_attention", True)),
        zero_init_head=bool(cfg.get("zero_init_torque_head", False)),
    )
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = model.init({"params": init_rng, "dropout": init_rng},
                        jnp.ones((1, history_length * F)), jnp.ones((1, F)), None, ts=data_dt)
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model params: {n_par:,}")

    if args.backend == "newton":
        from backends.newton_backend import NewtonBackend
        be = NewtonBackend(B, data_dt, sim_step_size, differentiable=True)
        sim_step = be.step_diff
    else:
        from backends.mjx_backend import MJXBackend
        be = MJXBackend(B, data_dt, sim_step_size)
        sim_step = be.step
    print(f"backend: {args.backend} | coupling: {'EXPLICIT' if explicit else 'implicit'}")

    data_j = jnp.asarray(data_values); gt_pos_j = jnp.asarray(gt_pos)
    q_traj_j = jnp.asarray(q_traj); v_traj_j = jnp.asarray(v_traj)
    gt_force_j = jnp.asarray(gt_force); force_valid_j = jnp.asarray(force_valid)

    tau_limit = jnp.array([5.0, 5.0, 5.0, 5.0, gripper_torque_clip])
    zeros_f = jnp.zeros((B, 3), jnp.float32)

    def smooth_l1(x):
        ax = jnp.abs(x)
        return jnp.where(ax < 1.0, 0.5 * x * x, ax - 0.5)

    # rollout loss, ported from train_actuator_diffsim.py:1149-1416
    def loss_fn(params, rngs_steps, q0, v0, hist0, tgt_seq, feat_seq, gtf_seq, fv_seq, training):
        def step_fn(carry, inp):
            q, v, hist = carry
            tgt, feats, gtf, fv, rk = inp              # (B,5),(B,F),(B,3),(B,3),(2,)

            # refresh the sim-state feature columns
            feat = feats
            feat = feat.at[:, 5:9].set(q[:, :4])
            ap_mm = q[:, 4] * 1000.0
            feat = feat.at[:, 9].set(ap_mm)
            feat = feat.at[:, 15:20].set(v[:, :5])
            feat = feat.at[:, 31:35].set(feat[:, 0:4] - q[:, :4])
            feat = feat.at[:, 35].set(feat[:, 30] - ap_mm)
            feat_n = normalize(feat)

            rngs = {"dropout": rk[0], "gumbel": rk[1]} if training else None
            tau_p, f_final, _f_raw, gate, _cond, _ = model.apply(
                params, hist.reshape(B, -1), feat_n, None, ts=data_dt,
                training=training, rngs=rngs)
            tau = jnp.clip(tau_p, -tau_limit, tau_limit)

            # implicit coupling: force supervised but not applied, xfrc stays 0
            xfrc = f_final if explicit else zeros_f
            q2, v2 = sim_step(q, v, tau, xfrc)

            # state surgery (train_actuator_diffsim.py:1267-1285)
            q2 = q2.at[:, 4].set(jnp.clip(q2[:, 4], GRIPPER_MIN, GRIPPER_MAX))
            q2 = q2.at[:, 5].set(jnp.clip(q2[:, 5], GRIPPER_MIN, GRIPPER_MAX))
            q_safe = jnp.nan_to_num(q2, nan=0.0)
            nan_mask = jnp.isnan(q2[:, :5])
            q_safe = q_safe.at[:, :5].set(jnp.where(nan_mask, tgt, q_safe[:, :5]))
            q2 = q_safe
            v2 = jnp.nan_to_num(jnp.clip(v2, -qvel_clip, qvel_clip),
                                nan=0.0, posinf=qvel_clip, neginf=-qvel_clip)

            hist2 = jnp.roll(hist, -1, axis=1).at[:, -1].set(feat_n)

            # losses
            arm_err = jnp.mean(smooth_l1(q2[:, :4] - tgt[:, :4]), axis=1)      # (B,)
            grip_err = smooth_l1((q2[:, 4] - tgt[:, 4]) * 1000.0)
            fmag = jnp.sqrt(jnp.sum(gtf ** 2, axis=1))
            has_f = (fmag > 0.01).astype(jnp.float32)
            focal = has_f * (force_focal - 1.0) + 1.0
            ferr = jnp.mean(smooth_l1(f_final - gtf), axis=1) * focal
            gate_p = jnp.clip(gate[:, 0], 1e-7, 1 - 1e-7)
            gate_err = -(has_f * jnp.log(gate_p) + (1 - has_f) * jnp.log(1 - gate_p))
            pj = jnp.abs(q2[:, :5] - tgt)                                       # (B,5)
            metrics = (jnp.mean(arm_err), jnp.mean(grip_err), jnp.mean(ferr),
                       jnp.mean(gate_err), jnp.mean(pj[:, :4]) ,
                       jnp.mean(jnp.abs(f_final - gtf)),
                       jnp.mean(pj, axis=0))
            return (q2, v2, hist2), metrics

        (_, _, _), ms = jax.lax.scan(step_fn, (q0, v0, hist0),
                                     (tgt_seq, feat_seq, gtf_seq, fv_seq, rngs_steps))
        arm_l, grip_l, force_l, gate_l, pos_mae, force_mae, pjm = ms
        total = (w_pos * (jnp.mean(arm_l) + w_grip * jnp.mean(grip_l))
                 + w_force * jnp.mean(force_l) + w_gate * jnp.mean(gate_l))
        aux = dict(arm=jnp.mean(arm_l), grip=jnp.mean(grip_l), force=jnp.mean(force_l),
                   gate=jnp.mean(gate_l), pos_mae=jnp.mean(pos_mae),
                   force_mae=jnp.mean(force_mae), per_joint=jnp.mean(pjm, axis=0))
        return total, aux

    lr = float(cfg["lr"])
    warm = min(max(int(cfg.get("lr_warmup_epochs", 0)), 1), max(epochs // 2, 1))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01, peak_value=lr, warmup_steps=warm,
        decay_steps=max(int(cfg.get("lr_decay_epochs", epochs)), warm + 1), end_value=lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(float(cfg.get("grad_clip", 1.0))),
                     optax.adamw(sched, b2=0.999, weight_decay=float(cfg.get("weight_decay", 1e-4))))
    tx = optax.apply_if_finite(tx, max_consecutive_errors=200)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, rng, q0, v0, hist0, tgt_seq, feat_seq, gtf_seq, fv_seq):
        n_steps = tgt_seq.shape[0]
        rngs_steps = jax.random.split(rng, n_steps * 2).reshape(n_steps, 2, 2)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, rngs_steps, q0, v0, hist0, tgt_seq, feat_seq, gtf_seq, fv_seq, True)
        return state.apply_gradients(grads=grads), loss, aux

    # rollout-length curriculum
    cur_eps = list(cfg.get("curriculum_epochs", []) or [])
    cur_steps = list(cfg.get("curriculum_steps", []) or [])

    def steps_for_epoch(ep):
        for e, st in zip(cur_eps, cur_steps):
            if ep < e:
                return int(st)
        return rollout_steps

    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema_params = jax.tree_util.tree_map(jnp.array, state.params) if ema_decay > 0 else None
    ema_update = jax.jit(lambda e, p: jax.tree_util.tree_map(
        lambda a, b: ema_decay * a + (1.0 - ema_decay) * b, e, p)) if ema_decay > 0 else None

    ckpt_dir = cfg.get("ckpt_dir")
    ckpt_interval = int(cfg.get("ckpt_interval", 0))
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    np_rng = np.random.default_rng(seed)

    def save_ckpt(tag, ep):
        """Pickle everything --resume needs: params, EMA, optimizer, both rng streams."""
        import pickle
        payload = dict(params=jax.device_get(state.params),
                       feature_mean=feat_mean, feature_std=feat_std, epoch=ep,
                       opt_state=jax.device_get(state.opt_state),
                       step=int(state.step),
                       rng=np.asarray(rng),
                       np_rng_state=np_rng.bit_generator.state)
        if ema_params is not None:
            payload["ema_params"] = jax.device_get(ema_params)
        tmp = os.path.join(ckpt_dir, f".ckpt_{tag}.pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh)
        os.replace(tmp, os.path.join(ckpt_dir, f"ckpt_{tag}.pkl"))  # atomic

    start_ep = 0
    if args.resume:
        import pickle
        payload = pickle.load(open(args.resume, "rb"))
        assert "opt_state" in payload, "resume needs a full-state checkpoint (with optimizer state)"
        state = state.replace(
            params=jax.tree_util.tree_map(jnp.asarray, payload["params"]),
            opt_state=jax.tree_util.tree_map(jnp.asarray, payload["opt_state"]),
            step=payload["step"])
        if ema_params is not None:
            assert "ema_params" in payload, "resume ckpt lacks EMA state"
            ema_params = jax.tree_util.tree_map(jnp.asarray, payload["ema_params"])
        rng = jnp.asarray(payload["rng"])
        np_rng.bit_generator.state = payload["np_rng_state"]
        start_ep = payload["epoch"]
        print(f"resumed from {args.resume} at epoch {start_ep}")

    logf = open(args.log_json, "a" if args.resume else "w") if args.log_json else None
    t0 = time.time()
    for ep in range(start_ep, epochs):
        cur = steps_for_epoch(ep)
        starts, traj_starts = sample_valid_indices(boundaries, history_length, cur, B, rng=np_rng)
        starts = np.asarray(starts); traj_starts = np.asarray(traj_starts)

        # init state + zero-padded history
        q0 = np.array(q_traj_j[starts]); v0 = np.array(v_traj_j[starts])
        rng, k = jax.random.split(rng)
        if init_pos_noise_std > 0:
            noise = np.asarray(jax.random.normal(k, (B, 4))) * init_pos_noise_std
            q0[:, :4] += noise
        hist_idx = starts[:, None] - history_length + np.arange(history_length)[None, :]
        valid = hist_idx >= traj_starts[:, None]
        hist = np.asarray(data_values)[np.maximum(hist_idx, traj_starts[:, None])]
        hist = np.where(valid[:, :, None], np.asarray(normalize(jnp.asarray(hist))), 0.0)

        # features at t, targets at t+1
        idx = starts[:, None] + np.arange(cur)[None, :]
        feat_seq = np.asarray(data_values)[idx]                    # (B,steps,F)
        tgt_seq = np.asarray(gt_pos)[idx + 1]                      # (B,steps,5)
        gtf_seq = np.asarray(gt_force)[idx + 1]
        fv_seq = np.asarray(force_valid)[idx + 1]
        to_t = lambda a: jnp.asarray(np.swapaxes(a, 0, 1), jnp.float32)   # (steps,B,...)

        rng, k = jax.random.split(rng)
        state, loss, aux = train_step(state, k,
                                      jnp.asarray(q0, jnp.float32), jnp.asarray(v0, jnp.float32),
                                      jnp.asarray(hist, jnp.float32),
                                      to_t(tgt_seq), to_t(feat_seq), to_t(gtf_seq), to_t(fv_seq))
        if ema_params is not None:
            ema_params = ema_update(ema_params, state.params)
        if ckpt_dir and ckpt_interval and (ep + 1) % ckpt_interval == 0:
            save_ckpt(f"ep{ep+1:06d}", ep + 1)
        if ep % 10 == 0 or ep == epochs - 1:
            pj = np.asarray(aux["per_joint"]); pj_deg = pj[:4] * 180 / np.pi
            rec = dict(epoch=ep, loss=float(loss), arm=float(aux["arm"]), grip=float(aux["grip"]),
                       force=float(aux["force"]), gate=float(aux["gate"]),
                       pos_mae=float(aux["pos_mae"]), force_mae=float(aux["force_mae"]),
                       j_deg=[round(float(x), 3) for x in pj_deg], grip_mm=round(float(pj[4] * 1000), 3),
                       sec=round(time.time() - t0, 1))
            print(f"[{args.backend}] ep {ep}: loss={rec['loss']:.4f} arm={rec['arm']:.5f} "
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
