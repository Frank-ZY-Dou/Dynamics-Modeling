"""Training-graph loss/grad parity: flax/JAX-newton vs torch.

One shared batch through a K=16 rollout with explicit coupling so tau_ext is
exercised, dropout off, TF32 off on both sides. Pass: loss rel diff <= 1e-4
and global grad cosine >= 0.9999. Also prints the worst per-tensor cosines.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NEWTON_FFI_GRAPH_MODE", "NONE")
os.environ.setdefault("NEWTON_CAPTURE", "0")
os.environ["MUJOCO_GL"] = "egl"          # override any pre-set value, setdefault won't
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import jax

jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import yaml
import mujoco

from public_import import load_dataset, sample_valid_indices, validate_mujoco_joint_limits, create_model
from backends.base import GRIPPER_MIN, GRIPPER_MAX, ROBOT_XML
from backends.newton_backend import NewtonBackend
from torch_native.model_torch import TransformerActuatorTorch, load_flax_params, flax_param_pairs
from torch_native.newton_backend_torch import NewtonBackendTorch
from torch_native.train_newton_torch import rollout_loss

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "parity_smoke.yaml")
K = 16
DEV = "cuda"
cfg = yaml.safe_load(open(CFG))
B = int(cfg["batch_size"])
H = int(cfg["history_length"])
data_dt = float(cfg["data_dt"])
HP = dict(gripper_torque_clip=1.5, qvel_clip=10.0,
          w_pos=float(cfg["pos_loss_weight"]), w_grip=float(cfg["gripper_loss_weight"]),
          w_force=float(cfg["force_loss_weight"]), w_gate=float(cfg["gate_loss_weight"]),
          force_focal=float(cfg["force_focal_weight"]), explicit=True)

# data via the public pipeline
xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ROBOT_XML))
mj_model = mujoco.MjModel.from_xml_path(xml)
data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries = load_dataset(
    cfg["datasets"], mj_model, 1, return_boundaries=True, cfg=cfg)
data_values = np.asarray(data_values); q_traj = np.asarray(q_traj); v_traj = np.asarray(v_traj)
gt_pos = np.asarray(gt_pos); gt_force = np.asarray(gt_force)
validate_mujoco_joint_limits(mj_model, data_values)
F = data_values.shape[1]

feat_mean = data_values.mean(axis=0).astype(np.float32)
feat_std = data_values.std(axis=0).astype(np.float32)
std_floor = np.array([0.05] * 5 + [0.05] * 4 + [1.0] + [10.0] * 5 + [0.1] * 5
                     + [0.1] * 5 + [1.0] * 5 + [1.0] + [0.05] * 4 + [1.0], dtype=np.float32)
feat_std = np.maximum(feat_std, std_floor)

# flax init at the config seed, weights copied into torch
fm = create_model(model_type="transformer", hidden_dim=int(cfg["hidden_dim"]),
                  latent_dim=int(cfg["latent_dim"]), dropout_rate=float(cfg["dropout_rate"]),
                  backbone_activation="silu", num_heads=int(cfg["num_heads"]),
                  num_layers=int(cfg["num_layers"]), d_ff=int(cfg["d_ff"]),
                  pool_type="mean", use_gated_attention=True, zero_init_head=False)
_, init_rng = jax.random.split(jax.random.PRNGKey(int(cfg["seed"])))
fparams = fm.init({"params": init_rng, "dropout": init_rng},
                  jnp.ones((1, H * F)), jnp.ones((1, F)), None, ts=data_dt)
tm = TransformerActuatorTorch(feature_dim=F, hidden_dim=int(cfg["hidden_dim"]),
                              latent_dim=int(cfg["latent_dim"]), num_heads=int(cfg["num_heads"]),
                              num_layers=int(cfg["num_layers"]), d_ff=int(cfg["d_ff"]),
                              dropout=float(cfg["dropout_rate"]), gated=True, n_joints=5).to(DEV)
load_flax_params(tm, jax.device_get(fparams))
tm.to(DEV)

jb = NewtonBackend(batch_size=B, data_dt=data_dt, sim_step_size=int(cfg["sim_step_size"]))
tb = NewtonBackendTorch(batch_size=B, data_dt=data_dt, sim_step_size=int(cfg["sim_step_size"]),
                        device=DEV)

# one shared batch, same sampler rng the trainers use
np_rng = np.random.default_rng(int(cfg["seed"]))
starts, traj_starts = sample_valid_indices(boundaries, H, K, B, rng=np_rng)
starts = np.asarray(starts); traj_starts = np.asarray(traj_starts)
q0 = q_traj[starts].astype(np.float32); v0 = v_traj[starts].astype(np.float32)
hist_idx = starts[:, None] - H + np.arange(H)[None, :]
valid = (hist_idx >= traj_starts[:, None]).astype(np.float32)
hist_raw = data_values[np.maximum(hist_idx, traj_starts[:, None])].astype(np.float32)
hist0_np = np.clip((hist_raw - feat_mean) / feat_std, -10, 10) * valid[:, :, None]
idx = starts[:, None] + np.arange(K)[None, :]
feat_seq = np.swapaxes(data_values[idx], 0, 1).astype(np.float32)       # (K,B,F)
tgt_seq = np.swapaxes(gt_pos[idx + 1], 0, 1).astype(np.float32)
gtf_seq = np.swapaxes(gt_force[idx + 1], 0, 1).astype(np.float32)

# JAX side mirrors train_newton_implicit.py loss_fn, eval mode
mean_j, std_j = jnp.asarray(feat_mean), jnp.asarray(feat_std)
tau_limit = jnp.array([5.0, 5.0, 5.0, 5.0, HP["gripper_torque_clip"]])
zeros_f = jnp.zeros((B, 3), jnp.float32)


def normalize_j(x):
    return jnp.clip((x - mean_j) / std_j, -10.0, 10.0)


def smooth_l1_j(x):
    ax = jnp.abs(x)
    return jnp.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


def jax_loss(params):
    def step_fn(carry, inp):
        q, v, hist = carry
        tgt, feats, gtf = inp
        feat = feats
        feat = feat.at[:, 5:9].set(q[:, :4])
        ap_mm = q[:, 4] * 1000.0
        feat = feat.at[:, 9].set(ap_mm)
        feat = feat.at[:, 15:20].set(v[:, :5])
        feat = feat.at[:, 31:35].set(feat[:, 0:4] - q[:, :4])
        feat = feat.at[:, 35].set(feat[:, 30] - ap_mm)
        feat_n = normalize_j(feat)
        tau_p, f_final, _fr, gate, _c, _ = fm.apply(params, hist.reshape(B, -1), feat_n,
                                                    None, ts=data_dt, training=False)
        tau = jnp.clip(tau_p, -tau_limit, tau_limit)
        xfrc = f_final if HP["explicit"] else zeros_f
        q2, v2 = jb.step_diff(q, v, tau, xfrc)
        q2 = q2.at[:, 4].set(jnp.clip(q2[:, 4], GRIPPER_MIN, GRIPPER_MAX))
        q2 = q2.at[:, 5].set(jnp.clip(q2[:, 5], GRIPPER_MIN, GRIPPER_MAX))
        q_safe = jnp.nan_to_num(q2, nan=0.0)
        nan_mask = jnp.isnan(q2[:, :5])
        q2 = q_safe.at[:, :5].set(jnp.where(nan_mask, tgt, q_safe[:, :5]))
        v2 = jnp.nan_to_num(jnp.clip(v2, -HP["qvel_clip"], HP["qvel_clip"]),
                            nan=0.0, posinf=HP["qvel_clip"], neginf=-HP["qvel_clip"])
        hist2 = jnp.roll(hist, -1, axis=1).at[:, -1].set(feat_n)
        arm_err = jnp.mean(smooth_l1_j(q2[:, :4] - tgt[:, :4]), axis=1)
        grip_err = smooth_l1_j((q2[:, 4] - tgt[:, 4]) * 1000.0)
        fmag = jnp.sqrt(jnp.sum(gtf ** 2, axis=1))
        has_f = (fmag > 0.01).astype(jnp.float32)
        focal = has_f * (HP["force_focal"] - 1.0) + 1.0
        ferr = jnp.mean(smooth_l1_j(f_final - gtf), axis=1) * focal
        gate_p = jnp.clip(gate[:, 0], 1e-7, 1 - 1e-7)
        gate_err = -(has_f * jnp.log(gate_p) + (1 - has_f) * jnp.log(1 - gate_p))
        ms = (jnp.mean(arm_err), jnp.mean(grip_err), jnp.mean(ferr), jnp.mean(gate_err))
        return (q2, v2, hist2), ms

    carry = (jnp.asarray(q0), jnp.asarray(v0), jnp.asarray(hist0_np))
    ms = []
    for k in range(K):
        carry, m = step_fn(carry, (jnp.asarray(tgt_seq[k]), jnp.asarray(feat_seq[k]),
                                   jnp.asarray(gtf_seq[k])))
        ms.append(m)
    arm, grip, force, gate = (jnp.mean(jnp.stack([m[i] for m in ms])) for i in range(4))
    return (HP["w_pos"] * (arm + HP["w_grip"] * grip)
            + HP["w_force"] * force + HP["w_gate"] * gate)


print("computing JAX loss+grads (eager, K=16)...")
jloss, jgrads = jax.value_and_grad(jax_loss)(fparams)
jgrads = jax.device_get(jgrads)

print("computing torch loss+grads...")
tt = lambda a: torch.from_numpy(a).to(DEV)
tloss, _aux = rollout_loss(tm, tb.step,
                           lambda x: ((x - tt(feat_mean)) / tt(feat_std)).clamp(-10, 10),
                           HP, tt(q0), tt(v0), tt(hist0_np),
                           tt(tgt_seq), tt(feat_seq), tt(gtf_seq), training=False)
tloss.backward()

failures = []
dl = abs(float(jloss) - float(tloss)) / max(abs(float(jloss)), 1e-30)
ok = dl < 1e-4
print(f"  [{'PASS' if ok else 'FAIL'}] loss parity: jax={float(jloss):.6f} "
      f"torch={float(tloss):.6f} rel={dl:.2e}")
if not ok:
    failures.append("loss")

pairs = flax_param_pairs(tm, jgrads)
gj_all, gt_all, worst = [], [], []
for param, garr in pairs:
    # condition heads are unused: torch leaves grad None, jax gives zeros
    g = param.grad if param.grad is not None else torch.zeros_like(param)
    gt = g.detach().cpu().numpy().ravel().astype(np.float64)
    gj = np.asarray(garr).ravel().astype(np.float64)
    gj_all.append(gj); gt_all.append(gt)
    denom = max(np.linalg.norm(gj) * np.linalg.norm(gt), 1e-30)
    worst.append((float(gj @ gt / denom), float(np.linalg.norm(gj)), tuple(param.shape)))
gj_all = np.concatenate(gj_all); gt_all = np.concatenate(gt_all)
gcos = float(gj_all @ gt_all / max(np.linalg.norm(gj_all) * np.linalg.norm(gt_all), 1e-30))
grel = float(np.linalg.norm(gj_all - gt_all) / max(np.linalg.norm(gj_all), 1e-30))
ok = gcos > 0.9999
print(f"  [{'PASS' if ok else 'FAIL'}] global grad: cos={gcos:.6f} rel={grel:.2e} "
      f"(n={gj_all.size:,})")
if not ok:
    failures.append("global-grad")
worst.sort()
print("  worst per-tensor cosines (cos, |g|_jax, shape):")
for c, n, s in worst[:5]:
    print(f"    cos={c:.6f} |g|={n:.3e} shape={s}")

print()
print("training-graph parity:", "PASS" if not failures else f"FAIL ({failures})")
sys.exit(0 if not failures else 1)
