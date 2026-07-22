"""Parity tests: torch-native Newton backend vs the JAX-newton path.

Both backends run in the same process on the same GPU. Covers tau_ext on 512
random samples (1e-5), single-step forward parity and torch determinism,
single-step VJPs on all four inputs (cos > 0.9999, rel < 1e-3), and an 8-step
rollout grad check with a finite-difference cross-check on the torch side.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NEWTON_FFI_GRAPH_MODE", "NONE")
os.environ.setdefault("NEWTON_CAPTURE", "0")

import numpy as np
import torch
import jax
import jax.numpy as jnp

from backends.newton_backend import NewtonBackend, _tau_ext
from torch_native.newton_backend_torch import NewtonBackendTorch, tau_ext_torch

B, DATA_DT, NSUB, K = 4, 0.032, 4, 8
DEV = "cuda"
rng = np.random.default_rng(7)
failures = []


def gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def sample_state():
    """Random OMX states kept away from the joint limits (j2 and j3 are the
    tight ones)."""
    q = np.zeros((B, 6), np.float32)
    q[:, 0] = rng.uniform(-0.5, 0.5, B)
    q[:, 1] = rng.uniform(-1.2, 0.2, B)
    q[:, 2] = rng.uniform(-0.1, 1.2, B)
    q[:, 3] = rng.uniform(-0.5, 0.5, B)
    g = rng.uniform(-0.005, 0.015, B)
    q[:, 4] = g
    q[:, 5] = g
    v = rng.normal(0.0, 0.2, (B, 6)).astype(np.float32)
    v[:, 5] = v[:, 4]
    c = rng.normal(0.0, 0.5, (B, 5)).astype(np.float32)
    f = rng.normal(0.0, 1.0, (B, 3)).astype(np.float32)
    return q, v, c, f


def rel_cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    rel = np.linalg.norm(a - b) / denom
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return rel, cos


print("=== Check A: tau_ext torch vs JAX (512 samples) ===")
qa = rng.uniform(-1.2, 1.2, (512, 4)).astype(np.float32)
fa = rng.normal(0.0, 2.0, (512, 3)).astype(np.float32)
tj = np.asarray(_tau_ext(jnp.asarray(qa), jnp.asarray(fa)))
tt = tau_ext_torch(torch.from_numpy(qa).to(DEV), torch.from_numpy(fa).to(DEV))
d = float(np.abs(tj - tt.detach().cpu().numpy()).max())
gate("tau_ext parity", d < 1e-5, f"max |jax - torch| = {d:.3e}")

print("=== building backends (jax + torch, same GPU) ===")
jb = NewtonBackend(batch_size=B, data_dt=DATA_DT, sim_step_size=NSUB)
tb = NewtonBackendTorch(batch_size=B, data_dt=DATA_DT, sim_step_size=NSUB, device=DEV)

print("=== Check B: single-step forward parity + determinism ===")
worst = 0.0
for _ in range(5):
    q, v, c, f = sample_state()
    jq, jv = jb.step(jnp.asarray(q), jnp.asarray(v), jnp.asarray(c), jnp.asarray(f))
    tq, tv = tb.step(torch.from_numpy(q).to(DEV), torch.from_numpy(v).to(DEV),
                     torch.from_numpy(c).to(DEV), torch.from_numpy(f).to(DEV))
    worst = max(worst,
                float(np.abs(np.asarray(jq) - tq.detach().cpu().numpy()).max()),
                float(np.abs(np.asarray(jv) - tv.detach().cpu().numpy()).max()))
gate("forward parity", worst < 1e-5, f"max |jax - torch| over 5 states = {worst:.3e}")

q, v, c, f = sample_state()
args_t = [torch.from_numpy(x).to(DEV) for x in (q, v, c, f)]
o1 = tb.step(*args_t)
o2 = tb.step(*args_t)
det = max(float((o1[0] - o2[0]).abs().max()), float((o1[1] - o2[1]).abs().max()))
gate("torch determinism", det == 0.0, f"repeat-call max diff = {det:.1e}")

print("=== Check C: single-step VJP parity (all 4 inputs) ===")
q, v, c, f = sample_state()
wq = rng.normal(0.0, 1.0, (B, 6)).astype(np.float32)
wv = rng.normal(0.0, 1.0, (B, 6)).astype(np.float32)

_, vjp_fn = jax.vjp(jb.step_diff, jnp.asarray(q), jnp.asarray(v),
                    jnp.asarray(c), jnp.asarray(f))
jg = vjp_fn((jnp.asarray(wq), jnp.asarray(wv)))

tq_, tv_, tc_, tf_ = (torch.from_numpy(x).to(DEV).requires_grad_(True)
                      for x in (q, v, c, f))
oq, ov = tb.step(tq_, tv_, tc_, tf_)
loss = (oq * torch.from_numpy(wq).to(DEV)).sum() + (ov * torch.from_numpy(wv).to(DEV)).sum()
tg = torch.autograd.grad(loss, (tq_, tv_, tc_, tf_))
for nm, a, b in zip(("d_qpos", "d_qvel", "d_ctrl", "d_xfrc"), jg, tg):
    rel, cos = rel_cos(np.asarray(a), b.detach().cpu().numpy())
    gate(f"vjp {nm}", cos > 0.9999 and rel < 1e-3, f"cos={cos:.6f} rel={rel:.2e}")

print(f"=== Check D: K={K} rollout grad parity + FD self-consistency ===")
q0, v0, _, _ = sample_state()
cs = [rng.normal(0.0, 0.5, (B, 5)).astype(np.float32) for _ in range(K)]
fs = [rng.normal(0.0, 1.0, (B, 3)).astype(np.float32) for _ in range(K)]
wq = rng.normal(0.0, 1.0, (B, 6)).astype(np.float32)
wv = rng.normal(0.0, 1.0, (B, 6)).astype(np.float32)


def jax_rollout_loss(q, v, cs_, fs_):
    for k in range(K):
        q, v = jb.step_diff(q, v, cs_[k], fs_[k])
    return jnp.sum(q * jnp.asarray(wq)) + jnp.sum(v * jnp.asarray(wv))


jl, jvjp = jax.vjp(jax_rollout_loss, jnp.asarray(q0), jnp.asarray(v0),
                   [jnp.asarray(x) for x in cs], [jnp.asarray(x) for x in fs])
jgq, jgv, jgc, jgf = jvjp(jnp.ones(()))


def torch_rollout_loss(q, v, cs_, fs_):
    for k in range(K):
        q, v = tb.step(q, v, cs_[k], fs_[k])
    return (q * torch.from_numpy(wq).to(DEV)).sum() + (v * torch.from_numpy(wv).to(DEV)).sum()


tq0 = torch.from_numpy(q0).to(DEV).requires_grad_(True)
tv0 = torch.from_numpy(v0).to(DEV).requires_grad_(True)
tcs = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in cs]
tfs = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in fs]
tl = torch_rollout_loss(tq0, tv0, tcs, tfs)
tg = torch.autograd.grad(tl, [tq0, tv0] + tcs + tfs)

dl = abs(float(jl) - float(tl)) / max(abs(float(jl)), 1e-30)
gate("rollout loss parity", dl < 1e-4, f"jax={float(jl):.6f} torch={float(tl):.6f} rel={dl:.2e}")
rel, cos = rel_cos(np.asarray(jgq), tg[0].detach().cpu().numpy())
gate("rollout d_q0", cos > 0.9999 and rel < 1e-3, f"cos={cos:.6f} rel={rel:.2e}")
rel, cos = rel_cos(np.asarray(jgv), tg[1].detach().cpu().numpy())
gate("rollout d_v0", cos > 0.9999 and rel < 1e-3, f"cos={cos:.6f} rel={rel:.2e}")
rel, cos = rel_cos(np.stack([np.asarray(x) for x in jgc]),
                   np.stack([t.detach().cpu().numpy() for t in tg[2:2 + K]]))
gate("rollout d_ctrl(all k)", cos > 0.9999 and rel < 1e-3, f"cos={cos:.6f} rel={rel:.2e}")
rel, cos = rel_cos(np.stack([np.asarray(x) for x in jgf]),
                   np.stack([t.detach().cpu().numpy() for t in tg[2 + K:]]))
gate("rollout d_xfrc(all k)", cos > 0.9999 and rel < 1e-3, f"cos={cos:.6f} rel={rel:.2e}")

# FD self-consistency on the torch side (central difference with large h, along a random ctrl direction)
dvec = rng.normal(0.0, 1.0, (K, B, 5)).astype(np.float32)
dvec /= np.linalg.norm(dvec)
h = 1e-3
with torch.no_grad():
    lp = torch_rollout_loss(torch.from_numpy(q0).to(DEV), torch.from_numpy(v0).to(DEV),
                            [torch.from_numpy(cs[k] + h * dvec[k]).to(DEV) for k in range(K)],
                            [torch.from_numpy(x).to(DEV) for x in fs])
    lm = torch_rollout_loss(torch.from_numpy(q0).to(DEV), torch.from_numpy(v0).to(DEV),
                            [torch.from_numpy(cs[k] - h * dvec[k]).to(DEV) for k in range(K)],
                            [torch.from_numpy(x).to(DEV) for x in fs])
fd = (float(lp) - float(lm)) / (2 * h)
an = float(sum((torch.from_numpy(dvec[k]).to(DEV) * tg[2 + k].detach()).sum()
               for k in range(K)))
ratio = an / fd if fd != 0 else float("inf")
gate("torch FD self-consistency (ctrl dir)", 0.95 < ratio < 1.05,
     f"analytic={an:.6f} fd={fd:.6f} ratio={ratio:.4f}")

print()
print("torch-backend parity:", "PASS" if not failures else f"FAIL ({failures})")
sys.exit(0 if not failures else 1)
