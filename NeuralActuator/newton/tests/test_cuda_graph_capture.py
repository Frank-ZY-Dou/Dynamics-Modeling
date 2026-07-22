"""CUDA-graph capture tests for the torch newton backend.

Graph replay reruns the same kernels in the same order, so capture should
match eager bitwise; grads are also checked against the JAX reference.

Covers: graphs actually built, fwd bitwise vs eager, K=8 rollout grads
bitwise, single-step VJP vs JAX step_diff, repeat determinism, and a rough
speed check.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NEWTON_FFI_GRAPH_MODE", "NONE")
os.environ.setdefault("NEWTON_CAPTURE", "0")

import numpy as np
import torch
import jax
import jax.numpy as jnp

from backends.newton_backend import NewtonBackend
from torch_native.newton_backend_torch import NewtonBackendTorch

B, DATA_DT, NSUB, K = 4, 0.032, 4, 8
DEV = "cuda"
rng = np.random.default_rng(11)
failures = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def sample_state():
    q = np.zeros((B, 6), np.float32)
    q[:, 0] = rng.uniform(-0.5, 0.5, B)
    q[:, 1] = rng.uniform(-1.2, 0.2, B)
    q[:, 2] = rng.uniform(-0.1, 1.2, B)
    q[:, 3] = rng.uniform(-0.5, 0.5, B)
    g = rng.uniform(-0.005, 0.015, B)
    q[:, 4] = g; q[:, 5] = g
    v = rng.normal(0.0, 0.2, (B, 6)).astype(np.float32); v[:, 5] = v[:, 4]
    c = rng.normal(0.0, 0.5, (B, 5)).astype(np.float32)
    f = rng.normal(0.0, 1.0, (B, 3)).astype(np.float32)
    return q, v, c, f


print("=== building backends (eager torch / captured torch / jax) ===")
tb_e = NewtonBackendTorch(B, DATA_DT, NSUB, device=DEV, capture=False)
tb_c = NewtonBackendTorch(B, DATA_DT, NSUB, device=DEV, capture=True)
jb = NewtonBackend(batch_size=B, data_dt=DATA_DT, sim_step_size=NSUB)

# warm both graphs (fwd + bwd) with one grad step
q, v, c, f = sample_state()
args = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in (q, v, c, f)]
oq, ov = tb_c.step(*args)
torch.autograd.grad(oq.sum() + ov.sum(), args)

print("=== Check A: graphs engaged ===")
gate("fwd graph built", bool(tb_c._graph_fwd), f"graph={type(tb_c._graph_fwd).__name__}")
gate("bwd graph built", bool(tb_c._graph_bwd), f"graph={type(tb_c._graph_bwd).__name__}")

print("=== Check B: forward bitwise (capture vs eager) ===")
worst = 0.0
for _ in range(5):
    q, v, c, f = sample_state()
    ts = [torch.from_numpy(x).to(DEV) for x in (q, v, c, f)]
    e = tb_e.step(*ts)
    g = tb_c.step(*ts)
    worst = max(worst, float((e[0] - g[0]).abs().max()), float((e[1] - g[1]).abs().max()))
gate("fwd bitwise", worst == 0.0, f"max diff {worst:.1e}")

print("=== Check C: K=8 rollout grads bitwise (capture vs eager) ===")
q0, v0, _, _ = sample_state()
cs = [rng.normal(0.0, 0.5, (B, 5)).astype(np.float32) for _ in range(K)]
fs = [rng.normal(0.0, 1.0, (B, 3)).astype(np.float32) for _ in range(K)]
wq = rng.normal(0.0, 1.0, (B, 6)).astype(np.float32)
wv = rng.normal(0.0, 1.0, (B, 6)).astype(np.float32)


def rollout_grads(be):
    tq = torch.from_numpy(q0).to(DEV).requires_grad_(True)
    tv = torch.from_numpy(v0).to(DEV).requires_grad_(True)
    tcs = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in cs]
    tfs = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in fs]
    q_, v_ = tq, tv
    for k in range(K):
        q_, v_ = be.step(q_, v_, tcs[k], tfs[k])
    loss = (q_ * torch.from_numpy(wq).to(DEV)).sum() + (v_ * torch.from_numpy(wv).to(DEV)).sum()
    gs = torch.autograd.grad(loss, [tq, tv] + tcs + tfs)
    return float(loss), [g.detach().cpu().numpy() for g in gs]


le, ge = rollout_grads(tb_e)
lc, gc = rollout_grads(tb_c)
worst = max(float(np.abs(a - b).max()) for a, b in zip(ge, gc))
gate("rollout loss bitwise", le == lc, f"eager={le:.6f} captured={lc:.6f}")
gate("rollout grads bitwise", worst == 0.0, f"max diff {worst:.1e}")

print("=== Check D: captured VJP vs JAX step_diff ===")
q, v, c, f = sample_state()
_, vjp_fn = jax.vjp(jb.step_diff, jnp.asarray(q), jnp.asarray(v), jnp.asarray(c), jnp.asarray(f))
jg = vjp_fn((jnp.asarray(wq), jnp.asarray(wv)))
ts = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in (q, v, c, f)]
oq, ov = tb_c.step(*ts)
loss = (oq * torch.from_numpy(wq).to(DEV)).sum() + (ov * torch.from_numpy(wv).to(DEV)).sum()
tg = torch.autograd.grad(loss, ts)
ok = True
det = []
for nm, a, b in zip(("dq", "dv", "dc", "df"), jg, tg):
    a = np.asarray(a, np.float64).ravel()
    b = b.detach().cpu().numpy().ravel().astype(np.float64)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    ok &= cos > 0.9999
    det.append(f"{nm}:{cos:.6f}")
gate("captured VJP vs JAX", ok, " ".join(det))

print("=== Check E: repeat-call determinism under capture ===")
ts = [torch.from_numpy(x).to(DEV) for x in (q, v, c, f)]
o1 = tb_c.step(*ts); o2 = tb_c.step(*ts)
d = max(float((o1[0] - o2[0]).abs().max()), float((o1[1] - o2[1]).abs().max()))
gate("determinism", d == 0.0, f"max diff {d:.1e}")

print("=== Check F: speed (indicative, shared GPU) ===")


def bench(be, iters=30):
    tq = torch.from_numpy(q0).to(DEV)
    tv = torch.from_numpy(v0).to(DEV)
    tc = torch.from_numpy(cs[0]).to(DEV).requires_grad_(True)
    tf = torch.from_numpy(fs[0]).to(DEV).requires_grad_(True)

    def one():
        q_, v_ = be.step(tq, tv, tc, tf)
        torch.autograd.grad((q_.sum() + v_.sum()), (tc, tf))

    one(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


ms_e, ms_c = bench(tb_e), bench(tb_c)
# On a shared GPU dispatch overlaps execution and capture shows little wall-clock
# gain; treat this as a non-regression check, real speed numbers need an idle GPU.
gate("captured not slower (indicative, shared GPU)", ms_c < ms_e * 1.15,
     f"eager {ms_e:.2f} ms/step -> captured {ms_c:.2f} ms/step ({ms_e/ms_c:.2f}x)")

print()
print("capture:", "PASS" if not failures else f"FAIL ({failures})")
sys.exit(0 if not failures else 1)
