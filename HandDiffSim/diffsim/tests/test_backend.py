"""Checks of the MuJoCo Warp hand backend.

A. forward parity against plain MuJoCo (float64) on random states
B. determinism and batch independence of the CUDA-graph forward
C. gradient of a rollout loss: backend VJPs against a global central difference of the loss (all inputs)
D. torch forward kinematics against MuJoCo site positions
E. servo law inverse round trip

    python -m diffsim.tests.test_backend
"""
import os
import sys
import time

import numpy as np
import torch

from diffsim.backend import HandWarpBackend, NQ, NU
from diffsim.kinematics_torch import HandKinematicsTorch
from diffsim.servo import ServoPD

B, DATA_DT, NSUB, K = 4, 0.02, 10, 6
DEV = "cuda"
rng = np.random.default_rng(3)
failures = []


def gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def sample_state(K_pip):
    q8 = rng.uniform(-0.5, 0.6, (B, 8)).astype(np.float32)
    q = np.zeros((B, NQ), np.float32)
    q[:, 0::3], q[:, 1::3] = q8[:, 0::2], q8[:, 1::2]
    q[:, 2::3] = K_pip(torch.from_numpy(q[:, 1::3])).numpy()
    v = rng.normal(0, 0.3, (B, NQ)).astype(np.float32)
    c = rng.normal(0, 0.02, (B, NU)).astype(np.float32)
    return q, v, c


def rel_cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    return np.linalg.norm(a - b) / denom, float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


t0 = time.time()
be = HandWarpBackend(B, DATA_DT, NSUB, device=DEV)
fk = HandKinematicsTorch("right", device="cpu")
print(f"backend built in {time.time() - t0:.1f}s: B={B}, W={be.W} worlds for the Jacobian, n_in={be.n_in}")

print("=== A: forward parity vs plain MuJoCo (f64) ===")
worst_q = worst_v = 0.0
for _ in range(3):
    q, v, c = sample_state(fk.pip_of_mcp)
    tq, tv = be.step(torch.from_numpy(q).to(DEV), torch.from_numpy(v).to(DEV), torch.from_numpy(c).to(DEV))
    rq, rv = be.reference_step(q, v, c)
    worst_q = max(worst_q, float(np.abs(tq.cpu().numpy() - rq).max()))
    worst_v = max(worst_v, float(np.abs(tv.cpu().numpy() - rv).max()))
gate("forward parity", worst_q < 1e-5 and worst_v < 1e-3, f"max |dq| = {worst_q:.2e} rad, max |dv| = {worst_v:.2e} rad/s")

print("=== B: determinism / batch independence ===")
q, v, c = sample_state(fk.pip_of_mcp)
args = [torch.from_numpy(x).to(DEV) for x in (q, v, c)]
o1 = be.step(*args); o2 = be.step(*args)
det = max(float((o1[0] - o2[0]).abs().max()), float((o1[1] - o2[1]).abs().max()))
gate("determinism", det == 0.0, f"repeat-call max diff {det:.1e}")
q2 = q.copy(); q2[1:] = rng.uniform(-0.5, 0.6, (B - 1, NQ))
o3 = be.step(torch.from_numpy(q2).to(DEV), args[1], args[2])
ind = float((o3[0][0] - o1[0][0]).abs().max())
gate("batch independence", ind == 0.0, f"sample 0 unchanged when others change: diff {ind:.1e}")

print(f"=== C: rollout gradient (K={K}) vs global finite differences ===")
q0, v0, _ = sample_state(fk.pip_of_mcp)
cs = [rng.normal(0, 0.02, (B, NU)).astype(np.float32) for _ in range(K)]
wq = rng.normal(0, 1, (B, NQ)).astype(np.float32)
wv = rng.normal(0, 0.1, (B, NQ)).astype(np.float32)


def rollout_loss(q, v, cs_):
    for k in range(K):
        q, v = be.step(q, v, cs_[k])
    return (q * torch.from_numpy(wq).to(DEV)).sum() + (v * torch.from_numpy(wv).to(DEV)).sum()


tq0 = torch.from_numpy(q0).to(DEV).requires_grad_(True)
tv0 = torch.from_numpy(v0).to(DEV).requires_grad_(True)
tcs = [torch.from_numpy(x).to(DEV).requires_grad_(True) for x in cs]
t0 = time.time()
L = rollout_loss(tq0, tv0, tcs)
grads = torch.autograd.grad(L, [tq0, tv0] + tcs)
torch.cuda.synchronize()
print(f"  forward+backward of a {K}-step rollout: {time.time() - t0:.2f}s")
for name, x0, g, h in (("d_q0", q0, grads[0], 2e-3), ("d_v0", v0, grads[1], 2e-2), ("d_ctrl", None, None, 2e-3)):
    dvec = rng.normal(0, 1, (K, B, NU) if name == "d_ctrl" else x0.shape).astype(np.float32)
    dvec /= np.linalg.norm(dvec)
    with torch.no_grad():
        if name == "d_ctrl":
            Lp = rollout_loss(torch.from_numpy(q0).to(DEV), torch.from_numpy(v0).to(DEV), [torch.from_numpy(cs[k] + h * dvec[k]).to(DEV) for k in range(K)])
            Lm = rollout_loss(torch.from_numpy(q0).to(DEV), torch.from_numpy(v0).to(DEV), [torch.from_numpy(cs[k] - h * dvec[k]).to(DEV) for k in range(K)])
            an = float(sum((torch.from_numpy(dvec[k]).to(DEV) * grads[2 + k]).sum() for k in range(K)))
        else:
            xp, xm = x0 + h * dvec, x0 - h * dvec
            if name == "d_q0":
                Lp = rollout_loss(torch.from_numpy(xp).to(DEV), torch.from_numpy(v0).to(DEV), [torch.from_numpy(x).to(DEV) for x in cs])
                Lm = rollout_loss(torch.from_numpy(xm).to(DEV), torch.from_numpy(v0).to(DEV), [torch.from_numpy(x).to(DEV) for x in cs])
            else:
                Lp = rollout_loss(torch.from_numpy(q0).to(DEV), torch.from_numpy(xp).to(DEV), [torch.from_numpy(x).to(DEV) for x in cs])
                Lm = rollout_loss(torch.from_numpy(q0).to(DEV), torch.from_numpy(xm).to(DEV), [torch.from_numpy(x).to(DEV) for x in cs])
            an = float((torch.from_numpy(dvec).to(DEV) * g).sum())
    fd = (float(Lp) - float(Lm)) / (2 * h)
    ratio = an / fd if fd != 0 else float("inf")
    gate(f"{name} directional derivative", 0.97 < ratio < 1.03, f"analytic {an:.5f} vs FD {fd:.5f} (ratio {ratio:.4f})")

# full gradient w.r.t. one ctrl step by component-wise FD (f64 reference sim), sample 0
k0 = K // 2
g_ref = np.zeros(NU)
h = 1e-3
for j in range(NU):
    def loss_np(cj):
        qs, vs = q0.astype(np.float64), v0.astype(np.float64)
        for k in range(K):
            ck = cs[k].astype(np.float64).copy()
            if k == k0:
                ck[0, j] = cj
            qs, vs = be.reference_step(qs, vs, ck)
        return float((qs[0] * wq[0]).sum() + (vs[0] * wv[0]).sum())
    g_ref[j] = (loss_np(cs[k0][0, j] + h) - loss_np(cs[k0][0, j] - h)) / (2 * h)
rel, cos = rel_cos(g_ref, grads[2 + k0][0].cpu().numpy())
gate("d_ctrl[k0] vs f64 component FD", cos > 0.999 and rel < 2e-2, f"cos {cos:.5f}, rel {rel:.2e}")

print("=== D: torch FK vs MuJoCo sites ===")
import mujoco
q, _, _ = sample_state(fk.pip_of_mcp)
tips, _ = fk(torch.from_numpy(q))
worst = 0.0
for b in range(B):
    d = be.mjd
    d.qpos[:] = q[b]
    mujoco.mj_forward(be.mjm, d)
    worst = max(worst, float(np.abs(d.site_xpos[be.tip_site] - tips[b].numpy()).max()))
gate("fk parity", worst < 1e-5, f"max |tip_torch - tip_mujoco| = {worst * 1e3:.4f} mm")

print("=== E: servo law inverse ===")
servo = ServoPD(kp=50.0, kd=0.5)
cmd = torch.randn(B, 8); qq = torch.randn(B, 8); vv = torch.randn(B, 8)
tau = servo.torque(cmd, qq, vv)
err = float((servo.command(tau, qq, vv) - cmd).abs().max())
gate("torque -> command round trip", err < 1e-5, f"max |cmd' - cmd| = {err:.1e}")

print()
print("hand backend tests:", "PASS" if not failures else f"FAIL ({failures})")
sys.exit(0 if not failures else 1)
