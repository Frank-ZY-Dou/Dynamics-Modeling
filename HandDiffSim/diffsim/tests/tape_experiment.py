"""Does a Warp tape through mujoco_warp give gradients if backward compilation is forced on?

Outcome on mujoco_warp 3.12.0 / warp 1.17.0: the forward runs under the tape (131 launches recorded, ~100 s of
compilation) and tape.backward() executes, but the gradient with respect to qfrc_applied is identically zero
while central differences give values of order 1e-2. The internal Data arrays carry no gradient storage and
the solver updates in place, so the adjoint chain is broken. This is why backend.py uses finite differences.

    python diffsim/tests/tape_experiment.py
"""
import sys, time, importlib, pkgutil, numpy as np, mujoco, warp as wp
wp.init()
import mujoco_warp as mjw
import mujoco_warp._src as src
flipped = []
for modinfo in pkgutil.iter_modules(src.__path__):
    name = f"mujoco_warp._src.{modinfo.name}"
    if name.endswith("_test"): continue
    try: importlib.import_module(name)
    except Exception as e: print("import fail", name, e); continue
    wmod = wp.get_module(name)
    if wmod is not None:
        wmod.options["enable_backward"] = True; flipped.append(name)
print("flipped enable_backward on", len(flipped), "modules", flush=True)
import os
xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "serial_hand", "models", "AH_Right", "serial_hand.xml")
mjm = mujoco.MjModel.from_xml_path(xml); mjm.opt.gravity[:] = 0
mjd = mujoco.MjData(mjm)
B = 2
m = mjw.put_model(mjm); d = mjw.put_data(mjm, mjd, nworld=B)
rng = np.random.default_rng(0)
q0 = rng.uniform(-0.3, 0.3, (B, 12)); q0[:, 2::3] = 0.99*q0[:, 1::3]
v0 = rng.normal(0, 0.5, (B, 12)); tau = rng.normal(0, 0.02, (B, 12)); tau[:, 2::3] = 0
d.qpos.assign(q0.astype(np.float32)); d.qvel.assign(v0.astype(np.float32)); d.qfrc_applied.assign(tau.astype(np.float32))
d.qpos.requires_grad = True; d.qvel.requires_grad = True; d.qfrc_applied.requires_grad = True
loss = wp.zeros(1, dtype=float, requires_grad=True)
w = wp.array(rng.normal(0, 1, (B, 12)).astype(np.float32), dtype=float)

@wp.kernel
def dot_loss(q: wp.array2d(dtype=float), w: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    i, j = wp.tid()
    wp.atomic_add(out, 0, q[i, j] * w[i, j])

def run(qq, vv, tt):
    d2 = mjw.put_data(mjm, mjd, nworld=B)
    d2.qpos.assign(qq.astype(np.float32)); d2.qvel.assign(vv.astype(np.float32)); d2.qfrc_applied.assign(tt.astype(np.float32))
    mjw.step(m, d2); wp.synchronize()
    return float((d2.qpos.numpy() * w.numpy()).sum())

t0 = time.time()
tape = wp.Tape()
try:
    with tape:
        mjw.step(m, d)
        wp.launch(dot_loss, dim=(B, 12), inputs=[d.qpos, w], outputs=[loss])
    wp.synchronize()
    print(f"forward under tape ok ({time.time()-t0:.0f}s incl. compile); launches recorded: {len(tape.launches)}", flush=True)
    t0 = time.time()
    tape.backward(loss); wp.synchronize()
    print(f"tape.backward ran ({time.time()-t0:.0f}s incl. compile)", flush=True)
    g_tau = d.qfrc_applied.grad.numpy(); g_v = d.qvel.grad.numpy()
    print("tape grad wrt qfrc_applied (world 0):", np.round(g_tau[0], 5))
    h = 1e-2; fd = []
    for j in range(12):
        tp = tau.copy(); tp[0, j] += h; tm = tau.copy(); tm[0, j] -= h
        fd.append((run(q0, v0, tp) - run(q0, v0, tm)) / (2*h))
    print("FD   grad wrt qfrc_applied (world 0):", np.round(fd, 5))
    print("max |tape - FD|:", np.abs(np.array(fd) - g_tau[0]).max(), " max|FD|:", np.abs(fd).max())
except Exception as e:
    import traceback; traceback.print_exc()
    print("TAPE EXPERIMENT FAILED:", type(e).__name__, str(e)[:800])
