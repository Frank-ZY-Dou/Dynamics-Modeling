"""Forward-dynamics parity: Newton backend vs the MJX reference.

    MODE=mjx JAX_ENABLE_X64=1 python tests/test_forward_parity.py run mjx64
    MODE=mjx python tests/test_forward_parity.py run mjx32
    MODE=newton python tests/test_forward_parity.py run newton
    python tests/test_forward_parity.py compare

16 lanes, 320 control steps of 4 substeps, data_dt=0.017. Lanes 0-3 hold
gravity only, 4-7 add constant torques, 8-15 add per-joint chirps, all inside
the trainer clamps; the same between-step surgery runs for every backend.
mjx32 vs mjx64 sets the float32 noise floor; newton has to stay within
max(3*floor, 0.01 rad) at step 60, 0.05 rad at step 320, and 1 mm on the
gripper.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

B = 16
N_STEPS = 320
DATA_DT = 0.017
SIM_STEP_SIZE = 4
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "g1"))

ARM_LO = np.array([-3.14159, -2.2, -2.0, -1.7])
ARM_HI = np.array([3.14159, 1.5, 1.7, 1.97])
GRIP_LO, GRIP_HI = -0.011, 0.02


def make_initial_states():
    """Mid-workspace poses. Without active torque J2's gravity equilibrium sits
    past its limit stop, so each lane gets a gravity-compensation hold torque,
    same job the network does during training."""
    rng = np.random.default_rng(0)
    lo = np.array([-1.0, -0.6, -0.5, -0.5])
    hi = np.array([1.0, 0.6, 0.5, 0.8])
    arm = rng.uniform(lo, hi, size=(B, 4))
    grip = np.full((B, 1), 0.0045) + rng.uniform(-0.002, 0.002, size=(B, 1))
    q0 = np.concatenate([arm, grip, grip], axis=1)  # equality: right = left
    v0 = np.zeros((B, 6))
    return q0, v0


def make_hold_torques(q0: np.ndarray) -> np.ndarray:
    """Per-lane gravity-compensation torques via MuJoCo inverse dynamics at rest
    (qvel=0, qacc=0), computed in plain MuJoCo f64 so no backend is favored."""
    import mujoco
    xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "robot", "omx_newton.xml"))
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    hold = np.zeros((B, 5))
    for b in range(B):
        d.qpos[:] = q0[b]; d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        hold[b, :4] = d.qfrc_inverse[:4]
        hold[b, 4] = 0.0        # finger gravity cancels pairwise via the equality
    return hold


_HOLD = None


def torque_program(t: float) -> np.ndarray:
    """Gravity hold plus per-lane excitation, (B,5). Everything stays well
    inside the trainer clamps."""
    tau = _HOLD.copy()
    const = np.array([0.15, 0.2, -0.12, -0.06, 0.005])
    for b in range(4, 8):
        tau[b] += const * (0.25 + 0.25 * (b - 4)) * (1 if b % 2 else -1)
    A = np.array([0.45, 0.35, 0.28, 0.18, 0.02])
    for b in range(8, 16):
        f = 0.3 + 0.15 * (b - 8)                      # 0.3 .. 1.35 Hz
        phase = np.arange(5) * 0.7 + b
        tau[b] += A * np.sin(2 * np.pi * f * t + phase)
    return tau


def surgery(q: np.ndarray, v: np.ndarray):
    """Trainer's between-step state surgery (train_actuator_diffsim.py:1267-1285)."""
    q = q.copy(); v = v.copy()
    q[:, 4] = np.clip(q[:, 4], GRIP_LO, GRIP_HI)
    q[:, 5] = np.clip(q[:, 5], GRIP_LO, GRIP_HI)
    v = np.clip(v, -10.0, 10.0)
    return q, v


def run(tag: str):
    mode = os.environ["MODE"]
    from backends.base import get_backend
    be = get_backend(mode, B, DATA_DT, SIM_STEP_SIZE)
    print(f"[{tag}] backend={be.name} dtype={be.dtype}")

    q, v = make_initial_states()
    global _HOLD
    _HOLD = make_hold_torques(q)
    traj = np.zeros((N_STEPS + 1, B, 6))
    traj[0] = q
    if mode == "mjx":
        import jax.numpy as jnp
        cast = lambda a: jnp.asarray(a, dtype=be.dtype)
    else:
        cast = lambda a: np.asarray(a, dtype=np.float32)

    for k in range(N_STEPS):
        tau = torque_program(k * DATA_DT)
        xfrc = np.zeros((B, 3))
        qn, vn = be.step(cast(q), cast(v), cast(tau), cast(xfrc))
        q, v = np.asarray(qn, dtype=np.float64), np.asarray(vn, dtype=np.float64)
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(v)), f"non-finite at step {k}"
        q, v = surgery(q, v)
        traj[k + 1] = q
        if (k + 1) % 80 == 0:
            print(f"  step {k+1}/{N_STEPS}")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"traj_{tag}.npz"), traj=traj)
    print(f"[{tag}] saved -> {OUT}/traj_{tag}.npz")


def compare():
    T = {t: np.load(os.path.join(OUT, f"traj_{t}.npz"))["traj"] for t in ("mjx64", "mjx32", "newton")}
    ref, f32, new = T["mjx64"], T["mjx32"], T["newton"]

    # Gate pass/fail on lanes that stay in range; training data is
    # limit-validated with a 0.05 rad margin anyway. At the limits MuJoCo's
    # constraint solver and Newton's penalty model just disagree.
    margin = np.minimum(ref[:, :, :4] - ARM_LO, ARM_HI - ref[:, :, :4]).min(axis=(0, 2))
    valid = margin > 0.03                          # (B,)
    n_valid = int(valid.sum())

    def maxdiff(a, b):
        return np.abs(a - b)[:, valid, :].max(axis=1)   # (N+1, 6) max over valid lanes

    floor = maxdiff(f32, ref)
    dnew = maxdiff(new, ref)
    lines = ["Forward parity report",
             f"lanes={B} steps={N_STEPS} dt={DATA_DT} substeps={SIM_STEP_SIZE}",
             f"in-range lanes gated: {n_valid}/{B} (min ref margin per lane: {np.round(margin, 3)})", ""]
    ok = n_valid >= 12
    if not ok:
        lines.append(f"FAIL: only {n_valid}/16 lanes stayed in range (need >= 12)")
    for step in (60, 320):
        fl_arm = floor[step, :4].max()
        nw_arm = dnew[step, :4].max()
        gate = max(3 * fl_arm, 0.01) if step == 60 else 0.05
        ok &= nw_arm <= gate
        lines.append(f"step {step:3d}: arm floor(f32)={fl_arm:.5f} rad | newton={nw_arm:.5f} rad | gate {gate:.5f} -> {'PASS' if nw_arm <= gate else 'FAIL'}")
    nw_grip = dnew[320, 4:6].max()
    ok &= nw_grip <= 1e-3
    lines.append(f"step 320: gripper newton={nw_grip*1000:.3f} mm | gate 1.000 mm -> {'PASS' if nw_grip <= 1e-3 else 'FAIL'}")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    report = "\n".join(lines)
    print(report)
    open(os.path.join(OUT, "report.txt"), "w").write(report + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        t = np.arange(N_STEPS + 1) * DATA_DT
        for j in range(4):
            axes[0].plot(t, dnew[:, j], label=f"J{j+1} newton")
            axes[0].plot(t, floor[:, j], "--", lw=0.8, label=f"J{j+1} f32 floor")
        axes[0].set(title="arm |dq| vs MJX-f64 [rad]", xlabel="time [s]", yscale="log")
        axes[0].legend(fontsize=6, ncol=2)
        axes[1].plot(t, dnew[:, 4] * 1e3, label="grip newton [mm]")
        axes[1].plot(t, floor[:, 4] * 1e3, "--", label="grip f32 floor [mm]")
        axes[1].set(title="gripper |dq| [mm]", xlabel="time [s]", yscale="log")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "parity.png"), dpi=140)
        print(f"plot -> {OUT}/parity.png")
    except Exception as e:  # plot is best-effort
        print("plot skipped:", e)
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "run":
        run(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
