"""Backend protocol for the OMX differentiable-sim trainer.

A backend exposes one batched, stateless control step:

    step(qpos (B,6), qvel (B,6), ctrl (B,5), xfrc (B,3)) -> (qpos', qvel')

- One control step = `sim_step_size` physics substeps at dt = data_dt / sim_step_size.
- `ctrl` is the already-clamped torque vector [tau1..tau4, tau_gripper].
- `xfrc` is a world-frame force at the grasp point (end_effector_target); zeros = off.
- Stateless: mid-rollout state surgery (gripper clamp, NaN guards, qvel clip)
  happens in the caller between steps, same as the reference trainer.
"""
from __future__ import annotations

OMX_NQ = 6   # 4 arm hinges + 2 gripper slides
OMX_NU = 5   # 4 arm torques + 1 gripper torque (left finger)
GRIPPER_MIN, GRIPPER_MAX = -0.011, 0.02

ROBOT_XML = "robot/omx_newton.xml"  # both backends load this; scene.xml minus contact


def get_backend(name: str, batch_size: int, data_dt: float, sim_step_size: int):
    if name == "mjx":
        from backends.mjx_backend import MJXBackend
        return MJXBackend(batch_size, data_dt, sim_step_size)
    if name == "newton":
        from backends.newton_backend import NewtonBackend
        return NewtonBackend(batch_size, data_dt, sim_step_size)
    raise ValueError(f"unknown backend {name!r} (expected 'mjx' or 'newton')")
