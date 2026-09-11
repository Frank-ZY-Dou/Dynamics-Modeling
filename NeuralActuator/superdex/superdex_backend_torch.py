"""Torch-native SuperDex backend: the Backward-Euler articulated-body engine
of the differentiable SuperDex fork (https://github.com/Frank-ZY-Dou/differentiable-superdex,
a fork of Meta's project_superdex with an adjoint step) driven through torch.autograd.

Same contract as the Newton and MuJoCo Warp backends: step(qpos (B,6), qvel (B,6),
ctrl (B,5), xfrc (B,3)) is one control step of n_sub substeps and a torch autograd
node whose backward runs the engine's adjoint. The engine is a CPU library, one
scene per lane, so the lanes run in worker processes (superdex_worker.py), each
owning a slice of the batch; the trainer's process only moves small arrays to and
from them. The workers may run a different Python from the trainer (the fork needs
Python 3.12): set SUPERDEX_PYTHON to that interpreter. SUPERDEX_WORKERS sets the
number of workers (default: one per lane, at most the CPU count); each holds a
core while it waits for the next step (SUPERDEX_SPIN=0 to let it block).

External force enters as tau_ext = J^T f through the same torch FK as the other
backends, added onto ctrl; the engine runs force-free. The gripper torque is split
in half onto the two finger joints (the Newton backend's equality surrogate).
Torques are clamped to the MuJoCo model's ctrlrange in torch, so the clamp
gradient is exact; the engine has no torque limit of its own. Joint damping and
armature are the MuJoCo model's; joint limits come from the URDF the robot
package was built from (robot/omx.urdf, the MuJoCo limits).
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
from multiprocessing.connection import Listener

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch

NDOF = 6   # 4 arm hinges + 2 gripper slides
NU = 5     # 4 arm torques + 1 gripper torque

_HERE = os.path.dirname(os.path.abspath(__file__))
_XML = os.path.join(_HERE, "..", "newton", "robot", "omx_newton.xml")   # ctrlrange
_BOT = os.path.join(_HERE, "robot", "omx.superdex_bot")
_WORKER = os.path.join(_HERE, "superdex_worker.py")


# FK chain constants for end_effector_target, the same chain as
# newton/torch_native/newton_backend_torch.py (a local copy, so this module
# needs neither newton nor jax installed):
#   p = t1 + Rz(q1) [ t2 + Ry(q2)(t3 + Ry(q3)(t4 + Ry(q4) t5)) ]
def _fk_batched(q: torch.Tensor) -> torch.Tensor:
    """(B,4) arm joint angles -> (B,3) world position of end_effector_target."""
    c1, s1 = torch.cos(q[:, 0]), torch.sin(q[:, 0])
    c2, s2 = torch.cos(q[:, 1]), torch.sin(q[:, 1])
    c3, s3 = torch.cos(q[:, 2]), torch.sin(q[:, 2])
    c4, s4 = torch.cos(q[:, 3]), torch.sin(q[:, 3])
    # Ry(a) @ (x, 0, z) = (c x + s z, 0, -s x + c z)
    u4x = 0.124 + 0.14 * c4
    u4z = -0.14 * s4
    u3x = 0.024 + c3 * u4x + s3 * u4z
    u3z = 0.128 - s3 * u4x + c3 * u4z
    u2x = c2 * u3x + s2 * u3z
    u2z = 0.0595 - s2 * u3x + c2 * u3z
    px = 0.012 + c1 * u2x
    py = s1 * u2x
    pz = 0.017 + u2z
    return torch.stack([px, py, pz], dim=-1)


def tau_ext_torch(q_arm: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Batched tau_ext = J_v(q)^T f via double-backward of the FK point,
    differentiable in both q_arm and f."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "tau_ext_torch needs autograd for J^T f; wrap eval rollouts in "
            "torch.no_grad() instead of torch.inference_mode()")
    need_graph = torch.is_grad_enabled() and (q_arm.requires_grad or f.requires_grad)
    with torch.enable_grad():
        q_in = q_arm if q_arm.requires_grad else q_arm.detach().requires_grad_(True)
        p = _fk_batched(q_in)
        (tau,) = torch.autograd.grad((p * f).sum(), q_in, create_graph=need_graph)
    return tau


class _SuperDexStepFn(torch.autograd.Function):
    """One control step (n_sub Backward-Euler substeps) as a torch autograd node.

    Backward recomputes the substeps in the workers under the engine's
    checkpointed adjoint sweep, so rollouts stay constant-memory.
    """

    @staticmethod
    def forward(ctx, backend, q, v, tau):
        ctx.backend = backend
        ctx.save_for_backward(q, v, tau)
        return backend._forward_raw(q.detach(), v.detach(), tau.detach())

    @staticmethod
    @torch.autograd.function.once_differentiable  # sim VJP is first-order only
    def backward(ctx, g_q, g_v):
        q, v, tau = ctx.saved_tensors
        dq, dv, dtau = ctx.backend._backward_raw(q, v, tau, g_q, g_v)
        return None, dq, dv, dtau


class SuperDexBackendTorch:
    name = "superdex_torch"

    def __init__(self, batch_size: int, data_dt: float, sim_step_size: int,
                 device: str = "cuda", workers: int | None = None,
                 python: str | None = None):
        self.B = batch_size
        self.n_sub = sim_step_size
        self.dt = data_dt / sim_step_size
        self.torch_device = torch.device(device)

        import mujoco
        mjm = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        assert mjm.nq == NDOF and mjm.nu == NU, (mjm.nq, mjm.nu)
        cr = mjm.actuator_ctrlrange.copy()
        self.ctrl_lo = torch.as_tensor(cr[:, 0], dtype=torch.float32, device=self.torch_device)
        self.ctrl_hi = torch.as_tensor(cr[:, 1], dtype=torch.float32, device=self.torch_device)
        self.clamp_hits = torch.zeros((), device=self.torch_device)  # monitor, no host sync
        self.max_adjoint_residual = 0.0                               # monitor
        self.nonfinite_lanes = 0                                      # monitor

        if workers is None:
            workers = int(os.environ.get("SUPERDEX_WORKERS", "0")) or min(batch_size, os.cpu_count() or 1)
        workers = max(1, min(int(workers), batch_size))
        counts = [batch_size // workers + (1 if i < batch_size % workers else 0) for i in range(workers)]
        bounds = np.cumsum([0] + counts)
        self.slices = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]

        python = python or os.environ.get("SUPERDEX_PYTHON") or sys.executable
        env = dict(os.environ, SUPERDEX_PRECISION="double", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        probe = subprocess.run([python, "-c", "import superdex.physics, superdex.robotics"],
                               env=env, capture_output=True, text=True, timeout=600)
        if probe.returncode != 0:
            raise RuntimeError(
                f"{python} cannot import the differentiable SuperDex fork (set SUPERDEX_PYTHON to an "
                f"interpreter that can; see superdex/requirements.txt):\n{probe.stderr[-2000:]}")
        authkey = os.urandom(16)
        self._listener = Listener(family="AF_UNIX", authkey=authkey)
        self._procs = [
            subprocess.Popen([python, _WORKER, self._listener.address, authkey.hex(), str(i), _BOT,
                              repr(self.dt), str(self.n_sub), str(counts[i])], env=env)
            for i in range(workers)]
        self._conns = [None] * workers
        # a worker that dies while building its lanes must raise here, not leave
        # accept() waiting forever: poll the processes between short waits
        self._listener._listener._socket.settimeout(2.0)
        for _ in range(workers):
            while True:
                dead = [i for i, p in enumerate(self._procs) if p.poll() is not None]
                if dead:
                    self.close()
                    raise RuntimeError(f"SuperDex worker {dead[0]} exited during startup (see its stderr)")
                try:
                    conn = self._listener.accept()
                    break
                except TimeoutError:
                    continue
            tag, index, ndof = conn.recv()
            assert tag == "ready" and ndof == NDOF, (tag, ndof)
            self._conns[index] = conn
        atexit.register(self.close)

    def close(self):
        for conn in getattr(self, "_conns", []):
            if conn is not None:
                try:
                    conn.send(("close",))
                    conn.close()
                except (OSError, EOFError):
                    pass
        self._conns = []
        for proc in getattr(self, "_procs", []):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs = []
        listener = getattr(self, "_listener", None)
        if listener is not None:
            listener.close()
            self._listener = None

    def __del__(self):
        self.close()

    def _exchange(self, kind: str, *arrays):
        """Send each worker its lanes' rows of the arrays, gather the replies in lane order."""
        for conn, sl in zip(self._conns, self.slices):
            conn.send((kind, *[a[sl] for a in arrays]))
        replies = []
        for i, conn in enumerate(self._conns):
            try:
                replies.append(conn.recv())
            except EOFError:
                raise RuntimeError(f"SuperDex worker {i} exited (see its stderr)") from None
        return replies

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(t.detach().to("cpu", torch.float64).numpy())

    def _forward_raw(self, q, v, tau):
        qn, vn, tn = (self._to_numpy(t) for t in (q, v, tau))
        replies = self._exchange("fwd", qn, vn, tn)
        q_out = np.concatenate([r[0] for r in replies])
        v_out = np.concatenate([r[1] for r in replies])
        self.nonfinite_lanes += int((~np.isfinite(q_out).all(axis=1)).sum())
        return (torch.as_tensor(q_out, dtype=q.dtype, device=q.device),
                torch.as_tensor(v_out, dtype=v.dtype, device=v.device))

    def _backward_raw(self, q, v, tau, g_q, g_v):
        qn, vn, tn, gqn, gvn = (self._to_numpy(t) for t in (q, v, tau, g_q, g_v))
        replies = self._exchange("bwd", qn, vn, tn, gqn, gvn)
        dq = np.concatenate([r[0] for r in replies])
        dv = np.concatenate([r[1] for r in replies])
        dtau = np.concatenate([r[2] for r in replies])
        self.max_adjoint_residual = max(self.max_adjoint_residual, max(r[3] for r in replies))
        return (torch.as_tensor(dq, dtype=q.dtype, device=q.device),
                torch.as_tensor(dv, dtype=v.dtype, device=v.device),
                torch.as_tensor(dtau, dtype=tau.dtype, device=tau.device))

    # public API, same shape contract as the Newton and MuJoCo Warp torch
    # backends: step(qpos (B,6), qvel (B,6), ctrl (B,5), xfrc (B,3))
    def step(self, qpos, qvel, ctrl, xfrc, route_force: bool = True):
        """Differentiable control step. External force enters as tau_ext = J^T f
        on the arm joints; the engine itself runs force-free.
        route_force=False skips the force route (implicit-coupling callers)."""
        if route_force:
            tau = tau_ext_torch(qpos[:, :4], xfrc)
            ctrl = ctrl + torch.nn.functional.pad(tau, (0, 1)).to(ctrl.dtype)
        lo, hi = self.ctrl_lo.to(ctrl.dtype), self.ctrl_hi.to(ctrl.dtype)
        clamped = torch.clamp(ctrl, lo, hi)
        with torch.no_grad():
            self.clamp_hits += (clamped != ctrl).any(dim=1).sum()
        half = 0.5 * clamped[:, 4:5]
        tau6 = torch.cat([clamped[:, :4], half, half], dim=1)
        return _SuperDexStepFn.apply(self, qpos, qvel, tau6)
