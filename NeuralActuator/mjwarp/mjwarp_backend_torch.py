"""Torch-native MuJoCo Warp backend: differentiable MuJoCo Warp with an
analytic adjoint, driven through torch.autograd.

The differentiable simulator is the adjoint branch of the mujoco_warp fork by
etaoxing (https://github.com/etaoxing/mujoco_warp, Apache-2.0), described in
the report "Differentiable MuJoCo Warp"
(https://etaoxing.com/reports/diff_mjw/diff_mjw.html). Install it from the
pinned commit in requirements.txt.

Mirrors newton/torch_native/newton_backend_torch.py: one control step
(n_sub out-of-place mjwarp substeps) is a torch autograd node whose backward
recomputes the substep chain under wp.Tape and reads the analytic adjoints.
External force is routed as tau_ext = J^T f through the same torch FK as the
Newton backend (diff_mjw has no xfrc_applied adjoint), added onto ctrl; the
sim itself runs force-free. Unlike the Newton backend, the gripper joint
equality stays native (diff_mjw has a joint-equality VJP) and joint limits are
MuJoCo's own constraint rows, so the model matches the MJX reference exactly.

ctrl within a control step is constant: it is written to the first Data and
propagates down the chain via step()'s untaped _copy_state, so each substep's
analytic ctrl adjoint lands in its own Data and dL/dctrl is their sum.
"""
from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import mujoco
import torch
import warp as wp

wp.config.quiet = True

import mujoco_warp as mjwarp

NDOF = 6   # 4 arm hinges + 2 gripper slides
NU = 5     # 4 arm torques + 1 gripper torque (left finger; right follows by equality)

_XML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "newton", "robot", "omx_newton.xml"))

_GRAD_ENABLED = False


def _enable_grad_once():
    global _GRAD_ENABLED
    if not _GRAD_ENABLED:
        mjwarp.enable_grad()  # one-way global; must precede put_model
        _GRAD_ENABLED = True


# FK chain constants for end_effector_target, same chain as
# newton/torch_native/newton_backend_torch.py (kept copy-local so this module
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


class _MJWarpStepFn(torch.autograd.Function):
    """One control step (n_sub mjwarp substeps) as a torch autograd node.

    Backward recomputes the chain under wp.Tape and reads the analytic
    adjoints; the tape lives for one step, so rollouts stay constant-memory.
    """

    @staticmethod
    def forward(ctx, backend, q, v, c):
        ctx.backend = backend
        ctx.save_for_backward(q, v, c)
        q_, v_, c_ = (t.detach().contiguous() for t in (q, v, c))
        qo = torch.empty_like(q_)
        vo = torch.empty_like(v_)
        backend._forward_raw(q_, v_, c_, qo, vo)
        return qo, vo

    @staticmethod
    @torch.autograd.function.once_differentiable  # sim VJP is first-order only
    def backward(ctx, g_q, g_v):
        q, v, c = ctx.saved_tensors
        dq, dv, dc = ctx.backend._backward_raw(
            q.detach().contiguous(), v.detach().contiguous(), c.detach().contiguous(),
            g_q.contiguous(), g_v.contiguous())
        return None, dq, dv, dc


@wp.kernel(enable_backward=False)
def _acc2d(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    i, j = wp.tid()
    dst[i, j] = dst[i, j] + src[i, j]


class MJWarpBackendTorch:
    name = "mjwarp_torch"

    def __init__(self, batch_size: int, data_dt: float, sim_step_size: int,
                 device: str = "cuda", capture: bool | None = None):
        self.B = batch_size
        self.n_sub = sim_step_size
        self.torch_device = torch.device(device)
        if self.torch_device.type != "cuda":
            raise ValueError("MJWarpBackendTorch requires a CUDA device")
        if self.torch_device.index is None:
            self.torch_device = torch.device("cuda", torch.cuda.current_device())

        wp.init()
        _enable_grad_once()
        self._wp_device = wp.device_from_torch(self.torch_device)

        mjm = mujoco.MjModel.from_xml_path(_XML)
        mjm.opt.timestep = data_dt / sim_step_size
        mjm.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        # Deliberate deviation from the MJX trainer's single-iteration solver
        # shortcut (train_actuator_diffsim.py:584-595): the analytic adjoint
        # here uses the implicit function theorem at a converged stationary
        # point, so with iterations=1 the backward would correspond to the
        # converged map while the forward is the one-iteration map, and the
        # control gradients disagree with finite differences by large factors.
        # MJX differentiates through its unrolled iteration and tolerates the
        # shortcut; this backend instead runs the solver to convergence
        # (MuJoCo defaults), which finite-difference checks verify exact and
        # which stays within 4e-6 rad of plain-MuJoCo float64 over 320-step
        # rollouts on this model.
        mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # same as mjx_backend
        assert mjm.nq == NDOF and mjm.nv == NDOF and mjm.nu == NU, (mjm.nq, mjm.nv, mjm.nu)
        # engine-side ctrl clamp with a clamp-blind analytic adjoint would
        # silently truncate the routed force and corrupt dctrl/df; clamping in
        # torch (see step()) keeps the engine clamp a no-op and lets autograd
        # supply the exact clamp gradient
        cr = mjm.actuator_ctrlrange.copy()
        self.ctrl_lo = torch.as_tensor(cr[:, 0], dtype=torch.float32,
                                       device=self.torch_device)
        self.ctrl_hi = torch.as_tensor(cr[:, 1], dtype=torch.float32,
                                       device=self.torch_device)
        self.clamp_hits = torch.zeros((), device=self.torch_device)  # monitor, no host sync
        mjd = mujoco.MjData(mjm)
        with wp.ScopedDevice(self._wp_device):
            self.m = mjwarp.put_model(mjm)
            # per-substep Data chain; grads opt in per array (analytic adjoint inputs)
            self.chain = [mjwarp.put_data(mjm, mjd, nworld=batch_size)
                          for _ in range(self.n_sub + 1)]
            for d in self.chain:
                d.qpos.requires_grad = True
                d.qvel.requires_grad = True
                d.ctrl.requires_grad = True
            # preallocated backward scratch; also primes the id(m) caches so no
            # host reads happen inside a captured backward
            self._bc = mjwarp.create_backward_context(self.m, self.chain[0])
            # fixed staging (CUDA graphs bake addresses)
            B = batch_size
            self.st_q = wp.zeros((B, NDOF), dtype=float)
            self.st_v = wp.zeros((B, NDOF), dtype=float)
            self.st_c = wp.zeros((B, NU), dtype=float)
            self.st_qo = wp.zeros((B, NDOF), dtype=float)
            self.st_vo = wp.zeros((B, NDOF), dtype=float)
            self.st_gq = wp.zeros((B, NDOF), dtype=float)
            self.st_gv = wp.zeros((B, NDOF), dtype=float)
            self.st_dq = wp.zeros((B, NDOF), dtype=float)
            self.st_dv = wp.zeros((B, NDOF), dtype=float)
            self.st_dc = wp.zeros((B, NU), dtype=float)

        # capture is illegal on the legacy default stream; use a side stream
        self._side = torch.cuda.Stream(device=self.torch_device)
        self._wp_side = wp.stream_from_torch(self._side)
        self._capture = (os.environ.get("MJWARP_TORCH_CAPTURE", "1") == "1"
                         if capture is None else capture)
        self._graph_fwd = None      # None = not built, False = capture failed
        self._graph_bwd = None

    def _on_side_stream(self):
        backend = self

        class _Scope:
            def __enter__(self):
                self.cur = torch.cuda.current_stream(backend.torch_device)
                backend._side.wait_stream(self.cur)
                self.ws = wp.ScopedStream(backend._wp_side)
                self.ws.__enter__()
                self.dv = wp.ScopedDevice(backend._wp_device)
                self.dv.__enter__()

            def __exit__(self, *exc):
                self.dv.__exit__(*exc)
                self.ws.__exit__(*exc)
                self.cur.wait_stream(backend._side)

        return _Scope()

    def _run(self, which: str, body):
        """Run a fixed launch sequence eagerly, or as a captured CUDA graph."""
        if not self._capture:
            body()
            return
        g = self._graph_fwd if which == "fwd" else self._graph_bwd
        if g is None:
            body()                       # warmup: lazy allocations land before capture
            try:
                with wp.ScopedCapture() as cap:
                    body()
                g = cap.graph
            except Exception as e:
                print(f"[mjwarp_torch] {which} graph capture failed ({e}); eager fallback")
                g = False
            if which == "fwd":
                self._graph_fwd = g
            else:
                self._graph_bwd = g
        if g:
            wp.capture_launch(g)
        else:
            body()

    def _load_inputs(self):
        d0 = self.chain[0]
        wp.copy(d0.qpos, self.st_q)
        wp.copy(d0.qvel, self.st_v)
        wp.copy(d0.ctrl, self.st_c)
        # stateless protocol: the solve must be a pure function of (q, v, ctrl),
        # and the backward's recompute must retrace the forward exactly, so the
        # warmstart never carries over from whatever ran last
        d0.qacc_warmstart.zero_()

    def _fwd_body(self):
        self._load_inputs()
        for t in range(self.n_sub):
            mjwarp.step(self.m, self.chain[t], self.chain[t + 1])
        d_fin = self.chain[self.n_sub]
        wp.copy(self.st_qo, d_fin.qpos)
        wp.copy(self.st_vo, d_fin.qvel)

    def _forward_raw(self, q_t, v_t, c_t, qo_t, vo_t):
        with self._on_side_stream():
            wp.copy(self.st_q, wp.from_torch(q_t))
            wp.copy(self.st_v, wp.from_torch(v_t))
            wp.copy(self.st_c, wp.from_torch(c_t))
            self._run("fwd", self._fwd_body)
            wp.copy(wp.from_torch(qo_t), self.st_qo)
            wp.copy(wp.from_torch(vo_t), self.st_vo)

    def _bwd_body(self):
        for d in self.chain:  # stale adjoints from the previous call
            for a in (d.qpos, d.qvel, d.ctrl):
                if a.grad is not None:
                    a.grad.zero_()
        self._load_inputs()
        tape = wp.Tape()
        with tape:
            for t in range(self.n_sub):
                mjwarp.step(self.m, self.chain[t], self.chain[t + 1])
        d_fin = self.chain[self.n_sub]
        wp.copy(d_fin.qpos.grad, self.st_gq)
        wp.copy(d_fin.qvel.grad, self.st_gv)
        with mjwarp.backward_context(self._bc):
            tape.backward()
        d0 = self.chain[0]
        wp.copy(self.st_dq, d0.qpos.grad)
        wp.copy(self.st_dv, d0.qvel.grad)
        # constant ctrl broadcast down the untaped copy chain: total ctrl
        # adjoint is the sum of the per-substep ctrl adjoints
        self.st_dc.zero_()
        for t in range(self.n_sub):
            wp.launch(_acc2d, dim=(self.B, NU),
                      inputs=[self.chain[t].ctrl.grad], outputs=[self.st_dc])

    def _backward_raw(self, q_t, v_t, c_t, gq_t, gv_t):
        dq = torch.empty_like(q_t)
        dv = torch.empty_like(v_t)
        dc = torch.empty_like(c_t)
        with self._on_side_stream():
            wp.copy(self.st_q, wp.from_torch(q_t))
            wp.copy(self.st_v, wp.from_torch(v_t))
            wp.copy(self.st_c, wp.from_torch(c_t))
            wp.copy(self.st_gq, wp.from_torch(gq_t))
            wp.copy(self.st_gv, wp.from_torch(gv_t))
            self._run("bwd", self._bwd_body)
            wp.copy(wp.from_torch(dq), self.st_dq)
            wp.copy(wp.from_torch(dv), self.st_dv)
            wp.copy(wp.from_torch(dc), self.st_dc)
        return dq, dv, dc

    # public API, same shape contract as backends/base.py and the Newton
    # torch backend: step(qpos (B,6), qvel (B,6), ctrl (B,5), xfrc (B,3))
    def step(self, qpos, qvel, ctrl, xfrc, route_force: bool = True):
        """Differentiable control step. External force enters as tau_ext = J^T f
        on the arm joints; the warp sim itself runs force-free.
        route_force=False skips the force route (implicit-coupling callers)."""
        if route_force:
            tau = tau_ext_torch(qpos[:, :4], xfrc)
            ctrl = ctrl + torch.nn.functional.pad(tau, (0, 1)).to(ctrl.dtype)
        # clamp in torch so the engine's ctrlrange clamp never binds: autograd
        # then gives the exact clamp gradient (zero when saturated), where the
        # engine's analytic ctrl adjoint is clamp-blind
        clamped = torch.clamp(ctrl, self.ctrl_lo, self.ctrl_hi)
        with torch.no_grad():
            self.clamp_hits += (clamped != ctrl).any(dim=1).sum()
        return _MJWarpStepFn.apply(self, qpos, qvel, clamped)
