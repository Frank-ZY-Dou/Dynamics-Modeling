"""Torch-native Newton backend: the Featherstone sim core from
backends/newton_backend.py driven through torch.autograd instead of the JAX FFI.
External force is routed as tau_ext = J^T f through a torch FK, which avoids
the newton 1.4.0 State.body_f adjoint bug in the Featherstone solver.
Keep module-level imports jax-free; backends.newton_backend imports jax lazily.
"""
from __future__ import annotations

import torch
import warp as wp
import newton

from backends.base import OMX_NU, ROBOT_XML  # noqa: F401
from backends.newton_backend import (
    NDOF,
    NBODY,
    _scatter_state,
    _set_ctrl,
    _fill_bodyf,
    _gather_state,
    _copy2d,
)
import os


# chain constants copied from backends/newton_backend.py::_fk_ee_pos
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


class _NewtonStepFn(torch.autograd.Function):
    """One control step (n_sub Featherstone substeps) as a torch autograd node.

    Backward recomputes the step under wp.Tape and replays adjoints; the tape
    only lives for one step, so long rollouts stay constant-memory.
    """

    @staticmethod
    def forward(ctx, backend, q, v, c, f):
        ctx.backend = backend
        ctx.save_for_backward(q, v, c, f)
        q_, v_, c_, f_ = (t.detach().contiguous() for t in (q, v, c, f))
        qo = torch.empty_like(q_)
        vo = torch.empty_like(v_)
        backend._forward_raw(q_, v_, c_, f_, qo, vo)
        return qo, vo

    @staticmethod
    @torch.autograd.function.once_differentiable  # sim VJP is first-order only
    def backward(ctx, g_q, g_v):
        q, v, c, f = ctx.saved_tensors
        dq, dv, dc, df = ctx.backend._backward_raw(
            q.detach().contiguous(), v.detach().contiguous(),
            c.detach().contiguous(), f.detach().contiguous(),
            g_q.contiguous(), g_v.contiguous())
        return None, dq, dv, dc, df


class NewtonBackendTorch:
    name = "newton_torch"

    def __init__(self, batch_size: int, data_dt: float, sim_step_size: int,
                 newton_substeps: int | None = None, device: str = "cuda",
                 capture: bool | None = None):
        self.B = batch_size
        self.n_sub = newton_substeps if newton_substeps is not None else sim_step_size
        assert self.n_sub % 2 == 0, "even substep count (buffer identity invariant)"
        self.dt = data_dt / self.n_sub
        self.torch_device = torch.device(device)
        if self.torch_device.type != "cuda":
            raise ValueError("NewtonBackendTorch requires a CUDA device")
        if self.torch_device.index is None:
            self.torch_device = torch.device("cuda", torch.cuda.current_device())
        # build all warp objects on the torch device; finalize() uses warp's
        # current device (cuda:0 by default), which would silently split the
        # sim and the tensors across GPUs
        wp.init()
        wp_device = wp.device_from_torch(self.torch_device)

        xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ROBOT_XML))
        with wp.ScopedDevice(wp_device):
            robot = newton.ModelBuilder()
            robot.add_mjcf(xml, parse_meshes=False, parse_visuals=False,
                           enable_self_collisions=False, skip_equality_constraints=True,
                           collapse_fixed_joints=False)
            scene = newton.ModelBuilder()
            scene.replicate(robot, world_count=batch_size)
            self.model = scene.finalize(requires_grad=True)
        assert str(self.model.device) == str(wp_device), (self.model.device, wp_device)
        assert self.model.joint_dof_count == batch_size * NDOF
        assert self.model.body_count == batch_size * NBODY

        # joint-limit gains matched to MuJoCo's constraint stiffness
        lim_ke = float(os.environ.get("NEWTON_LIMIT_KE", "250.0"))
        lim_kd = float(os.environ.get("NEWTON_LIMIT_KD", "10.0"))
        self.model.joint_limit_ke.fill_(lim_ke)
        self.model.joint_limit_kd.fill_(lim_kd)

        self.solver = newton.solvers.SolverFeatherstone(
            self.model, angular_damping=0.0, update_mass_matrix_interval=1)
        self.control = self.model.control()
        self.s0 = self.model.state()
        self.s1 = self.model.state()

        B, dev = self.B, self.model.device
        self.st_q = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_v = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_c = wp.zeros((B, OMX_NU), dtype=float, device=dev)
        self.st_f = wp.zeros((B, 3), dtype=float, device=dev)
        self.st_qo = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_vo = wp.zeros((B, NDOF), dtype=float, device=dev)
        # per-substep state chain + grad-enabled staging for the backward
        self.chain = [self.model.state() for _ in range(self.n_sub + 1)]
        rg = dict(dtype=float, device=dev, requires_grad=True)
        self.in_q = wp.zeros((B, NDOF), **rg)
        self.in_v = wp.zeros((B, NDOF), **rg)
        self.in_c = wp.zeros((B, OMX_NU), **rg)
        self.in_f = wp.zeros((B, 3), **rg)
        self.out_q = wp.zeros((B, NDOF), **rg)
        self.out_v = wp.zeros((B, NDOF), **rg)
        # CUDA graphs bake addresses, so grad seeds/results need fixed staging
        self.st_gq = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_gv = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_dq = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_dv = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_dc = wp.zeros((B, OMX_NU), dtype=float, device=dev)
        self.st_df = wp.zeros((B, 3), dtype=float, device=dev)

        # capture is illegal on the legacy default stream, so use a side stream
        self._side = torch.cuda.Stream(device=self.torch_device)
        self._wp_side = wp.stream_from_torch(self._side)
        self._capture = (os.environ.get("NEWTON_TORCH_CAPTURE", "1") == "1"
                         if capture is None else capture)
        self._graph_fwd = None      # None = not built yet, False = capture failed
        self._graph_bwd = None

    # warp work runs on the side stream, fenced against the caller's stream
    def _on_side_stream(self):
        backend = self

        class _Scope:
            def __enter__(self):
                self.cur = torch.cuda.current_stream(backend.torch_device)
                backend._side.wait_stream(self.cur)
                self.ws = wp.ScopedStream(backend._wp_side)
                self.ws.__enter__()

            def __exit__(self, *exc):
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
            body()                       # warmup: lazy allocations must land before capture
            try:
                with wp.ScopedCapture() as cap:
                    body()
                g = cap.graph
            except Exception as e:
                print(f"[newton_torch] {which} graph capture failed ({e}); eager fallback")
                g = False
            if which == "fwd":
                self._graph_fwd = g
            else:
                self._graph_bwd = g
        if g:
            wp.capture_launch(g)
        else:
            body()

    def _fwd_body(self):
        B = self.B
        wp.launch(_scatter_state, dim=(B, NDOF), inputs=[self.st_q, self.st_v],
                  outputs=[self.s0.joint_q, self.s0.joint_qd])
        wp.launch(_set_ctrl, dim=(B, NDOF), inputs=[self.st_c],
                  outputs=[self.control.joint_f])
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        for _ in range(self.n_sub):
            wp.launch(_fill_bodyf, dim=(B, NBODY), inputs=[self.st_f],
                      outputs=[self.s0.body_f])
            self.solver.step(self.s0, self.s1, self.control, None, self.dt)
            self.s0, self.s1 = self.s1, self.s0

        wp.launch(_gather_state, dim=(B, NDOF),
                  inputs=[self.s0.joint_q, self.s0.joint_qd],
                  outputs=[self.st_qo, self.st_vo])

    def _forward_raw(self, q_t, v_t, c_t, f_t, qo_t, vo_t):
        with self._on_side_stream():
            wp.copy(self.st_q, wp.from_torch(q_t))
            wp.copy(self.st_v, wp.from_torch(v_t))
            wp.copy(self.st_c, wp.from_torch(c_t))
            wp.copy(self.st_f, wp.from_torch(f_t))
            self._run("fwd", self._fwd_body)
            wp.copy(wp.from_torch(qo_t), self.st_qo)
            wp.copy(wp.from_torch(vo_t), self.st_vo)

    def _bwd_body(self):
        """Backward as a fixed launch sequence over persistent staging, so it
        can be graph-captured. tape.zero() at the end keeps grads clean for
        the next call."""
        B = self.B
        for a in (self.in_q, self.in_v, self.in_c, self.in_f,
                  self.out_q, self.out_v, self.control.joint_f):
            if a.grad is not None:
                a.grad.zero_()

        tape = wp.Tape()
        with tape:
            s = self.chain[0]
            wp.launch(_scatter_state, dim=(B, NDOF), inputs=[self.in_q, self.in_v],
                      outputs=[s.joint_q, s.joint_qd])
            wp.launch(_set_ctrl, dim=(B, NDOF), inputs=[self.in_c],
                      outputs=[self.control.joint_f])
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)
            for k in range(self.n_sub):
                s_in, s_out = self.chain[k], self.chain[k + 1]
                wp.launch(_fill_bodyf, dim=(B, NBODY), inputs=[self.in_f],
                          outputs=[s_in.body_f])
                self.solver.step(s_in, s_out, self.control, None, self.dt)
            s_fin = self.chain[self.n_sub]
            wp.launch(_gather_state, dim=(B, NDOF),
                      inputs=[s_fin.joint_q, s_fin.joint_qd],
                      outputs=[self.out_q, self.out_v])

        wp.copy(self.out_q.grad, self.st_gq)
        wp.copy(self.out_v.grad, self.st_gv)
        tape.backward()
        wp.launch(_copy2d, dim=(B, NDOF), inputs=[self.in_q.grad], outputs=[self.st_dq])
        wp.launch(_copy2d, dim=(B, NDOF), inputs=[self.in_v.grad], outputs=[self.st_dv])
        wp.launch(_copy2d, dim=(B, OMX_NU), inputs=[self.in_c.grad], outputs=[self.st_dc])
        wp.launch(_copy2d, dim=(B, 3), inputs=[self.in_f.grad], outputs=[self.st_df])
        tape.zero()

    def _backward_raw(self, q_t, v_t, c_t, f_t, gq_t, gv_t):
        dq = torch.empty_like(q_t)
        dv = torch.empty_like(v_t)
        dc = torch.empty_like(c_t)
        df = torch.empty_like(f_t)
        with self._on_side_stream():
            wp.copy(self.in_q, wp.from_torch(q_t))
            wp.copy(self.in_v, wp.from_torch(v_t))
            wp.copy(self.in_c, wp.from_torch(c_t))
            wp.copy(self.in_f, wp.from_torch(f_t))
            wp.copy(self.st_gq, wp.from_torch(gq_t))
            wp.copy(self.st_gv, wp.from_torch(gv_t))
            self._run("bwd", self._bwd_body)
            wp.copy(wp.from_torch(dq), self.st_dq)
            wp.copy(wp.from_torch(dv), self.st_dv)
            wp.copy(wp.from_torch(dc), self.st_dc)
            wp.copy(wp.from_torch(df), self.st_df)
        return dq, dv, dc, df

    # public API
    def step(self, qpos, qvel, ctrl, xfrc, route_force: bool = True):
        """Differentiable control step. External force enters as tau_ext = J^T f
        on the arm joints; the warp sim itself runs force-free.
        route_force=False skips the force route (implicit-coupling callers)."""
        if route_force:
            tau = tau_ext_torch(qpos[:, :4], xfrc)
            ctrl = ctrl + torch.nn.functional.pad(tau, (0, 1)).to(ctrl.dtype)
        zf = torch.zeros_like(xfrc)
        return _NewtonStepFn.apply(self, qpos, qvel, ctrl, zf)
