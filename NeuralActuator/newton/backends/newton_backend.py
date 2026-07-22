"""Newton (SolverFeatherstone) backend: batched forward stepping and gradients.

B replicated worlds in generalized coordinates; joint_q/joint_qd are flat (B*6,).
Joint torques go in through Control.joint_f. Featherstone has no equality
constraint, so the gripper coupling is emulated: split the gripper torque
half-and-half across the two slide DOFs and report their pair average
(2I qbar'' = F - 2b qbar', the axis-opposed gravity cancels).
External force enters at end_effector_target's COM via State.body_f; first 3
components are linear force at the COM, last 3 torque (newton state.py:148-153).
Gradients use jax.custom_vjp. The forward is the fast in-place path; the backward
recomputes the control step under a wp.Tape over a persistent chain of per-substep
States (no in-place overwrites on tape) and replays the adjoints. A tape only
lives for one control step, so long rollouts don't accumulate tape memory.
angular_damping=0 since Featherstone's default 0.05 has no MuJoCo counterpart.
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp
import newton

from backends.base import OMX_NQ, OMX_NU, ROBOT_XML

NDOF = OMX_NQ          # 6 dofs per world
NBODY = 7              # link2..5, gripper_left, gripper_right, end_effector_target
EE_BODY = 6            # end_effector_target index within a world (verified at build)


@wp.kernel
def _scatter_state(q2d: wp.array2d(dtype=float), v2d: wp.array2d(dtype=float),
                   joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float)):
    i, j = wp.tid()
    joint_q[i * NDOF + j] = q2d[i, j]
    joint_qd[i * NDOF + j] = v2d[i, j]


@wp.kernel
def _set_ctrl(ctrl: wp.array2d(dtype=float), joint_f: wp.array(dtype=float)):
    i, j = wp.tid()
    f = 0.0
    if j < 4:
        f = ctrl[i, j]              # tau1..tau4
    else:
        f = 0.5 * ctrl[i, 4]        # equality surrogate: half torque to each finger
    joint_f[i * NDOF + j] = f


@wp.kernel
def _fill_bodyf(xfrc: wp.array2d(dtype=float), body_f: wp.array(dtype=wp.spatial_vector)):
    """Zero body_f except the ee body, which gets the world-frame force.
    One write per element per launch, so this is safe under a tape."""
    i, j = wp.tid()
    fv = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if j == EE_BODY:
        fv = wp.spatial_vector(xfrc[i, 0], xfrc[i, 1], xfrc[i, 2], 0.0, 0.0, 0.0)
    body_f[i * NBODY + j] = fv


@wp.kernel
def _gather_state(joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float),
                  q2d: wp.array2d(dtype=float), v2d: wp.array2d(dtype=float)):
    i, j = wp.tid()
    if j < 4:
        q2d[i, j] = joint_q[i * NDOF + j]
        v2d[i, j] = joint_qd[i * NDOF + j]
    else:                       # gripper: report the pair average
        qb = 0.5 * (joint_q[i * NDOF + 4] + joint_q[i * NDOF + 5])
        vb = 0.5 * (joint_qd[i * NDOF + 4] + joint_qd[i * NDOF + 5])
        q2d[i, j] = qb
        v2d[i, j] = vb


@wp.kernel
def _copy2d(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


class NewtonBackend:
    name = "newton"

    def __init__(self, batch_size: int, data_dt: float, sim_step_size: int,
                 newton_substeps: int | None = None, differentiable: bool = True):
        self.B = batch_size
        self.differentiable = differentiable
        self.n_sub = newton_substeps if newton_substeps is not None else sim_step_size
        assert self.n_sub % 2 == 0, "need even n_sub for graph capture"
        self.dt = data_dt / self.n_sub

        xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ROBOT_XML))
        robot = newton.ModelBuilder()
        robot.add_mjcf(xml, parse_meshes=False, parse_visuals=False,
                       enable_self_collisions=False, skip_equality_constraints=True,
                       collapse_fixed_joints=False)
        scene = newton.ModelBuilder()
        scene.replicate(robot, world_count=batch_size)
        self.model = scene.finalize(requires_grad=differentiable)
        assert self.model.joint_dof_count == batch_size * NDOF, self.model.joint_dof_count
        assert self.model.body_count == batch_size * NBODY, self.model.body_count

        # Match MuJoCo's effective joint-limit stiffness. Newton's default penalty
        # (ke=1e4, kd=10) is ~40x stiffer than MuJoCo's constraint with default
        # solref=[0.02,1] (k ~ I_eff/0.02^2 ~ 250, kd ~ 2*I_eff/0.02 ~ 10 for
        # I_eff ~ armature 0.1). Overly stiff penalties poison early-training
        # gradients when the arm rides its limits.
        lim_ke = float(os.environ.get("NEWTON_LIMIT_KE", "250.0"))
        lim_kd = float(os.environ.get("NEWTON_LIMIT_KD", "10.0"))
        self.model.joint_limit_ke.fill_(lim_ke)
        self.model.joint_limit_kd.fill_(lim_kd)

        self.solver = newton.solvers.SolverFeatherstone(
            self.model, angular_damping=0.0, update_mass_matrix_interval=1)
        self.control = self.model.control()
        self.s0 = self.model.state()
        self.s1 = self.model.state()
        self.dtype = np.float32

        B, dev = self.B, self.model.device
        # persistent non-grad staging buffers for the manually CUDA-graph-captured forward
        self.st_q = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_v = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_c = wp.zeros((B, OMX_NU), dtype=float, device=dev)
        self.st_f = wp.zeros((B, 3), dtype=float, device=dev)
        self.st_qo = wp.zeros((B, NDOF), dtype=float, device=dev)
        self.st_vo = wp.zeros((B, NDOF), dtype=float, device=dev)
        self._graph = None
        if differentiable:
            # state chain for the backward tape
            self.chain = [self.model.state() for _ in range(self.n_sub + 1)]
            rg = dict(dtype=float, device=dev, requires_grad=True)
            self.in_q = wp.zeros((B, NDOF), **rg)
            self.in_v = wp.zeros((B, NDOF), **rg)
            self.in_c = wp.zeros((B, OMX_NU), **rg)
            self.in_f = wp.zeros((B, 3), **rg)
            self.out_q = wp.zeros((B, NDOF), **rg)
            self.out_v = wp.zeros((B, NDOF), **rg)

        # fast forward: no tape, CUDA-graph captured
        def _fwd_body():
            """Launch sequence over the staging buffers; capturable."""
            wp.launch(_scatter_state, dim=(B, NDOF), inputs=[self.st_q, self.st_v],
                      outputs=[self.s0.joint_q, self.s0.joint_qd])
            wp.launch(_set_ctrl, dim=(B, NDOF), inputs=[self.st_c], outputs=[self.control.joint_f])
            newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
            for _ in range(self.n_sub):
                wp.launch(_fill_bodyf, dim=(B, NBODY), inputs=[self.st_f], outputs=[self.s0.body_f])
                self.solver.step(self.s0, self.s1, self.control, None, self.dt)
                self.s0, self.s1 = self.s1, self.s0
            wp.launch(_gather_state, dim=(B, NDOF),
                      inputs=[self.s0.joint_q, self.s0.joint_qd],
                      outputs=[self.st_qo, self.st_vo])
            # even n_sub: s0/s1 swap back each call, so replays see the same buffers

        def _fwd(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                 ctrl: wp.array2d(dtype=float), xfrc: wp.array2d(dtype=float),
                 qpos_out: wp.array2d(dtype=float), qvel_out: wp.array2d(dtype=float)):
            wp.copy(self.st_q, qpos); wp.copy(self.st_v, qvel)
            wp.copy(self.st_c, ctrl); wp.copy(self.st_f, xfrc)
            if os.environ.get("NEWTON_CAPTURE", "1") == "1":
                if self._graph is None:
                    _fwd_body()                      # warmup (allocations happen here)
                    try:
                        with wp.ScopedCapture() as cap:
                            _fwd_body()
                        self._graph = cap.graph
                    except Exception as e:           # capture unsupported -> fallback
                        print(f"[newton_backend] graph capture failed ({e}); eager fallback")
                        self._graph = False
                if self._graph:
                    wp.capture_launch(self._graph)
                else:
                    _fwd_body()
            else:
                _fwd_body()
            wp.copy(qpos_out, self.st_qo); wp.copy(qvel_out, self.st_vo)

        # backward: recompute under tape
        def _bwd(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                 ctrl: wp.array2d(dtype=float), xfrc: wp.array2d(dtype=float),
                 g_q: wp.array2d(dtype=float), g_v: wp.array2d(dtype=float),
                 d_qpos: wp.array2d(dtype=float), d_qvel: wp.array2d(dtype=float),
                 d_ctrl: wp.array2d(dtype=float), d_xfrc: wp.array2d(dtype=float)):
            # stage inputs off-tape into the grad arrays
            wp.launch(_copy2d, dim=(B, NDOF), inputs=[qpos], outputs=[self.in_q])
            wp.launch(_copy2d, dim=(B, NDOF), inputs=[qvel], outputs=[self.in_v])
            wp.launch(_copy2d, dim=(B, OMX_NU), inputs=[ctrl], outputs=[self.in_c])
            wp.launch(_copy2d, dim=(B, 3), inputs=[xfrc], outputs=[self.in_f])
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

            wp.copy(self.out_q.grad, g_q)
            wp.copy(self.out_v.grad, g_v)
            tape.backward()
            wp.launch(_copy2d, dim=(B, NDOF), inputs=[self.in_q.grad], outputs=[d_qpos])
            wp.launch(_copy2d, dim=(B, NDOF), inputs=[self.in_v.grad], outputs=[d_qvel])
            wp.launch(_copy2d, dim=(B, OMX_NU), inputs=[self.in_c.grad], outputs=[d_ctrl])
            wp.launch(_copy2d, dim=(B, 3), inputs=[self.in_f.grad], outputs=[d_xfrc])
            tape.zero()

        import inspect
        kwargs = {}
        sig = inspect.signature(wp.jax_callable).parameters
        gm = os.environ.get("NEWTON_FFI_GRAPH_MODE", "")
        if gm and "graph_mode" in sig:
            from warp.jax_experimental.ffi import JaxCallableGraphMode as _GM
            kwargs["graph_mode"] = getattr(_GM, gm)
        elif "graph_compatible" in sig:
            kwargs["graph_compatible"] = os.environ.get("NEWTON_GRAPH", "1") == "1"
        self._jax_fwd = wp.jax_callable(_fwd, num_outputs=2, **kwargs)
        self._out_dims_fwd = {"qpos_out": (B, NDOF), "qvel_out": (B, NDOF)}
        if differentiable:
            self._jax_bwd = wp.jax_callable(_bwd, num_outputs=4, **kwargs)
            self._out_dims_bwd = {"d_qpos": (B, NDOF), "d_qvel": (B, NDOF),
                                  "d_ctrl": (B, OMX_NU), "d_xfrc": (B, 3)}
            self._build_custom_vjp()

    # ---------------- public API ----------------
    def step(self, qpos, qvel, ctrl, xfrc):
        """Forward-only step. External force routed as tau_ext = J^T f in JAX
        (see module notes); the warp path runs force-free."""
        import jax
        import jax.numpy as jnp
        if not hasattr(self, "_route_j"):
            def _route(q, c, f):
                c = c.at[:, :4].add(_tau_ext(q[:, :4], f).astype(c.dtype))
                return c, jnp.zeros_like(f)
            self._route_j = jax.jit(_route)
        q = jnp.asarray(qpos)
        c, zf = self._route_j(q, jnp.asarray(ctrl), jnp.asarray(xfrc))
        return self._jax_fwd(q, qvel, c, zf, output_dims=self._out_dims_fwd)

    def step_bodyf(self, qpos, qvel, ctrl, xfrc):
        """Raw body_f force route; adjoint broken in newton 1.4.0, forward-only."""
        return self._jax_fwd(qpos, qvel, ctrl, xfrc, output_dims=self._out_dims_fwd)

    def _build_custom_vjp(self):
        import jax

        @jax.custom_vjp
        def newton_step(q, v, c, f):
            out = self._jax_fwd(q, v, c, f, output_dims=self._out_dims_fwd)
            return (out[0], out[1])

        def fwd(q, v, c, f):
            out = self._jax_fwd(q, v, c, f, output_dims=self._out_dims_fwd)
            return (out[0], out[1]), (q, v, c, f)

        def bwd(res, cot):
            q, v, c, f = res
            g_q, g_v = cot
            return tuple(self._jax_bwd(q, v, c, f, g_q, g_v,
                                       output_dims=self._out_dims_bwd))

        newton_step.defvjp(fwd, bwd)
        self._newton_step_raw = newton_step

        def step_diff(q, v, c, f):
            import jax.numpy as jnp
            c = c.at[:, :4].add(_tau_ext(q[:, :4], f).astype(c.dtype))
            zf = jnp.zeros_like(f)
            return newton_step(q, v, c, zf)

        self.step_diff = step_diff

# JAX-side force routing
# Upstream Featherstone (newton 1.4.0) corrupts the adjoint of State.body_f's
# configuration dependence (in-place wrench-frame shift inside eval_rigid_tau, see
# solver_featherstone.py:451 comment). Workaround: convert the world-frame force at
# end_effector_target to joint torques tau_ext = J_v(q)^T f with a differentiable JAX
# forward kinematics of the arm chain, and feed Newton through its force-free path,
# whose adjoints are exact. ZOH within a control step (tau_ext held at step-start q).
# Chain from robot/omx_newton.xml (identity orientations, joint pos all zero):
#   world -> link2 t1=(0.012,0,0.017)   joint1 hinge z
#   link2 -> link3 t2=(0,0,0.0595)      joint2 hinge y
#   link3 -> link4 t3=(0.024,0,0.128)   joint3 hinge y
#   link4 -> link5 t4=(0.124,0,0)       joint4 hinge y
#   link5 -> ee    t5=(0.14,0,0)        (welded; gripper slides do not move the ee)


def _fk_ee_pos(q_arm):
    """(4,) arm joint angles -> (3,) world position of end_effector_target (jnp)."""
    import jax.numpy as jnp
    t1 = jnp.array([0.012, 0.0, 0.017]); t2 = jnp.array([0.0, 0.0, 0.0595])
    t3 = jnp.array([0.024, 0.0, 0.128]); t4 = jnp.array([0.124, 0.0, 0.0])
    t5 = jnp.array([0.14, 0.0, 0.0])

    def Rz(a):
        c, s = jnp.cos(a), jnp.sin(a)
        return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def Ry(a):
        c, s = jnp.cos(a), jnp.sin(a)
        return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return t1 + Rz(q_arm[0]) @ (t2 + Ry(q_arm[1]) @ (t3 + Ry(q_arm[2]) @ (t4 + Ry(q_arm[3]) @ t5)))


def _tau_ext(q_arm_b, f_b):
    """Batched tau_ext = J_v(q)^T f via VJP of the FK point. (B,4),(B,3)->(B,4)."""
    import jax

    def one(q, f):
        _, vjp = jax.vjp(_fk_ee_pos, q)
        return vjp(f)[0]

    return jax.vmap(one)(q_arm_b, f_b)
