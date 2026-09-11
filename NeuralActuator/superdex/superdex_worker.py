"""Physics worker of the SuperDex backend: one process, a few lanes.

Runs in a Python that has the differentiable SuperDex fork installed (see
requirements.txt); the trainer's own process needs only torch. Started by
superdex_backend_torch.SuperDexBackendTorch, which sends numpy arrays over a
multiprocessing connection:

    ("fwd", q (L,6), v (L,6), tau (L,6))        -> (q_out (L,6), v_out (L,6))
    ("bwd", q, v, tau, g_q, g_v)                 -> (dq, dv, dtau, max adjoint residual)
    ("close",)

A control step is n_sub Backward-Euler substeps of the fork's engine on one
scene per lane, contact-free, with the torques entering as external forces on
the joints (constant over the control step). The step's output velocity is the
Backward-Euler pose difference over the last substep, (q_K - q_{K-1}) / dt: the
engine's own joint-velocity reading differs from it only by a term of order
dt^2 (below 1e-6 rad/s at the speeds of the training data), and with this
definition a gradient with respect to the output velocity is a gradient with
respect to two poses, which the fork's adjoint seeds exactly. The backward
recomputes the substeps under the fork's checkpointed adjoint sweep
(DifferentiableRollout) with those pose adjoints and returns the gradients with
respect to the input pose, the input velocity and the torque.

Each worker spins on its connection between messages (SUPERDEX_SPIN=0 to block
instead) and reports its engine time with SUPERDEX_PROFILE=1.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

os.environ.setdefault("SUPERDEX_PRECISION", "double")

from multiprocessing.connection import Client

import numpy as np

import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics import diffsim
from superdex.physics.diffsim_rollout import DifferentiableRollout

NDOF = 6            # 4 arm revolute joints + 2 gripper prismatic joints
JOINT_DAMPING = 1.0   # the MJCF joint damping (robot/omx_newton.xml), viscous friction here
JOINT_ARMATURE = 0.1  # the MJCF armature, the engine's joint inertia
# the adjoint assumes a converged Newton solve; SUPERDEX_SOLVER_TOL tightens it
# (a finite-difference check of the gradients sees the solve's tolerance as noise)
SOLVER_TOL = float(os.environ.get("SUPERDEX_SOLVER_TOL", "1e-10"))
SOLVER_MAX_ITER = 200


class _PoseFunctional:
    """The loss g . q on the joint pose, the seed of an adjoint sweep."""

    def __init__(self, actor, g):
        self.actor = actor
        self.g = np.ascontiguousarray(g, dtype=np.float64)

    def value(self):
        q = np.zeros(NDOF)
        self.actor.get_articulated_pose(q)
        return float(self.g @ q)

    def accumulate_output_grad(self):
        diffsim.get_articulated_pose_backward(self.actor, self.g.copy())


class Lane:
    def __init__(self, bot_path: str, dt: float, n_sub: int, index: int):
        self.bot_path, self.dt, self.n_sub, self.index = bot_path, dt, n_sub, index
        self.dofs = np.arange(NDOF, dtype=np.int32)
        self._build()

    def _build(self):
        self.scene = physics.create_scene(f"omx_lane{self.index}")
        self.scene.set_gravity([0.0, 0.0, -9.81])
        self.context = robotics.create_context()
        prefab = robotics.load_bot_prefab_from_file(self.bot_path)
        for i in range(len(prefab.links)):  # contact-free, as the other backends
            link = prefab.links[i]
            link.collider_type = physics.ColliderType.NONE
            prefab.links[i] = link
        bot = robotics.create_bot(self.scene, prefab, self.context)
        self.actor = bot.get_articulated_actor()
        if self.actor.get_num_dofs() != NDOF:
            raise RuntimeError(f"expected {NDOF} dofs, got {self.actor.get_num_dofs()}")
        diffsim.make_scene_differentiable(self.scene)
        params = self.scene.get_solver_params()
        solver = params.non_linear_solver
        solver.abs_tol = solver.rel_tol = SOLVER_TOL
        solver.max_iter = SOLVER_MAX_ITER
        params.non_linear_solver = solver
        self.scene.set_solver_params(params)
        # per-joint parameter spans (the fixed joints of the model included)
        n_joints = len(prefab.joints)
        friction = []
        for _ in range(n_joints):
            fp = physics.ArticulatedJointFrictionParams()
            fp.viscous = JOINT_DAMPING
            friction.append(fp)
        self.actor.set_articulated_joint_friction_params(friction)
        self.actor.set_articulated_joint_inertia_params(np.full(n_joints, JOINT_ARMATURE))
        self.driver = DifferentiableRollout(self.scene, dt=self.dt, num_steps=self.n_sub)
        entries = self.driver.entries
        if len(entries) != 1 or list(entries[0].force_dofs) != list(range(NDOF)):
            raise RuntimeError("expected one articulated actor with external forces on all six dofs")

    def _rebuild(self):
        physics.destroy_scene(self.scene)
        self._build()

    def _set_state(self, q, v):
        self.actor.set_articulated_pose_from_joints(np.ascontiguousarray(q, dtype=np.float64))
        self.actor.set_articulated_joint_velocities(np.ascontiguousarray(v, dtype=np.float64))

    def _pose(self):
        q = np.zeros(NDOF)
        self.actor.get_articulated_pose(q)
        return q

    def forward(self, q, v, tau):
        self._set_state(q, v)
        tau = np.ascontiguousarray(tau, dtype=np.float64)
        q_prev = self._pose()
        for _ in range(self.n_sub):
            q_prev = self._pose()
            self.actor.set_external_forces_on_dofs(self.dofs, tau)
            self.scene.step(self.dt)
        q_out = self._pose()
        return q_out, (q_out - q_prev) / self.dt

    def backward(self, q, v, tau, g_q, g_v):
        self._set_state(q, v)
        tau = np.ascontiguousarray(tau, dtype=np.float64)
        g_q = np.asarray(g_q, dtype=np.float64)
        g_v = np.asarray(g_v, dtype=np.float64)
        # v_out = (q_K - q_{K-1}) / dt: its adjoint lands on the last pose and the one before
        seed_last = _PoseFunctional(self.actor, g_q + g_v / self.dt)
        seed_previous = _PoseFunctional(self.actor, -g_v / self.dt)

        def apply_inputs(step):
            self.actor.set_external_forces_on_dofs(self.dofs, tau)

        def step_losses(step):
            return [seed_previous] if step == self.n_sub - 2 else []

        result = self.driver.run(apply_inputs=apply_inputs, terminal_losses=[seed_last],
                                 step_losses=step_losses)
        grads = next(iter(result.gradients.values()))
        dq = np.array(grads.initial_pose, dtype=np.float64)
        dv = np.array(grads.initial_velocity, dtype=np.float64)
        if self.n_sub == 1:  # q_{K-1} is the input pose itself
            dq -= g_v / self.dt
        dtau = grads.external_forces.sum(axis=1)  # constant over the substeps
        return dq, dv, dtau, float(result.max_adjoint_residual)


def main():
    address, authkey = sys.argv[1], bytes.fromhex(sys.argv[2])
    worker_index = int(sys.argv[3])
    bot_path, dt, n_sub, n_lanes = sys.argv[4], float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    physics.initialize(num_worker_threads=0)
    lanes = [Lane(bot_path, dt, n_sub, i) for i in range(n_lanes)]
    conn = Client(address, authkey=authkey)
    conn.send(("ready", worker_index, NDOF))
    nan_row = np.full(NDOF, np.nan)
    # SUPERDEX_PROFILE=1: the engine time of this worker's calls (excluding the
    # wait for the next message), reported on stderr every 200 backward calls
    profile = os.environ.get("SUPERDEX_PROFILE") == "1"
    # A worker computes for a few milliseconds per message and then waits for the
    # trainer's network pass and the other lanes. Blocking in recv() lets a
    # frequency-scaling governor clock the core down between messages, which made
    # the next call two to three times slower (measured with schedutil: a 15 ms idle
    # gap turned a 6.5 ms backward into 18 ms), so by default the worker spins on
    # the connection instead, holding its core. SUPERDEX_SPIN=0 blocks.
    spin = os.environ.get("SUPERDEX_SPIN", "1") == "1"
    stats = dict(fwd=0.0, nf=0, bwd=0.0, nb=0, wait=0.0)
    t_idle = time.perf_counter()
    while True:
        if spin:
            while not conn.poll(0):
                pass
        msg = conn.recv()
        t_start = time.perf_counter()
        stats["wait"] += t_start - t_idle
        kind = msg[0]
        if kind == "close":
            break
        if kind == "ping":
            conn.send(("pong",))
        elif kind == "fwd":
            _, q, v, tau = msg
            q_out, v_out = np.empty_like(q), np.empty_like(v)
            for i, lane in enumerate(lanes):
                try:
                    q_out[i], v_out[i] = lane.forward(q[i], v[i], tau[i])
                    if not (np.all(np.isfinite(q_out[i])) and np.all(np.isfinite(v_out[i]))):
                        raise FloatingPointError("non-finite state")
                except Exception:
                    # the trainer's state surgery handles a lane that diverged; the scene is
                    # rebuilt so the next step starts from a clean engine state
                    traceback.print_exc(file=sys.stderr)
                    q_out[i], v_out[i] = nan_row, nan_row
                    lane._rebuild()
            conn.send((q_out, v_out))
            stats["fwd"] += time.perf_counter() - t_start
            stats["nf"] += 1
        elif kind == "bwd":
            _, q, v, tau, g_q, g_v = msg
            dq, dv, dtau = np.empty_like(q), np.empty_like(v), np.empty_like(tau)
            max_residual = 0.0
            for i, lane in enumerate(lanes):
                try:
                    dq[i], dv[i], dtau[i], residual = lane.backward(q[i], v[i], tau[i], g_q[i], g_v[i])
                    max_residual = max(max_residual, residual)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    dq[i], dv[i], dtau[i] = nan_row, nan_row, nan_row  # the optimizer skips the update
                    lane._rebuild()
            conn.send((dq, dv, dtau, max_residual))
            stats["bwd"] += time.perf_counter() - t_start
            stats["nb"] += 1
            if profile and stats["nb"] % 200 == 0:
                print(f"[superdex worker {worker_index}] fwd {stats['fwd'] / max(stats['nf'], 1) * 1e3:.2f} ms/call "
                      f"({stats['nf']}), bwd {stats['bwd'] / stats['nb'] * 1e3:.2f} ms/call ({stats['nb']}), "
                      f"waiting {stats['wait'] / (stats['nf'] + stats['nb']) * 1e3:.2f} ms/call",
                      file=sys.stderr, flush=True)
        else:
            raise ValueError(f"unknown message {kind!r}")
        t_idle = time.perf_counter()
    for lane in lanes:
        physics.destroy_scene(lane.scene)
    physics.shutdown()


if __name__ == "__main__":
    main()
