"""MJX reference backend.

Same solver settings as the public trainer: timestep = data_dt / sim_step_size,
Newton solver with 1 iteration, contact disabled. Precision follows the process
JAX config: run with JAX_ENABLE_X64=1 for the f64 reference, without it for f32.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from backends.base import OMX_NQ, OMX_NU, ROBOT_XML


class MJXBackend:
    differentiable = True
    name = "mjx"

    def __init__(self, batch_size: int, data_dt: float, sim_step_size: int):
        self.B = batch_size
        xml = os.path.join(os.path.dirname(__file__), "..", ROBOT_XML)
        mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(xml))

        # solver setup from train_actuator_diffsim.py:584-595
        mj_model.opt.timestep = data_dt / sim_step_size
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 0
        mj_model.opt.tolerance = 0
        mj_model.opt.ls_tolerance = 0
        mj_model.opt.noslip_iterations = 0
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        assert mj_model.nq == OMX_NQ and mj_model.nu == OMX_NU, (mj_model.nq, mj_model.nu)
        self.ee_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_target"))
        self.nbody = mj_model.nbody
        self.mj_model = mj_model
        self.mjx_model = mjx.put_model(mj_model)
        self._data0 = mjx.make_data(self.mjx_model)
        n_sub = sim_step_size

        def _one(q, v, ctrl5, f3):
            d = self._data0.replace(qpos=q, qvel=v)
            ctrl = jnp.zeros(OMX_NU, dtype=q.dtype).at[:].set(ctrl5)
            d = d.replace(ctrl=ctrl)
            xfrc = jnp.zeros((self.nbody, 6), dtype=q.dtype).at[self.ee_id, :3].set(f3)
            d = d.replace(xfrc_applied=xfrc)
            d = jax.lax.fori_loop(0, n_sub, lambda i, dd: mjx.step(self.mjx_model, dd), d)
            return d.qpos, d.qvel

        self._step = jax.jit(jax.vmap(_one))
        self.dtype = self._data0.qpos.dtype

    def step(self, qpos, qvel, ctrl, xfrc):
        return self._step(qpos, qvel, ctrl, xfrc)
