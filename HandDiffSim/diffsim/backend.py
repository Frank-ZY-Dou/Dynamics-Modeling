"""MuJoCo Warp backend for the simplified AmazingHand: one batched control step with gradients.

Interface (same shape as the NeuralActuator backends):

    step(qpos (B,12), qvel (B,12), ctrl (B,8), xfrc (B,4,3) or None) -> (qpos', qvel')

One control step is `n_sub` MuJoCo substeps at the model timestep. `ctrl` is either a joint torque on the
eight driven hinges (abd, mcp of each finger; "torque" mode, applied through qfrc_applied) or a target angle
for the model's own position servos ("position" mode). `xfrc` is a world-frame force at each fingertip.

Gradients. MuJoCo Warp (3.12) compiles its kernels without adjoints, so a Warp tape cannot be run through
mjw.step. Instead the backward pass re-runs the same step on B x (1 + 2 n_in) worlds with every input
perturbed by +-h and forms the step Jacobian by central differences on the GPU; the vector-Jacobian products
are then a batched matmul. For this smooth, contact-free system the truncation error is O(h^2) and the
result is exact to float32 rounding. Both passes are captured as CUDA graphs after a warm-up call.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from .paths import serial_model_xml

NQ = 12          # abd, mcp, pip per finger
NV = 12
NU = 8           # abd, mcp per finger
NTIP = 4
FINGERS = (1, 2, 3, 4)


def load_hand_model(side: str = "right", gravity: bool = True, actuation: str = "torque") -> mujoco.MjModel:
    """Plain MuJoCo model of the simplified hand, prepared for the backend."""
    mjm = mujoco.MjModel.from_xml_path(str(serial_model_xml(side)))
    if not gravity:
        mjm.opt.gravity[:] = 0.0
    if actuation == "torque":
        # the XML carries position servos (kp = 50); silence them so ctrl-free torque drive is possible
        mjm.actuator_gainprm[:, :] = 0.0
        mjm.actuator_biasprm[:, :] = 0.0
    elif actuation != "position":
        raise ValueError(actuation)
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return mjm


class _HandStepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, backend, q, v, c, f):
        ctx.backend = backend
        ctx.save_for_backward(q, v, c, f)
        q_, v_, c_, f_ = (t.detach().contiguous().float() for t in (q, v, c, f))
        qo, vo = backend._forward_raw(q_, v_, c_, f_)
        return qo, vo

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_q, g_v):
        q, v, c, f = ctx.saved_tensors
        dq, dv, dc, df = ctx.backend._backward_raw(
            q.detach().contiguous().float(), v.detach().contiguous().float(),
            c.detach().contiguous().float(), f.detach().contiguous().float(),
            g_q.contiguous().float(), g_v.contiguous().float())
        return None, dq, dv, dc, (df if ctx.needs_input_grad[4] else None)


class HandWarpBackend:
    name = "mjwarp_fd"

    def __init__(self, batch_size: int, data_dt: float, n_sub: int, side: str = "right", gravity: bool = True,
                 actuation: str = "torque", device: str = "cuda", capture: bool | None = None,
                 h_q: float = 1e-3, h_v: float = 1e-3, h_u: float = 1e-3, h_f: float = 1e-2,
                 grad_xfrc: bool = False):
        import mujoco_warp as mjw
        self.mjw = mjw
        self.B = int(batch_size)
        self.n_sub = int(n_sub)
        self.data_dt = float(data_dt)
        self.actuation = actuation
        self.torch_device = torch.device(device)
        if self.torch_device.type != "cuda":
            raise ValueError("HandWarpBackend needs a CUDA device")
        if self.torch_device.index is None:
            self.torch_device = torch.device("cuda", torch.cuda.current_device())
        wp.init()
        self.wp_device = wp.device_from_torch(self.torch_device)

        self.mjm = load_hand_model(side, gravity, actuation)
        self.mjm.opt.timestep = self.data_dt / self.n_sub
        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)
        assert self.mjm.nq == NQ and self.mjm.nv == NV and self.mjm.nu == NU
        # driven dofs in ctrl order: f1_abd, f1_mcp, f2_abd, ...
        self.dof_u = np.array([self.mjm.jnt_dofadr[mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, f"f{n}_{j}")]
                               for n in FINGERS for j in ("abd", "mcp")])
        self.tip_body = np.array([self.mjm.site_bodyid[mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, f"tip{n}")]
                                  for n in FINGERS])
        self.tip_site = np.array([mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, f"tip{n}") for n in FINGERS])

        # finite-difference layout: inputs x = [qpos(12), qvel(12), ctrl(8)] (+ xfrc(12) when grad_xfrc)
        self.grad_xfrc = bool(grad_xfrc)
        self.n_in = NQ + NV + NU + (NTIP * 3 if self.grad_xfrc else 0)
        self.h = torch.cat([torch.full((NQ,), h_q), torch.full((NV,), h_v), torch.full((NU,), h_u)]
                           + ([torch.full((NTIP * 3,), h_f)] if self.grad_xfrc else [])).to(self.torch_device)
        self.W = self.B * (1 + 2 * self.n_in)

        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(self.mjm)
            self.d_fwd = mjw.put_data(self.mjm, self.mjd, nworld=self.B)
            self.d_jac = mjw.put_data(self.mjm, self.mjd, nworld=self.W)
        # staging tensors with fixed addresses; Warp aliases them zero-copy for the Data uploads
        dev, nb = self.torch_device, self.mjm.nbody
        self.st = {"fwd": self._staging(self.B, nb), "jac": self._staging(self.W, nb)}
        self._capture = (os.environ.get("HAND_WARP_CAPTURE", "1") == "1") if capture is None else bool(capture)
        self._graph = {"fwd": None, "jac": None}
        self.dtype = torch.float32

    # ------------------------------------------------------------------ warp <-> torch plumbing
    def _staging(self, n, nbody):
        dev = self.torch_device
        return dict(q=torch.zeros(n, NQ, device=dev), v=torch.zeros(n, NV, device=dev), qfrc=torch.zeros(n, NV, device=dev),
                    ctrl=torch.zeros(n, NU, device=dev), xfrc=torch.zeros(n, nbody, 6, device=dev))

    def _cur_stream(self):
        """Warp handle of torch's current stream: all eager Warp work runs there, so no cross-stream fences are needed."""
        return wp.stream_from_torch(torch.cuda.current_stream(self.torch_device))

    def _stage(self, st, q, v, c, f):
        """Torch side: fill the staging tensors from the inputs (any shape of leading batch = len(q))."""
        st["q"].copy_(q)
        st["v"].copy_(v)
        st["qfrc"].zero_()
        st["ctrl"].zero_()
        if self.actuation == "torque":
            st["qfrc"][:, self.dof_u] = c
        else:
            st["ctrl"].copy_(c)
        # fingertip forces as body wrenches at the distal bodies (force only; the tip site sits within a few mm of
        # the distal body origin, the moment arm is dropped)
        st["xfrc"].zero_()
        st["xfrc"][:, self.tip_body, :3] = f

    def _upload(self, d, st):
        """Warp side: staging tensors -> mujoco_warp Data, on torch's current stream."""
        with wp.ScopedStream(self._cur_stream()):
            wp.copy(d.qpos, wp.from_torch(st["q"]))
            wp.copy(d.qvel, wp.from_torch(st["v"]))
            wp.copy(d.qfrc_applied, wp.from_torch(st["qfrc"]))
            wp.copy(d.ctrl, wp.from_torch(st["ctrl"]))
            wp.copy(d.xfrc_applied, wp.from_torch(st["xfrc"], dtype=wp.spatial_vector))
            d.qacc_warmstart.zero_()
            d.time.zero_()

    def _substeps(self, d):
        for _ in range(self.n_sub):
            self.mjw.step(self.m, d)

    def _ensure_graph(self, which, d):
        """Compile kernels, allocate, and capture the substep sequence once; the Data state is scratch here."""
        if not self._capture or self._graph[which] is not None:
            return
        with wp.ScopedStream(self._cur_stream()):
            self._substeps(d)                                  # warm-up: lazy compilation and allocation
        torch.cuda.synchronize(self.torch_device)
        try:
            # capture on Warp's own stream (mujoco_warp allocates small buffers inside step, which Warp's
            # stream-ordered allocator only supports on its streams); replay happens on torch's current stream
            with wp.ScopedCapture(device=self.wp_device) as cap:
                self._substeps(d)
            self._graph[which] = cap.graph
        except Exception as e:
            print(f"[hand_warp] {which}: CUDA graph capture failed ({e}); running eagerly")
            self._graph[which] = False
        torch.cuda.synchronize(self.torch_device)

    def _run(self, which, d):
        g = self._graph[which]
        if self._capture and g:
            wp.capture_launch(g, stream=self._cur_stream())
        else:
            with wp.ScopedStream(self._cur_stream()):
                self._substeps(d)

    def _download(self, d, n):
        with wp.ScopedStream(self._cur_stream()):
            q = wp.to_torch(d.qpos).clone()
            v = wp.to_torch(d.qvel).clone()
        return q.reshape(n, NQ), v.reshape(n, NV)

    # ------------------------------------------------------------------ forward / backward
    def _forward_raw(self, q, v, c, f):
        self._ensure_graph("fwd", self.d_fwd)
        self._stage(self.st["fwd"], q, v, c, f)
        self._upload(self.d_fwd, self.st["fwd"])
        self._run("fwd", self.d_fwd)
        return self._download(self.d_fwd, self.B)

    def _perturbed_inputs(self, q, v, c, f):
        """(W, ...) inputs: for each sample, the base point followed by +h and -h along every input."""
        B, n = self.B, self.n_in
        x = torch.cat([q, v, c] + ([f.reshape(B, -1)] if self.grad_xfrc else []), dim=1)     # (B, n)
        eye = torch.eye(n, device=self.torch_device) * self.h[None, :]                         # (n, n)
        pert = torch.cat([torch.zeros(1, n, device=self.torch_device), eye, -eye], dim=0)       # (1+2n, n)
        X = (x[:, None, :] + pert[None, :, :]).reshape(self.W, n)
        qj, vj, cj = X[:, :NQ], X[:, NQ:NQ + NV], X[:, NQ + NV:NQ + NV + NU]
        if self.grad_xfrc:
            fj = X[:, NQ + NV + NU:].reshape(self.W, NTIP, 3)
        else:
            fj = f[:, None].expand(B, 1 + 2 * n, NTIP, 3).reshape(self.W, NTIP, 3)
        return qj.contiguous(), vj.contiguous(), cj.contiguous(), fj.contiguous()

    def step_jacobian(self, q, v, c, f):
        """Central-difference Jacobians of (qpos', qvel') w.r.t. the inputs: (B, 12, n_in) each, plus the base outputs."""
        self._ensure_graph("jac", self.d_jac)
        qj, vj, cj, fj = self._perturbed_inputs(q, v, c, f)
        self._stage(self.st["jac"], qj, vj, cj, fj)
        self._upload(self.d_jac, self.st["jac"])
        self._run("jac", self.d_jac)
        Qo, Vo = self._download(self.d_jac, self.W)
        n = self.n_in
        Qo = Qo.reshape(self.B, 1 + 2 * n, NQ)
        Vo = Vo.reshape(self.B, 1 + 2 * n, NV)
        Jq = (Qo[:, 1:1 + n] - Qo[:, 1 + n:]) / (2 * self.h[None, :, None])     # (B, n, 12)
        Jv = (Vo[:, 1:1 + n] - Vo[:, 1 + n:]) / (2 * self.h[None, :, None])
        return Jq.transpose(1, 2), Jv.transpose(1, 2), Qo[:, 0], Vo[:, 0]

    def _backward_raw(self, q, v, c, f, g_q, g_v):
        Jq, Jv, _, _ = self.step_jacobian(q, v, c, f)
        gx = torch.einsum("bj,bji->bi", g_q, Jq) + torch.einsum("bj,bji->bi", g_v, Jv)   # (B, n_in)
        dq, dv, dc = gx[:, :NQ], gx[:, NQ:NQ + NV], gx[:, NQ + NV:NQ + NV + NU]
        df = gx[:, NQ + NV + NU:].reshape(self.B, NTIP, 3) if self.grad_xfrc else torch.zeros_like(f)
        return dq, dv, dc, df

    # ------------------------------------------------------------------ public API
    def step(self, qpos, qvel, ctrl, xfrc=None):
        """Differentiable control step; gradients w.r.t. qpos, qvel, ctrl (and xfrc when grad_xfrc=True)."""
        if xfrc is None:
            xfrc = torch.zeros(self.B, NTIP, 3, device=self.torch_device)
        return _HandStepFn.apply(self, qpos, qvel, ctrl, xfrc)

    def reference_step(self, qpos, qvel, ctrl, xfrc=None):
        """Same control step in plain MuJoCo (float64, CPU), for parity checks."""
        q = np.asarray(qpos, np.float64); v = np.asarray(qvel, np.float64); c = np.asarray(ctrl, np.float64)
        f = None if xfrc is None else np.asarray(xfrc, np.float64)
        out_q, out_v = np.zeros_like(q), np.zeros_like(v)
        for b in range(q.shape[0]):
            d = self.mjd
            mujoco.mj_resetData(self.mjm, d)
            d.qpos[:] = q[b]; d.qvel[:] = v[b]
            if self.actuation == "torque":
                d.qfrc_applied[self.dof_u] = c[b]
            else:
                d.ctrl[:] = c[b]
            if f is not None:
                d.xfrc_applied[self.tip_body, :3] = f[b]
            for _ in range(self.n_sub):
                mujoco.mj_step(self.mjm, d)
            out_q[b], out_v[b] = d.qpos, d.qvel
        return out_q, out_v
