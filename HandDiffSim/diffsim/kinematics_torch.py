"""Differentiable forward kinematics of the simplified hand in torch, from the identified geometry JSON.

Same conventions as serial_hand/ah_serial/kinematics.py (MuJoCo: a body's hinge rotates it about an axis given
in the body frame, around the joint anchor). qpos layout: f1_abd, f1_mcp, f1_pip, f2_abd, ... (12,).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .paths import identified_params


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis (3,) unit, angle (...,) -> (..., 3, 3)."""
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1.0 - c
    x, y, z = axis
    R = torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)
    return R


class HandKinematicsTorch:
    def __init__(self, side: str = "right", device="cuda", dtype=torch.float32):
        with open(identified_params(side)) as fh:
            params = json.load(fh)
        self.device, self.dtype = torch.device(device), dtype
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), dtype=dtype, device=self.device)  # noqa: E731
        self.f = []
        for n in ("1", "2", "3", "4"):
            g = params["fingers"][n]["geometry"]
            RG, RL, RD = (_quat_to_mat(np.asarray(g[k + "_quat"])) for k in "GLD")
            pG, pL, pD = (np.asarray(g[k + "_pos"]) for k in "GLD")
            self.f.append(dict(
                G_pos=t(pG), G_R=t(RG),
                L_pos=t(RG.T @ (pL - pG)), L_R=t(RG.T @ RL),
                D_pos=t(RL.T @ (pD - pL)), D_R=t(RL.T @ RD),
                abd=(t(g["abd_jpos"]), t(np.asarray(g["abd_axis"]) / np.linalg.norm(g["abd_axis"]))),
                mcp=(t(g["mcp_jpos"]), t(np.asarray(g["mcp_axis"]) / np.linalg.norm(g["mcp_axis"]))),
                pip=(t(g["pip_jpos"]), t(np.asarray(g["pip_axis"]) / np.linalg.norm(g["pip_axis"]))),
                tip_pos=t(g["tip_pos"]), tip_R=t(_quat_to_mat(np.asarray(g["tip_quat"])))))
            self.pip_coef = [t(params["fingers"][k]["fits"]["coupling_pip_of_mcp"]["coef_ascending"]) for k in ("1", "2", "3", "4")]

    @staticmethod
    def _hinge_body(p_par, R_par, rel_pos, rel_R, jpos, jaxis, q):
        """p_par (B,3), R_par (B,3,3); rel_pos (3,), rel_R (3,3); q (B,) -> body pose (B,3), (B,3,3)."""
        R_pre = R_par @ rel_R
        p_pre = p_par + (R_par @ rel_pos)
        anchor = p_pre + (R_pre @ jpos)
        R = R_pre @ rot_axis_angle(jaxis, q)
        return anchor - (R @ jpos), R

    def pip_of_mcp(self, mcp: torch.Tensor) -> torch.Tensor:
        """(B,4) -> (B,4) fingertip joint angle from the knuckle angle (identified polynomial)."""
        out = []
        for i in range(4):
            c = self.pip_coef[i]
            x = mcp[:, i]
            out.append(sum(c[k] * x ** k for k in range(len(c))))
        return torch.stack(out, dim=1)

    def forward(self, qpos: torch.Tensor):
        """qpos (B,12) -> tips (B,4,3), tip rotations (B,4,3,3), in the palm (= world) frame."""
        B = qpos.shape[0]
        I = torch.eye(3, device=self.device, dtype=self.dtype).expand(B, 3, 3)
        o = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        tips, rots = [], []
        for i, g in enumerate(self.f):
            abd, mcp, pip = qpos[:, 3 * i], qpos[:, 3 * i + 1], qpos[:, 3 * i + 2]
            pG, RG = self._hinge_body(o, I, g["G_pos"], g["G_R"], *g["abd"], abd)
            pL, RL = self._hinge_body(pG, RG, g["L_pos"], g["L_R"], *g["mcp"], mcp)
            pD, RD = self._hinge_body(pL, RL, g["D_pos"], g["D_R"], *g["pip"], pip)
            tips.append(pD + RD @ g["tip_pos"])
            rots.append(RD @ g["tip_R"])
        return torch.stack(tips, 1), torch.stack(rots, 1)

    __call__ = forward
