"""Position-servo torque law of the simplified hand and its inverse.

The upstream model drives every abd/mcp hinge with a MuJoCo position actuator, tau = kp (cmd - q) with kp = 50
and a critically damped kv derived from dampratio. Written out as a torch function it is the simplest
"command -> torque" model, and solving it for cmd gives "torque -> command": the servo target that produces a
given joint torque at the current state.
"""
from __future__ import annotations

import torch


class ServoPD:
    def __init__(self, kp: float = 50.0, kd: float = 0.0, tau_max: float | None = None):
        self.kp, self.kd, self.tau_max = float(kp), float(kd), tau_max

    def torque(self, cmd: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """cmd, q, v (B,8) -> tau (B,8)."""
        tau = self.kp * (cmd - q) - self.kd * v
        if self.tau_max is not None:
            tau = tau.clamp(-self.tau_max, self.tau_max)
        return tau

    def command(self, tau: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Inverse: the command that yields torque tau at state (q, v). Differentiable in all three."""
        return q + (tau + self.kd * v) / self.kp


def driven_joints(qpos: torch.Tensor) -> torch.Tensor:
    """(B,12) -> (B,8): abd and mcp of every finger, dropping the coupled pip."""
    return qpos.reshape(qpos.shape[0], 4, 3)[:, :, :2].reshape(qpos.shape[0], 8)


def with_pip(q8: torch.Tensor, pip: torch.Tensor) -> torch.Tensor:
    """(B,8) driven angles + (B,4) pip -> (B,12) qpos."""
    B = q8.shape[0]
    return torch.cat([q8.reshape(B, 4, 2), pip[:, :, None]], dim=2).reshape(B, 12)
