"""
MLP-based Neural Actuator model.

Stateless feed-forward architecture using TorqueNet, ForceNet, and ConditionNet.
"""

import flax.linen as nn
from .base import TorqueNet, ForceNet, ConditionNet


class NeuralActuator(nn.Module):
    """MLP-based Neural Actuator.

    Interface aligned with other actuator models for unified usage:
    - Accepts state parameter (ignored, for API compatibility)
    - Accepts ts parameter (ignored, for API compatibility)
    - Returns 6 values: torque, final_force, raw_force, gate, condition, new_state
    """
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, history_input, current_state, state=None, ts: float = 1.0, training: bool = False):
        """Forward pass.

        Args:
            history_input: (batch, history_len * feature_dim)
            current_state: (batch, state_dim)
            state: ignored, for API compatibility with stateful models
            ts: ignored, for API compatibility with time-aware models
            training: whether in training mode

        Returns:
            torque: predicted torques (batch, 5)
            final_force: gated force prediction (batch, 3)
            raw_force: raw force before gating (batch, 3)
            gate: contact gate values (batch, 1) - 1=has contact, 0=no contact
            condition: motor condition (batch, 5) - 1=normal, 0=degraded per motor
            new_state: None (MLP is stateless)
        """
        torque, z = TorqueNet(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate
        )(history_input, current_state, training=training)

        final_force, raw_force, gate = ForceNet(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )(history_input, current_state, training=training)

        condition = ConditionNet(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate
        )(history_input, current_state, training=training)

        # Return None for state to match stateful model interface
        return torque, final_force, raw_force, gate, condition, None
