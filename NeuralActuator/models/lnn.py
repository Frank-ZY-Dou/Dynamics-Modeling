"""
Liquid Neural Network (LNN) Actuator model.

Time-aware recurrent architecture using Closed-form Continuous-time (CfC) cells.
Based on: https://github.com/mlech26l/ncps
"""

import jax.numpy as jnp
import flax.linen as nn
from .base import gumbel_sigmoid


class CfCCell(nn.Module):
    """Closed-form Continuous-time (CfC) Cell.

    Based on official ncps implementation.

    Key features:
    - Backbone network processes [x, h] jointly
    - Two feedforward heads (ff1, ff2) for nonlinear transformation
    - Time-dependent interpolation: t_interp = sigmoid(time_a * ts + time_b)
    - Closed-form solution: h_new = ff1 * (1 - t_interp) + t_interp * ff2
    """
    hidden_dim: int
    dropout_rate: float = 0.1
    backbone_activation: str = "silu"  # silu, tanh, relu, gelu

    @nn.compact
    def __call__(self, h, x, ts: float = 1.0, training: bool = False):
        """Forward pass.

        Args:
            h: hidden state (batch, hidden_dim)
            x: input (batch, input_dim)
            ts: time step / time elapsed since last step (scalar or (batch,))
            training: whether in training mode

        Returns:
            h_new: updated hidden state (batch, hidden_dim)
        """
        # === Backbone: shared feature extraction from [x, h] ===
        backbone_input = jnp.concatenate([x, h], axis=-1)

        # Select activation function
        if self.backbone_activation == "silu":
            act_fn = nn.silu
        elif self.backbone_activation == "tanh":
            act_fn = nn.tanh
        elif self.backbone_activation == "relu":
            act_fn = nn.relu
        elif self.backbone_activation == "gelu":
            act_fn = nn.gelu
        else:
            act_fn = nn.silu

        # Backbone layer
        backbone_out = nn.Dense(self.hidden_dim)(backbone_input)
        backbone_out = act_fn(backbone_out)
        backbone_out = nn.Dropout(self.dropout_rate, deterministic=not training)(backbone_out)

        # === Head Networks ===
        # ff1: first nonlinear path
        ff1 = nn.Dense(self.hidden_dim)(backbone_out)
        ff1 = nn.silu(ff1)

        # ff2: second nonlinear path
        ff2 = nn.Dense(self.hidden_dim)(backbone_out)
        ff2 = nn.silu(ff2)

        # === Time-dependent gating ===
        time_a = nn.Dense(self.hidden_dim)(backbone_out)
        time_b = nn.Dense(self.hidden_dim)(backbone_out)

        # Ensure ts is broadcastable
        if jnp.ndim(ts) == 0:
            ts = jnp.ones((h.shape[0], 1)) * ts
        elif jnp.ndim(ts) == 1:
            ts = ts[:, None]  # (batch,) -> (batch, 1)

        # Time interpolation factor
        t_interp = nn.sigmoid(time_a * ts + time_b)

        # === Closed-form solution ===
        h_new = ff1 * (1.0 - t_interp) + t_interp * ff2

        return h_new


class LNNActuator(nn.Module):
    """Liquid Neural Network Actuator with CfC cells.

    Uses time-aware CfC cells for both torque and force prediction.
    The network maintains recurrent hidden states that evolve according
    to the closed-form continuous-time dynamics.

    Features:
    - Learnable initial hidden states (h0_torque, h0_force)
    - Time-aware CfC cells with proper ts handling
    - Gumbel-Sigmoid gating for force prediction
    """
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout_rate: float = 0.1
    backbone_activation: str = "silu"  # silu, tanh, relu, gelu

    def get_initial_state(self, params, batch_size: int = 1):
        """Get learnable initial hidden states.

        Args:
            params: model parameters containing 'lnn_init' collection
            batch_size: number of samples in batch

        Returns:
            lnn_state: tuple of (h_torque, h_force), each (batch, hidden_dim)
        """
        if 'lnn_init' in params:
            h0_torque = params['lnn_init']['h0_torque']
            h0_force = params['lnn_init']['h0_force']
            h0_torque = jnp.broadcast_to(h0_torque, (batch_size, self.hidden_dim))
            h0_force = jnp.broadcast_to(h0_force, (batch_size, self.hidden_dim))
        else:
            h0_torque = jnp.zeros((batch_size, self.hidden_dim))
            h0_force = jnp.zeros((batch_size, self.hidden_dim))
        return (h0_torque, h0_force)

    @nn.compact
    def __call__(self, history_input, current_state, lnn_state, ts: float = 1.0, training: bool = False):
        """Forward pass.

        Args:
            history_input: (batch, history_len * feature_dim)
            current_state: (batch, state_dim)
            lnn_state: tuple of (h_torque, h_force), each (batch, hidden_dim)
                       If None, uses learnable initial states
            ts: time step / elapsed time since last step (scalar or (batch,))
            training: whether in training mode

        Returns:
            torque: predicted torques (batch, 5)
            final_force: gated force prediction (batch, 3)
            raw_force: raw force before gating (batch, 3)
            gate: contact gate values (batch, 1) - 1=has contact, 0=no contact
            condition: motor condition (batch, 1) - 1=normal, 0=degraded
            new_lnn_state: tuple of updated hidden states
        """
        # Learnable initial hidden states (registered as parameters)
        h0_torque_param = self.param('h0_torque', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))
        h0_force_param = self.param('h0_force', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))

        # Use provided state or broadcast learnable initial state
        if lnn_state is None:
            batch_size = history_input.shape[0]
            h_torque_old = jnp.broadcast_to(h0_torque_param, (batch_size, self.hidden_dim))
            h_force_old = jnp.broadcast_to(h0_force_param, (batch_size, self.hidden_dim))
        else:
            h_torque_old, h_force_old = lnn_state

        # --- Torque Path ---
        x_torque = jnp.concatenate([history_input, current_state], axis=-1)

        h_torque_new = CfCCell(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            backbone_activation=self.backbone_activation
        )(h_torque_old, x_torque, ts=ts, training=training)

        # Decoder: hidden -> latent -> torque
        z = nn.Dense(self.latent_dim)(h_torque_new)
        z = nn.silu(z)

        t = nn.Dense(self.hidden_dim)(z)
        t = nn.silu(t)
        t = nn.Dropout(self.dropout_rate, deterministic=not training)(t)
        torque = nn.Dense(5)(t)

        # --- Force Path ---
        x_force = jnp.concatenate([history_input, current_state], axis=-1)

        h_force_new = CfCCell(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            backbone_activation=self.backbone_activation
        )(h_force_old, x_force, ts=ts, training=training)

        # Raw Force (f)
        raw_force = nn.Dense(3)(h_force_new)

        # Gate (g) with Gumbel-Sigmoid for training
        gate_logit = nn.Dense(1)(h_force_new)

        if training:
            try:
                rng_gumbel = self.make_rng('gumbel')
                gate = gumbel_sigmoid(gate_logit, tau=1.0, rng=rng_gumbel, hard=True)
            except:
                gate = nn.sigmoid(gate_logit)
        else:
            gate = (nn.sigmoid(gate_logit) > 0.5).astype(jnp.float32)

        # Final Force (g * f)
        final_force = gate * raw_force

        # --- Per-Motor Condition Head ---
        # Predicts motor condition for each of 5 motors from torque hidden state
        cond = nn.Dense(self.latent_dim)(h_torque_new)
        cond = nn.silu(cond)
        cond = nn.Dropout(self.dropout_rate, deterministic=not training)(cond)
        cond_logits = nn.Dense(5)(cond)  # 5 motors
        condition = nn.sigmoid(cond_logits)  # (batch, 5), 1=normal, 0=degraded per motor

        return torque, final_force, raw_force, gate, condition, (h_torque_new, h_force_new)
