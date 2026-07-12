"""
GRU-based Neural Actuator model.

Recurrent architecture using standard GRU cells for temporal processing.
"""

import jax.numpy as jnp
import flax.linen as nn
from .base import gumbel_sigmoid


class GRUCell(nn.Module):
    """Standard GRU Cell implementation.

    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    where:
        z_t = sigmoid(W_z @ x + U_z @ h + b_z)  # update gate
        r_t = sigmoid(W_r @ x + U_r @ h + b_r)  # reset gate
        h_tilde = tanh(W_h @ x + U_h @ (r_t * h) + b_h)  # candidate
    """
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, h, x, training: bool = False):
        """Forward pass.

        Args:
            h: hidden state (batch, hidden_dim)
            x: input (batch, input_dim)
            training: whether in training mode

        Returns:
            h_new: updated hidden state (batch, hidden_dim)
        """
        # Concatenate input and hidden for efficient computation
        combined = jnp.concatenate([x, h], axis=-1)

        # Update gate
        z = nn.Dense(self.hidden_dim)(combined)
        z = nn.sigmoid(z)

        # Reset gate
        r = nn.Dense(self.hidden_dim)(combined)
        r = nn.sigmoid(r)

        # Candidate hidden state
        combined_reset = jnp.concatenate([x, r * h], axis=-1)
        h_tilde = nn.Dense(self.hidden_dim)(combined_reset)
        h_tilde = nn.tanh(h_tilde)

        # New hidden state
        h_new = (1 - z) * h + z * h_tilde

        return h_new


class GRUActuator(nn.Module):
    """GRU-based Neural Actuator.

    Uses standard GRU cells for both torque and force prediction paths.
    Maintains recurrent hidden states for temporal modeling.

    Features:
    - Learnable initial hidden states (h0_torque, h0_force)
    - Separate GRU cells for torque and force paths
    - Gumbel-Sigmoid gating for force prediction
    - Per-motor condition prediction
    """
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout_rate: float = 0.1
    num_layers: int = 1  # Number of stacked GRU layers

    def get_initial_state(self, params, batch_size: int = 1):
        """Get learnable initial hidden states.

        Args:
            params: model parameters containing 'gru_init' collection
            batch_size: number of samples in batch

        Returns:
            gru_state: tuple of (h_torque, h_force), each (batch, hidden_dim)
        """
        if 'gru_init' in params:
            h0_torque = params['gru_init']['h0_torque']
            h0_force = params['gru_init']['h0_force']
            h0_torque = jnp.broadcast_to(h0_torque, (batch_size, self.hidden_dim))
            h0_force = jnp.broadcast_to(h0_force, (batch_size, self.hidden_dim))
        else:
            h0_torque = jnp.zeros((batch_size, self.hidden_dim))
            h0_force = jnp.zeros((batch_size, self.hidden_dim))
        return (h0_torque, h0_force)

    @nn.compact
    def __call__(self, history_input, current_state, state=None, ts: float = 1.0, training: bool = False):
        """Forward pass.

        Args:
            history_input: (batch, history_len * feature_dim)
            current_state: (batch, state_dim)
            state: tuple of (h_torque, h_force), each (batch, hidden_dim)
                   If None, uses learnable initial states
            ts: time step (ignored for GRU, kept for API compatibility)
            training: whether in training mode

        Returns:
            torque: predicted torques (batch, 5)
            final_force: gated force prediction (batch, 3)
            raw_force: raw force before gating (batch, 3)
            gate: contact gate values (batch, 1) - 1=has contact, 0=no contact
            condition: motor condition (batch, 5) - 1=normal, 0=degraded per motor
            new_state: tuple of updated hidden states
        """
        # Learnable initial hidden states (registered as parameters)
        h0_torque_param = self.param('h0_torque', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))
        h0_force_param = self.param('h0_force', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))

        # Use provided state or broadcast learnable initial state
        if state is None:
            batch_size = history_input.shape[0]
            h_torque_old = jnp.broadcast_to(h0_torque_param, (batch_size, self.hidden_dim))
            h_force_old = jnp.broadcast_to(h0_force_param, (batch_size, self.hidden_dim))
        else:
            h_torque_old, h_force_old = state

        # --- Torque Path ---
        x_torque = jnp.concatenate([history_input, current_state], axis=-1)

        # Apply GRU layers
        h_torque_new = h_torque_old
        for i in range(self.num_layers):
            h_torque_new = GRUCell(
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                name=f'gru_torque_{i}'
            )(h_torque_new, x_torque, training=training)

        # Decoder: hidden -> latent -> torque
        z = nn.Dense(self.latent_dim)(h_torque_new)
        z = nn.silu(z)

        t = nn.Dense(self.hidden_dim)(z)
        t = nn.silu(t)
        t = nn.Dropout(self.dropout_rate, deterministic=not training)(t)
        torque = nn.Dense(5)(t)

        # --- Force Path ---
        x_force = jnp.concatenate([history_input, current_state], axis=-1)

        # Apply GRU layers
        h_force_new = h_force_old
        for i in range(self.num_layers):
            h_force_new = GRUCell(
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                name=f'gru_force_{i}'
            )(h_force_new, x_force, training=training)

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
