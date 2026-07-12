"""
LSTM-based Neural Actuator model.

Recurrent architecture using standard LSTM cells for temporal processing.
"""

import jax.numpy as jnp
import flax.linen as nn
from .base import gumbel_sigmoid


class LSTMCell(nn.Module):
    """Standard LSTM Cell implementation.

    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    where:
        f_t = sigmoid(W_f @ x + U_f @ h + b_f)  # forget gate
        i_t = sigmoid(W_i @ x + U_i @ h + b_i)  # input gate
        o_t = sigmoid(W_o @ x + U_o @ h + b_o)  # output gate
        g_t = tanh(W_g @ x + U_g @ h + b_g)      # cell gate
    """
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, state, x, training: bool = False):
        """Forward pass.

        Args:
            state: tuple of (h, c) where h and c are (batch, hidden_dim)
            x: input (batch, input_dim)
            training: whether in training mode

        Returns:
            new_state: tuple of (h_new, c_new)
        """
        h, c = state

        # Concatenate input and hidden for efficient computation
        combined = jnp.concatenate([x, h], axis=-1)

        # Forget gate
        f = nn.Dense(self.hidden_dim)(combined)
        f = nn.sigmoid(f)

        # Input gate
        i = nn.Dense(self.hidden_dim)(combined)
        i = nn.sigmoid(i)

        # Output gate
        o = nn.Dense(self.hidden_dim)(combined)
        o = nn.sigmoid(o)

        # Cell gate (candidate)
        g = nn.Dense(self.hidden_dim)(combined)
        g = nn.tanh(g)

        # New cell state
        c_new = f * c + i * g

        # New hidden state
        h_new = o * nn.tanh(c_new)

        return (h_new, c_new)


class LSTMActuator(nn.Module):
    """LSTM-based Neural Actuator.

    Uses standard LSTM cells for both torque and force prediction paths.
    Maintains recurrent hidden states (h, c) for temporal modeling.

    Features:
    - Learnable initial hidden states (h0, c0 for torque and force)
    - Separate LSTM cells for torque and force paths
    - Gumbel-Sigmoid gating for force prediction
    - Per-motor condition prediction
    """
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout_rate: float = 0.1
    num_layers: int = 1  # Number of stacked LSTM layers

    def get_initial_state(self, params, batch_size: int = 1):
        """Get learnable initial hidden states.

        Args:
            params: model parameters containing 'lstm_init' collection
            batch_size: number of samples in batch

        Returns:
            lstm_state: tuple of ((h_torque, c_torque), (h_force, c_force))
        """
        if 'lstm_init' in params:
            h0_torque = jnp.broadcast_to(params['lstm_init']['h0_torque'], (batch_size, self.hidden_dim))
            c0_torque = jnp.broadcast_to(params['lstm_init']['c0_torque'], (batch_size, self.hidden_dim))
            h0_force = jnp.broadcast_to(params['lstm_init']['h0_force'], (batch_size, self.hidden_dim))
            c0_force = jnp.broadcast_to(params['lstm_init']['c0_force'], (batch_size, self.hidden_dim))
        else:
            h0_torque = jnp.zeros((batch_size, self.hidden_dim))
            c0_torque = jnp.zeros((batch_size, self.hidden_dim))
            h0_force = jnp.zeros((batch_size, self.hidden_dim))
            c0_force = jnp.zeros((batch_size, self.hidden_dim))
        return ((h0_torque, c0_torque), (h0_force, c0_force))

    @nn.compact
    def __call__(self, history_input, current_state, state=None, ts: float = 1.0, training: bool = False):
        """Forward pass.

        Args:
            history_input: (batch, history_len * feature_dim)
            current_state: (batch, state_dim)
            state: tuple of ((h_torque, c_torque), (h_force, c_force))
                   If None, uses learnable initial states
            ts: time step (ignored for LSTM, kept for API compatibility)
            training: whether in training mode

        Returns:
            torque: predicted torques (batch, 5)
            final_force: gated force prediction (batch, 3)
            raw_force: raw force before gating (batch, 3)
            gate: contact gate values (batch, 1) - 1=has contact, 0=no contact
            condition: motor condition (batch, 5) - 1=normal, 0=degraded per motor
            new_state: tuple of updated hidden states ((h_torque, c_torque), (h_force, c_force))
        """
        # Learnable initial hidden states (registered as parameters)
        h0_torque_param = self.param('h0_torque', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))
        c0_torque_param = self.param('c0_torque', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))
        h0_force_param = self.param('h0_force', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))
        c0_force_param = self.param('c0_force', nn.initializers.normal(stddev=0.01), (1, self.hidden_dim))

        # Use provided state or broadcast learnable initial state
        if state is None:
            batch_size = history_input.shape[0]
            h_torque_old = jnp.broadcast_to(h0_torque_param, (batch_size, self.hidden_dim))
            c_torque_old = jnp.broadcast_to(c0_torque_param, (batch_size, self.hidden_dim))
            h_force_old = jnp.broadcast_to(h0_force_param, (batch_size, self.hidden_dim))
            c_force_old = jnp.broadcast_to(c0_force_param, (batch_size, self.hidden_dim))
        else:
            (h_torque_old, c_torque_old), (h_force_old, c_force_old) = state

        # --- Torque Path ---
        x_torque = jnp.concatenate([history_input, current_state], axis=-1)

        # Apply LSTM layers
        h_torque, c_torque = h_torque_old, c_torque_old
        for i in range(self.num_layers):
            h_torque, c_torque = LSTMCell(
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                name=f'lstm_torque_{i}'
            )((h_torque, c_torque), x_torque, training=training)

        # Decoder: hidden -> latent -> torque
        z = nn.Dense(self.latent_dim)(h_torque)
        z = nn.silu(z)

        t = nn.Dense(self.hidden_dim)(z)
        t = nn.silu(t)
        t = nn.Dropout(self.dropout_rate, deterministic=not training)(t)
        torque = nn.Dense(5)(t)

        # --- Force Path ---
        x_force = jnp.concatenate([history_input, current_state], axis=-1)

        # Apply LSTM layers
        h_force, c_force = h_force_old, c_force_old
        for i in range(self.num_layers):
            h_force, c_force = LSTMCell(
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                name=f'lstm_force_{i}'
            )((h_force, c_force), x_force, training=training)

        # Raw Force (f)
        raw_force = nn.Dense(3)(h_force)

        # Gate (g) with Gumbel-Sigmoid for training
        gate_logit = nn.Dense(1)(h_force)

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
        cond = nn.Dense(self.latent_dim)(h_torque)
        cond = nn.silu(cond)
        cond = nn.Dropout(self.dropout_rate, deterministic=not training)(cond)
        cond_logits = nn.Dense(5)(cond)  # 5 motors
        condition = nn.sigmoid(cond_logits)  # (batch, 5), 1=normal, 0=degraded per motor

        return torque, final_force, raw_force, gate, condition, ((h_torque, c_torque), (h_force, c_force))
