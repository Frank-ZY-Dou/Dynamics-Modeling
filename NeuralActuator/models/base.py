"""
Base classes and shared components for neural actuator models.

All actuator models must follow the unified interface:
    torque, final_force, raw_force, gate, condition, new_state = model(
        history_input, current_state, state, ts, training
    )

Where:
    - history_input: (batch, history_len * feature_dim)
    - current_state: (batch, feature_dim)
    - state: hidden state tuple or None
    - ts: time step
    - training: bool

Returns:
    - torque: (batch, 5)
    - final_force: (batch, 3)
    - raw_force: (batch, 3)
    - gate: (batch, 1) - contact gate (1=has contact, 0=no contact)
    - condition: (batch, 1) - motor condition (1=normal, 0=degraded)
    - new_state: updated state or None
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional


def gumbel_sigmoid(logits, tau=1.0, rng=None, hard=False):
    """Sample from Gumbel-Sigmoid distribution.

    Args:
        logits: input logits
        tau: temperature parameter
        rng: JAX random key
        hard: if True, use straight-through estimator

    Returns:
        Sampled values in [0, 1]
    """
    if rng is None:
        return nn.sigmoid(logits)

    # Sample Gumbel noise: Gumbel(0, 1) = -log(-log(U)) where U ~ Uniform(0, 1)
    u = jax.random.uniform(rng, shape=logits.shape, minval=1e-10, maxval=1.0 - 1e-10)
    gumbel_noise = -jnp.log(-jnp.log(u))

    # Gumbel-Sigmoid: sigmoid((logits + noise) / tau)
    y_soft = nn.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        # Straight-Through Estimator
        y_hard = (y_soft > 0.5).astype(jnp.float32)
        y = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
        return y
    else:
        return y_soft


class TorqueNet(nn.Module):
    """MLP network for torque prediction.

    Encoder -> Latent -> Decoder architecture.
    Uses LayerNorm for training stability (Pre-LN style).
    """
    latent_dim: int = 16
    hidden_dim: int = 32
    dropout_rate: float = 0.1
    output_dim: int = 5  # 4 arm joints + 1 gripper
    use_layernorm: bool = True  # Enable LayerNorm for stability

    @nn.compact
    def __call__(self, history_input, current_state, training: bool = False):
        # Input normalization
        x = jnp.concatenate([history_input, current_state], axis=-1)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)

        # Encoder
        x = nn.Dense(self.hidden_dim)(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Latent Z
        z = nn.Dense(self.latent_dim)(x)
        if self.use_layernorm:
            z = nn.LayerNorm()(z)
        z = nn.silu(z)

        # Torque Decoder
        t = nn.Dense(self.hidden_dim)(z)
        if self.use_layernorm:
            t = nn.LayerNorm()(t)
        t = nn.silu(t)
        t = nn.Dropout(self.dropout_rate, deterministic=not training)(t)
        t = nn.Dense(self.hidden_dim)(t)
        if self.use_layernorm:
            t = nn.LayerNorm()(t)
        t = nn.silu(t)
        t = nn.Dropout(self.dropout_rate, deterministic=not training)(t)
        torque = nn.Dense(self.output_dim)(t)

        return torque, z


class ForceNet(nn.Module):
    """MLP network for force prediction with gating.

    Predicts raw force and a binary gate for contact detection.
    Uses Gumbel-Sigmoid during training for differentiable sampling.
    Uses LayerNorm for training stability.
    """
    hidden_dim: int = 32
    dropout_rate: float = 0.1
    output_dim: int = 3  # Force x, y, z
    use_layernorm: bool = True  # Enable LayerNorm for stability

    @nn.compact
    def __call__(self, history_input, current_state, training: bool = False):
        x = jnp.concatenate([history_input, current_state], axis=-1)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim)(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Raw Force (f)
        raw_force = nn.Dense(self.output_dim)(x)

        # Gate (g)
        gate_logit = nn.Dense(1)(x)

        if training:
            try:
                rng_gumbel = self.make_rng('gumbel')
                gate = gumbel_sigmoid(gate_logit, tau=1.0, rng=rng_gumbel, hard=True)
            except:
                gate = nn.sigmoid(gate_logit)
        else:
            # Inference: Hard threshold
            gate = (nn.sigmoid(gate_logit) > 0.5).astype(jnp.float32)

        # Final Force (g * f)
        final_force = gate * raw_force

        return final_force, raw_force, gate


class ConditionNet(nn.Module):
    """MLP network for per-motor condition prediction.

    Predicts motor operating condition for each of 5 motors:
    - c = 1: normal motor operation
    - c = 0: degraded or damaged motor state

    Output shape: (batch, 5) - one condition per motor
    This enables condition monitoring by detecting anomalous patterns
    from current-torque discrepancies and thermal signatures.
    Uses LayerNorm for training stability.
    """
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout_rate: float = 0.1
    num_motors: int = 5  # Number of motors to predict condition for
    use_layernorm: bool = True  # Enable LayerNorm for stability

    @nn.compact
    def __call__(self, history_input, current_state, training: bool = False):
        x = jnp.concatenate([history_input, current_state], axis=-1)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.latent_dim)(x)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Per-motor condition logits -> sigmoid to get probability in [0, 1]
        cond_logits = nn.Dense(self.num_motors)(x)  # (batch, 5)
        condition = nn.sigmoid(cond_logits)  # (batch, 5)

        return condition
