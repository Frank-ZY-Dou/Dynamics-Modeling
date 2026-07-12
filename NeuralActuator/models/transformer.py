"""
Transformer-based Neural Actuator model.

Uses self-attention to capture temporal dependencies in actuator history.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    d_model: int
    max_len: int = 100

    @nn.compact
    def __call__(self, seq_len: int):
        """Generate positional encodings.

        Args:
            seq_len: Length of sequence

        Returns:
            pe: (1, seq_len, d_model) positional encodings
        """
        position = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))

        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe[None, :, :]  # (1, seq_len, d_model)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding.

    Each position has a learned embedding vector.
    Better for short, fixed-length sequences like our actuator history.
    """
    d_model: int
    max_len: int = 16  # Max sequence length (history_length + 1)

    @nn.compact
    def __call__(self, seq_len: int):
        """Generate learnable positional encodings.

        Args:
            seq_len: Length of sequence

        Returns:
            pe: (1, seq_len, d_model) positional encodings
        """
        # Learnable position embeddings
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )
        # Return only the first seq_len positions
        return pos_embedding[None, :seq_len, :]  # (1, seq_len, d_model)


class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with elementwise output gating.

    Gate is learned from extended query projection:
    - q_proj outputs 2x dimension: [query, gate_score]
    - attn_output = attn_output * sigmoid(gate_score)

    Reference: Qwen3 Gated Attention
    """
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1
    use_gated_attention: bool = True

    @nn.compact
    def __call__(self, x, training: bool = False):
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            training: whether in training mode

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.num_heads

        # Q projection: output 2x for gated attention
        if self.use_gated_attention:
            q_out = nn.Dense(self.d_model * 2, use_bias=True)(x)  # (batch, seq, d_model*2)
            query, gate_score = jnp.split(q_out, 2, axis=-1)  # each (batch, seq, d_model)
        else:
            query = nn.Dense(self.d_model, use_bias=True)(x)
            gate_score = None

        # K, V projections
        key = nn.Dense(self.d_model, use_bias=True)(x)    # (batch, seq, d_model)
        value = nn.Dense(self.d_model, use_bias=True)(x)  # (batch, seq, d_model)

        # Reshape to (batch, num_heads, seq, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate, deterministic=not training)(attn_weights)

        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)

        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Apply elementwise gating
        if self.use_gated_attention and gate_score is not None:
            attn_output = attn_output * nn.sigmoid(gate_score)

        # Output projection
        output = nn.Dense(self.d_model, use_bias=True)(attn_output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-norm and optional gated attention."""
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    use_gated_attention: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            training: whether in training mode

        Returns:
            x: (batch, seq_len, d_model)
        """
        # Pre-norm self-attention with optional elementwise gating
        residual = x
        x_norm = nn.LayerNorm()(x)

        if self.use_gated_attention:
            # Gated multi-head attention
            attn_out = GatedMultiHeadAttention(
                num_heads=self.num_heads,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                use_gated_attention=True
            )(x_norm, training=training)
        else:
            # Standard multi-head attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x_norm, x_norm)

        attn_out = nn.Dropout(self.dropout_rate, deterministic=not training)(attn_out)
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.d_ff)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = residual + x

        return x


class TransformerActuator(nn.Module):
    """Transformer-based Neural Actuator.

    Architecture:
    1. Project input features to d_model
    2. Add sinusoidal positional encoding
    3. Transformer encoder layers
    4. Pool sequence (mean or last token)
    5. Decode to torque, force, and motor condition

    Interface aligned with other actuator models:
    - Accepts state parameter (ignored, for API compatibility)
    - Accepts ts parameter (ignored, for API compatibility)
    - Returns 6 values: torque, final_force, raw_force, gate, condition, new_state

    Motor Condition Head:
    - Outputs c ∈ [0, 1] where c=1 indicates normal operation, c=0 indicates degraded/damaged
    - Trained with BCE loss on motor condition labels
    - Enables condition monitoring without dedicated sensors
    """
    hidden_dim: int = 32      # d_model
    latent_dim: int = 16      # intermediate latent dim
    dropout_rate: float = 0.1
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 64            # feedforward hidden dim
    pool_type: str = "mean"   # "mean" or "last"
    use_gated_attention: bool = False  # Elementwise gated attention (Qwen3 style)
    zero_init_head: bool = False  # zero-init torque output layer (residual starts at Kt*I)
    n_joints: int = 5         # torque/condition head output dim (5 = OMX, 6 = SO-101)

    @nn.compact
    def __call__(self, history_input, current_state, state=None, ts: float = 1.0, training: bool = False):
        """Forward pass.

        Args:
            history_input: (batch, history_len * feature_dim) - flattened history
            current_state: (batch, feature_dim) - current features
            state: ignored, for API compatibility with stateful models
            ts: ignored, for API compatibility with time-aware models
            training: whether in training mode

        Returns:
            torque: predicted torques (batch, n_joints)
            final_force: gated force prediction (batch, 3)
            raw_force: raw force before gating (batch, 3)
            gate: contact gate values (batch, 1) - 1=has contact, 0=no contact
            condition: motor condition (batch, n_joints) - 1=normal, 0=degraded
            new_state: None (Transformer is stateless per-sequence)
        """
        batch_size = history_input.shape[0]

        # Infer history_len and feature_dim from shapes
        # history_input: (batch, history_len * feature_dim)
        # current_state: (batch, feature_dim)
        feature_dim = current_state.shape[-1]
        history_flat_dim = history_input.shape[-1]
        history_len = history_flat_dim // feature_dim

        # Reshape history: (batch, history_len * feature_dim) -> (batch, history_len, feature_dim)
        history = history_input.reshape(batch_size, history_len, feature_dim)

        # Append current state as the last token
        # sequence: (batch, history_len + 1, feature_dim)
        current_expanded = current_state[:, None, :]  # (batch, 1, feature_dim)
        sequence = jnp.concatenate([history, current_expanded], axis=1)
        seq_len = sequence.shape[1]

        # Project to d_model
        x = nn.Dense(self.hidden_dim)(sequence)  # (batch, seq_len, d_model)

        # Add learnable positional encoding (better for short, fixed-length sequences)
        pe = LearnablePositionalEncoding(d_model=self.hidden_dim, max_len=16)(seq_len)
        x = x + pe

        # Dropout on input
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Transformer encoder layers
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                use_gated_attention=self.use_gated_attention
            )(x, training=training)

        # Final layer norm
        x = nn.LayerNorm()(x)

        # Pool sequence to single vector
        if self.pool_type == "mean":
            pooled = jnp.mean(x, axis=1)  # (batch, d_model)
        else:  # "last"
            pooled = x[:, -1, :]  # (batch, d_model) - last token (current state)

        # === Torque Head ===
        z = nn.Dense(self.latent_dim)(pooled)
        z = nn.silu(z)

        t = nn.Dense(self.hidden_dim)(z)
        t = nn.silu(t)
        t = nn.Dropout(self.dropout_rate, deterministic=not training)(t)
        if self.zero_init_head:
            torque = nn.Dense(self.n_joints, kernel_init=nn.initializers.zeros)(t)
        else:
            torque = nn.Dense(self.n_joints)(t)

        # === Force Head ===
        f = nn.Dense(self.hidden_dim)(pooled)
        f = nn.silu(f)
        f = nn.Dropout(self.dropout_rate, deterministic=not training)(f)

        # Raw Force
        raw_force = nn.Dense(3)(f)

        # Contact gate: g = sigmoid(logit) in [0, 1]
        gate_logit = nn.Dense(1)(f)
        gate = nn.sigmoid(gate_logit)

        # Final Force (g * f)
        final_force = gate * raw_force

        # === Per-Motor Condition Head ===
        # Predicts motor operating condition for each motor: 1=normal, 0=degraded/damaged
        # Uses separate MLP from pooled representation
        cond = nn.Dense(self.latent_dim)(pooled)
        cond = nn.silu(cond)
        cond = nn.Dropout(self.dropout_rate, deterministic=not training)(cond)
        cond_logits = nn.Dense(self.n_joints)(cond)  # one per motor
        condition = nn.sigmoid(cond_logits)  # (batch, n_joints), range [0, 1] per motor

        # Return None for state (Transformer is stateless per forward pass)
        return torque, final_force, raw_force, gate, condition, None
