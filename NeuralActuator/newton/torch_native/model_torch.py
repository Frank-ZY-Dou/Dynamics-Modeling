"""Torch port of the flax TransformerActuator; loads flax checkpoints.

Parity notes:
- flax nn.gelu defaults to the tanh approximation -> torch F.gelu(approximate="tanh")
- flax LayerNorm eps = 1e-6 (torch default is 1e-5)
- flax Dense kernel is (in, out) -> torch Linear weight is its transpose
- learnable positional encoding: (max_len=16, d_model), first seq_len rows used
Interface matches the flax model: forward(history_flat, current) ->
(torque, final_force, raw_force, gate, condition, None).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, gated: bool):
        super().__init__()
        self.d_model, self.h, self.gated = d_model, num_heads, gated
        self.q = nn.Linear(d_model, d_model * 2 if gated else d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        hd = self.d_model // self.h
        if self.gated:
            q_out = self.q(x)
            query, gate_score = q_out.chunk(2, dim=-1)
        else:
            query, gate_score = self.q(x), None
        key, value = self.k(x), self.v(x)
        q = query.view(B, S, self.h, hd).transpose(1, 2)
        k = key.view(B, S, self.h, hd).transpose(1, 2)
        v = value.view(B, S, self.h, hd).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        w = self.drop(F.softmax(w, dim=-1))
        a = (w @ v).transpose(1, 2).reshape(B, S, self.d_model)
        if self.gated:
            a = a * torch.sigmoid(gate_score)
        return self.o(a)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, gated: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = GatedMHA(d_model, num_heads, dropout, gated)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        h = self.ff2(self.drop(F.gelu(self.ff1(self.ln2(x)), approximate="tanh")))
        return x + self.drop(h)


class TransformerActuatorTorch(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int, num_heads: int,
                 num_layers: int, d_ff: int, dropout: float = 0.1, gated: bool = True,
                 n_joints: int = 5, max_len: int = 16, pool_type: str = "mean",
                 zero_init_head: bool = False):
        super().__init__()
        self.pool_type = pool_type
        self.inp = nn.Linear(feature_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(max_len, hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_heads, d_ff, dropout, gated) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.tz = nn.Linear(hidden_dim, latent_dim)
        self.tt = nn.Linear(latent_dim, hidden_dim)
        self.t_out = nn.Linear(hidden_dim, n_joints)
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.fr = nn.Linear(hidden_dim, 3)
        self.fg = nn.Linear(hidden_dim, 1)
        self.cz = nn.Linear(hidden_dim, latent_dim)
        self.co = nn.Linear(latent_dim, n_joints)
        if zero_init_head:   # mirrors the flax zero_init_torque_head knob
            nn.init.zeros_(self.t_out.weight)
            nn.init.zeros_(self.t_out.bias)

    def forward(self, history_flat: torch.Tensor, current: torch.Tensor):
        B = history_flat.shape[0]
        Fd = current.shape[-1]
        H = history_flat.shape[-1] // Fd
        seq = torch.cat([history_flat.view(B, H, Fd), current[:, None, :]], dim=1)
        S = seq.shape[1]
        x = self.inp(seq) + self.pos[None, :S, :]
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        pooled = x.mean(dim=1) if self.pool_type == "mean" else x[:, -1, :]

        z = F.silu(self.tz(pooled))
        t = self.drop(F.silu(self.tt(z)))
        torque = self.t_out(t)

        f = self.drop(F.silu(self.ff(pooled)))
        raw_force = self.fr(f)
        gate = torch.sigmoid(self.fg(f))
        final_force = gate * raw_force

        c = self.drop(F.silu(self.cz(pooled)))
        condition = torch.sigmoid(self.co(c))
        return torque, final_force, raw_force, gate, condition, None


def flax_param_pairs(model: TransformerActuatorTorch, flax_tree: dict):
    """Walk a flax param tree (or a same-shaped grad tree) and return
    (torch_parameter, numpy_array) pairs, transposing Dense kernels to Linear layout."""
    import numpy as np

    p = flax_tree["params"] if "params" in flax_tree else flax_tree
    pairs = []

    def np_(x):
        return np.asarray(x)

    def lin(dst: nn.Linear, node):
        pairs.append((dst.weight, np_(node["kernel"]).T))
        pairs.append((dst.bias, np_(node["bias"])))

    def ln(dst: nn.LayerNorm, node):
        pairs.append((dst.weight, np_(node["scale"])))
        pairs.append((dst.bias, np_(node["bias"])))

    lin(model.inp, p["Dense_0"])
    pairs.append((model.pos, np_(p["LearnablePositionalEncoding_0"]["pos_embedding"])))
    for i, layer in enumerate(model.layers):
        q = p[f"TransformerEncoderLayer_{i}"]
        ln(layer.ln1, q["LayerNorm_0"])
        a = q["GatedMultiHeadAttention_0"]
        lin(layer.attn.q, a["Dense_0"])
        lin(layer.attn.k, a["Dense_1"])
        lin(layer.attn.v, a["Dense_2"])
        lin(layer.attn.o, a["Dense_3"])
        ln(layer.ln2, q["LayerNorm_1"])
        lin(layer.ff1, q["Dense_0"])
        lin(layer.ff2, q["Dense_1"])
    ln(model.ln_f, p["LayerNorm_0"])
    lin(model.tz, p["Dense_1"])
    lin(model.tt, p["Dense_2"])
    lin(model.t_out, p["Dense_3"])
    lin(model.ff, p["Dense_4"])
    lin(model.fr, p["Dense_5"])
    lin(model.fg, p["Dense_6"])
    lin(model.cz, p["Dense_7"])
    lin(model.co, p["Dense_8"])
    return pairs


def load_flax_params(model: TransformerActuatorTorch, flax_params: dict) -> None:
    """Copy a flax TransformerActuator param tree into the torch port."""
    import numpy as np

    for param, arr in flax_param_pairs(model, flax_params):
        assert tuple(param.shape) == tuple(arr.shape), (param.shape, arr.shape)
        # copy=True: jax.device_get arrays are read-only and the optimizer writes
        # params in place, so don't wrap the original buffer
        param.data = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
