"""Torque-surrogate networks for the hand, same calling convention as the NeuralActuator models:

    torque = model(history_flat (B, H*F), current (B, F))        -> (B, 8)

MLP: encoder -> latent -> decoder with LayerNorm (the NeuralActuator TorqueNet layout).
Transformer: the NeuralActuator TransformerActuator reduced to its torque head (gated attention, learnable
positional encoding, mean pooling over the H+1 tokens).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorqueMLP(nn.Module):
    def __init__(self, feature_dim: int, history_len: int, hidden_dim: int = 128, latent_dim: int = 32,
                 dropout: float = 0.0, n_out: int = 8, zero_init_head: bool = True):
        super().__init__()
        d_in = feature_dim * (history_len + 1)
        self.norm_in = nn.LayerNorm(d_in)
        self.enc = nn.Sequential(nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.latent = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU())
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, n_out)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, history_flat: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(torch.cat([history_flat, current], dim=-1))
        return self.head(self.dec(self.latent(self.enc(x))))


class _GatedMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout, gated):
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
            query, gate = self.q(x).chunk(2, dim=-1)
        else:
            query, gate = self.q(x), None
        q = query.view(B, S, self.h, hd).transpose(1, 2)
        k = self.k(x).view(B, S, self.h, hd).transpose(1, 2)
        v = self.v(x).view(B, S, self.h, hd).transpose(1, 2)
        w = self.drop(F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd), dim=-1))
        a = (w @ v).transpose(1, 2).reshape(B, S, self.d_model)
        if self.gated:
            a = a * torch.sigmoid(gate)
        return self.o(a)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, gated):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = _GatedMHA(d_model, num_heads, dropout, gated)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff2(self.drop(F.gelu(self.ff1(self.ln2(x)), approximate="tanh"))))


class TransformerTorque(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 96, latent_dim: int = 48, num_heads: int = 4,
                 num_layers: int = 2, d_ff: int = 192, dropout: float = 0.05, gated: bool = True, n_out: int = 8,
                 max_len: int = 32, zero_init_head: bool = True):
        super().__init__()
        self.inp = nn.Linear(feature_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(max_len, hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_EncoderLayer(hidden_dim, num_heads, d_ff, dropout, gated) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.tz = nn.Linear(hidden_dim, latent_dim)
        self.tt = nn.Linear(latent_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_out)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, history_flat: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        B = history_flat.shape[0]
        Fd = current.shape[-1]
        H = history_flat.shape[-1] // Fd
        seq = torch.cat([history_flat.view(B, H, Fd), current[:, None, :]], dim=1)
        x = self.drop(self.inp(seq) + self.pos[None, :seq.shape[1], :])
        for layer in self.layers:
            x = layer(x)
        pooled = self.ln_f(x).mean(dim=1)
        z = F.silu(self.tz(pooled))
        return self.head(self.drop(F.silu(self.tt(z))))


def create_model(kind: str, feature_dim: int, history_len: int, **kw) -> nn.Module:
    if kind == "mlp":
        return TorqueMLP(feature_dim, history_len, **kw)
    if kind == "transformer":
        return TransformerTorque(feature_dim, **kw)
    raise ValueError(kind)
