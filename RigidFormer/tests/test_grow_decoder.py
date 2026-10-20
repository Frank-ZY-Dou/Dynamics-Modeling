"""The grown decoder (scripts/grow_decoder.py) computes the same function as the trained one. CPU only.

    python -m pytest tests/test_grow_decoder.py -q
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.decoder import BidirectionalDecoder  # noqa: E402

_spec = importlib.util.spec_from_file_location('grow_decoder', ROOT / 'scripts' / 'grow_decoder.py')
grow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grow)


class Wrap(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder


def make_decoder(num_layers, save):
    # the released models' switches (no global state, object self-attention, gated attention, per-layer FiLM,
    # one Block AttnRes block with the final aggregation as the last features), small dimensions
    return BidirectionalDecoder(
        state_dim=64, num_layers=num_layers, num_heads=2, dropout=0.0, activation='swiglu', return_all_layers=True,
        use_rope=True, norm_type='rms_norm', qk_norm=True, num_register_tokens=4, rope_set_aggregation=True,
        rope_dim=24, dim_per_coord_3d=8, dim_per_coord_multi=8, layer_indices_to_save=save, elementwise_gate=True,
        state_update_enabled=False, objects_self_attention=True, block_attn_res=True,
        block_attn_res_block_size=num_layers, block_attn_res_final_aggregate=True, film_conditioning=True)


def inputs(B=3, N=5, D=64):
    """Object features, anchor positions, step-size features and the additive padding mask (the last object of
    sample 0 is padding), as in training. With a mask the attention runs through SDPA, which also runs on the CPU."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, N, D, generator=g)
    pos = torch.randn(B, N, 3, generator=g) * 2
    time_features = torch.tensor([[1.0, 1.0], [5.0, 25.0], [10.0, 100.0]])[:B]
    mask = torch.zeros(B, 1, 1, N)
    mask[0, ..., -1] = float('-inf')
    return x, pos, time_features, mask


@pytest.mark.parametrize('old,new', [(2, 4), (2, 6), (4, 8)])
def test_grown_decoder_keeps_the_function(old, new):
    torch.manual_seed(0)
    rule = list(range(old + 1)) if old < 4 else [0, 1, 2, old]
    small = Wrap(make_decoder(old, set(rule)))
    # move every parameter away from its initial value (FiLM heads and depth-attention queries start at zero), so
    # that a wrong copy or a missed zeroing shows up in the outputs
    with torch.no_grad():
        for p in small.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    sd = grow.grow_state_dict(small.state_dict(), old, new)
    big_indices = rule[:-1] + [new]
    big = Wrap(make_decoder(new, set(big_indices)))
    big.load_state_dict(sd, strict=True)
    small.eval()
    big.eval()
    x, pos, tf, mask = inputs()
    with torch.no_grad():
        _, out_small = small.decoder(None, x, obj_anchor_pos=pos, mask=mask, time_features=tf)
        _, out_big = big.decoder(None, x, obj_anchor_pos=pos, mask=mask, time_features=tf)
    assert len(out_small) == len(out_big) == len(rule)
    for a, b in zip(out_small, out_big):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (a - b).abs().max()
    # the new layers are part of the computation: once their output projections move, the output changes
    with torch.no_grad():
        big.decoder.layers[new - 1].obs_ffn.fc2.weight.normal_(0, 0.1)
        _, out_moved = big.decoder(None, x, obj_anchor_pos=pos, mask=mask, time_features=tf)
    assert not torch.allclose(out_moved[-1], out_small[-1], atol=1e-3)


def test_new_layers_start_with_zero_residual_outputs_and_copied_queries():
    torch.manual_seed(0)
    small = Wrap(make_decoder(2, {0, 1, 2}))
    with torch.no_grad():
        for p in small.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    sd = grow.grow_state_dict(small.state_dict(), 2, 4)
    for layer in (2, 3):
        for name in ('obj_self_attn.out_proj', 'obs_ffn.fc2', 'layer_film.film_net.2'):
            for part in ('weight', 'bias'):
                assert sd[f'decoder.layers.{layer}.{name}.{part}'].abs().max() == 0
        # the rest of the layer is a copy of layer (layer - 2)
        assert torch.equal(sd[f'decoder.layers.{layer}.obj_self_attn.q_proj.weight'],
                           sd[f'decoder.layers.{layer - 2}.obj_self_attn.q_proj.weight'])
        assert torch.equal(sd[f'decoder.attn_res_queries.{layer}'], sd['decoder.attn_res_queries.1'])


def test_grown_config():
    cfg = {'model': {'decoder': {'num_layers': 4, 'block_attn_res': {'enable': True, 'block_size': 4}},
                     'predictor': {'hidden_dims': [768, 256]}}}
    new = grow.grown_config(cfg, 4, 8)
    assert new['model']['decoder']['num_layers'] == 8
    assert new['model']['decoder']['block_attn_res']['block_size'] == 8
    assert new['model']['predictor']['layer_indices'] == [0, 1, 2, 8]
    assert cfg['model']['decoder']['num_layers'] == 4                    # input not modified
    two_blocks = {'model': {'decoder': {'num_layers': 8, 'block_attn_res': {'enable': True, 'block_size': 4}},
                            'predictor': {}}}
    with pytest.raises(ValueError):
        grow.grown_config(two_blocks, 8, 16)
