"""Deepen a trained model's decoder without changing its function, as the start of a fine-tuning run.

The new layers continue the stack: layer L+i starts as a copy of layer i of the trained decoder, with the output
projections of its residual branches (object self-attention out_proj, feed-forward fc2) and its step-size FiLM head
set to zero, so it adds nothing to the residual stream until training moves those weights. With Block AttnRes the
whole decoder becomes one block (block_size = new depth) and every new layer's depth-attention query is the last
trained layer's, so each new layer and the final aggregation see the same sources with the same weights as the
trained model's final aggregation. The predictor keeps reading the trained layers (model.predictor.layer_indices,
e.g. [0, 1, 2, 8] for 4 -> 8 layers), so the grown model computes the same function as the trained one.

    python scripts/grow_decoder.py --checkpoint checkpoints/rigidformer_movib.pth --layers 8 \
        --out exps/grown/rigidformer_movib_8l.pth

Fine-tune it with a config whose model section is the one stored in the output checkpoint (printed at the end).
"""
import argparse
import copy
import re
from pathlib import Path

import torch

ZERO_IN_NEW_LAYERS = ('obj_self_attn.out_proj.', 'obs_ffn.fc2.', 'layer_film.film_net.2.')


def grow_state_dict(sd, old_layers, new_layers):
    prefix = 'module.' if any(k.startswith('module.') for k in sd) else ''
    layer_re = re.compile(r'^' + re.escape(prefix) + r'decoder\.layers\.(\d+)\.(.+)$')
    query_key = lambda i: f'{prefix}decoder.attn_res_queries.{i}'
    out = dict(sd)
    found = sorted({int(m.group(1)) for k in sd for m in [layer_re.match(k)] if m})
    if found != list(range(old_layers)):
        raise ValueError(f'checkpoint decoder layers {found}, expected 0..{old_layers - 1}')
    zeroed = 0
    for k, v in sd.items():
        m = layer_re.match(k)
        if not m:
            continue
        src, rest = int(m.group(1)), m.group(2)
        for new in range(old_layers + src, new_layers, old_layers):
            w = v.clone()
            if rest.startswith(ZERO_IN_NEW_LAYERS):
                w.zero_()
                zeroed += 1
            out[f'{prefix}decoder.layers.{new}.{rest}'] = w
    if query_key(0) in sd:
        for new in range(old_layers, new_layers):
            out[query_key(new)] = sd[query_key(old_layers - 1)].clone()
    n_new = new_layers - old_layers
    if zeroed != len(ZERO_IN_NEW_LAYERS) * 2 * n_new:
        raise ValueError(f'zeroed {zeroed} tensors, expected {len(ZERO_IN_NEW_LAYERS) * 2 * n_new} '
                         f'(weight and bias of {ZERO_IN_NEW_LAYERS} per new layer)')
    return out


def grown_config(cfg, old_layers, new_layers):
    cfg = copy.deepcopy(cfg)
    dec = cfg['model']['decoder']
    if int(dec['num_layers']) != old_layers:
        raise ValueError(f"config has {dec['num_layers']} decoder layers, checkpoint {old_layers}")
    rule = [0, 1, 2, old_layers] if old_layers >= 4 else list(range(old_layers + 1))
    old_indices = cfg['model']['predictor'].get('layer_indices') or rule
    if old_indices[-1] != old_layers:
        raise ValueError(f'predictor reads layers {old_indices}; the last one must be the last decoder layer')
    dec['num_layers'] = new_layers
    bar = dec.get('block_attn_res') or {}
    if bar.get('enable', False):
        if int(bar.get('block_size', 4)) < old_layers:
            raise ValueError('growing a decoder with more than one Block AttnRes block does not keep the function')
        bar['block_size'] = new_layers
    cfg['model']['predictor']['layer_indices'] = list(old_indices[:-1]) + [new_layers]
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--layers', type=int, required=True, help='new decoder depth (a multiple of the trained depth)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck.get('config')
    if not isinstance(cfg, dict):
        raise ValueError('the checkpoint stores no config; the grown model needs one')
    old = int(cfg['model']['decoder']['num_layers'])
    if args.layers <= old or args.layers % old:
        raise ValueError(f'--layers must be a multiple of the trained depth {old} and larger')
    sd = grow_state_dict(ck['model_state_dict'], old, args.layers)
    new_cfg = grown_config(cfg, old, args.layers)
    # weights only: a fine-tuning run started from it begins a fresh schedule at epoch 0
    out = {'model_state_dict': sd, 'config': new_cfg, 'grown_from': str(args.checkpoint)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    n_old = sum(v.numel() for v in ck['model_state_dict'].values())
    n_new = sum(v.numel() for v in sd.values())
    print(f'decoder {old} -> {args.layers} layers: {n_old / 1e6:.1f}M -> {n_new / 1e6:.1f}M parameters; wrote {args.out}')
    print(f"config: model.decoder.num_layers={args.layers}, "
          f"model.decoder.block_attn_res.block_size={new_cfg['model']['decoder'].get('block_attn_res', {}).get('block_size')}, "
          f"model.predictor.layer_indices={new_cfg['model']['predictor']['layer_indices']}")


if __name__ == '__main__':
    main()
