"""Convert a torch trainer checkpoint (.pt, state_dict or ema_state_dict) into
the flax pickle format ({params, feature_mean, feature_std}) that the flax
eval pipeline loads. Weight mapping goes through model_torch.flax_param_pairs.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_native.model_torch import TransformerActuatorTorch, flax_param_pairs


def torch_to_flax_params(model: TransformerActuatorTorch, flax_template) -> dict:
    """Fill a flax param tree (template from model.init) with this torch model's
    weights, transposing Linear kernels back to flax (in, out) layout."""
    import jax

    out = jax.tree_util.tree_map(lambda x: np.array(x), flax_template)  # fresh copy
    # pairs are views into `out`, kernels already transposed
    pairs = flax_param_pairs(model, out)
    for param, arr in pairs:
        t = param.detach().cpu().numpy().astype(np.float32)
        assert arr.shape == tuple(t.shape), (arr.shape, t.shape)
        arr[...] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use_ema", action="store_true", help="export ema_state_dict")
    ap.add_argument("--hidden_dim", type=int, default=192)
    ap.add_argument("--latent_dim", type=int, default=96)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=384)
    ap.add_argument("--feature_dim", type=int, default=36)
    ap.add_argument("--history_length", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["MUJOCO_GL"] = "egl"  # public_import pulls in mujoco, needs headless GL
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import jax
    import jax.numpy as jnp
    from public_import import create_model

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = payload["ema_state_dict" if args.use_ema else "state_dict"]

    tm = TransformerActuatorTorch(
        feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, d_ff=args.d_ff)
    tm.load_state_dict(sd)
    tm.eval()

    fm = create_model(model_type="transformer", hidden_dim=args.hidden_dim,
                      latent_dim=args.latent_dim, dropout_rate=0.1,
                      backbone_activation="silu", num_heads=args.num_heads,
                      num_layers=args.num_layers, d_ff=args.d_ff, pool_type="mean",
                      use_gated_attention=True, zero_init_head=False)
    template = fm.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)},
                       jnp.ones((1, args.history_length * args.feature_dim)),
                       jnp.ones((1, args.feature_dim)), None, ts=0.017)
    template = jax.device_get(template)
    fparams = torch_to_flax_params(tm, template)

    # roundtrip check before writing
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        h = rng.standard_normal((3, args.history_length * args.feature_dim)).astype(np.float32)
        c = rng.standard_normal((3, args.feature_dim)).astype(np.float32)
        fo = fm.apply(fparams, jnp.asarray(h), jnp.asarray(c), None, ts=0.017, training=False)
        with torch.no_grad():
            to = tm(torch.from_numpy(h), torch.from_numpy(c))
        for i in range(5):
            worst = max(worst, float(np.abs(np.asarray(fo[i]) - to[i].numpy()).max()))
    assert worst < 1e-4, f"roundtrip mismatch {worst:.3e}"

    out_payload = dict(params=fparams,
                       feature_mean=np.asarray(payload["feature_mean"]),
                       feature_std=np.asarray(payload["feature_std"]),
                       epoch=payload.get("epoch"))
    with open(args.out, "wb") as fh:
        pickle.dump(out_payload, fh)
    print(f"exported {'EMA' if args.use_ema else 'raw'} params -> {args.out} "
          f"(roundtrip max |flax-torch| = {worst:.2e})")


if __name__ == "__main__":
    main()
