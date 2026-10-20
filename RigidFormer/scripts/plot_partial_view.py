"""Images of the simulated depth camera (data.partial_view) in one MOVi scene at the observation frame, as the
evaluation camera sees it: the depth image and the per-object segmentation whose pixels, back-projected, are the partial
point clouds. Objects take the colours of the rollout videos (src/utils/rollout_render.py, by object index); images are
cropped to 16:9 around the centre.

    python scripts/plot_partial_view.py --config configs/movi_b_partial.yaml --scene movi_40 --out partial_movi_40
    # writes partial_movi_40_depth.png and partial_movi_40_segmentation.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.object_scale import ObjectScale  # noqa: E402
from src.data.partial_view import PartialView  # noqa: E402
from src.utils.rollout_render import PALETTE  # noqa: E402

FLOOR = (221, 222, 224)          # the light floor's rendered tone in the videos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default='configs/movi_b_partial.yaml')
    ap.add_argument('--scene', required=True, help='scene folder name, e.g. movi_40')
    ap.add_argument('--frame', type=int, default=None,
                    help='observation frame (default: the last warm-up frame of the evaluation at step size 1)')
    ap.add_argument('--data', default=None, help='dataset folder (default: data.dataset_path of the config)')
    ap.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()

    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import colormaps
    path = Path(args.config)
    cfg = yaml.safe_load(open(path if path.is_absolute() else REPO / path))
    frame = args.frame if args.frame is not None else int(cfg['training']['metrics_test']['eval_warmup_frames']) - 1
    pv = PartialView(cfg['data']['partial_view'], REPO)
    scene_dir = Path(args.data or REPO / cfg['data']['dataset_path']) / args.scene
    shapes = [o['asset_id'] for o in json.loads((scene_dir / 'conversion.json').read_text())['objects']]
    traj = np.load(scene_dir / 'vis_traj.npy')[0]
    sizes = ObjectScale(REPO / cfg['data']['object_scale_file']).scales[args.scene]
    depth, segment = pv.images(args.scene, shapes, sizes, traj[frame, :, :3].astype(np.float64),
                               traj[frame, :, 3:7].astype(np.float64))
    H, W = depth.shape
    h = int(round(W * 9 / 16))
    rows = slice((H - h) // 2, (H - h) // 2 + h)

    # contrast stretched to the objects' depths (1st to 99th percentile, widened by a quarter of that range on each
    # side): with a low camera the floor spans far more depth than the objects
    finite = np.isfinite(depth)
    obj = finite & (segment >= 0)
    lo, hi = np.percentile(depth[obj] if obj.any() else depth[finite], [1, 99])
    lo, hi = lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)
    near = np.clip((hi - depth) / max(hi - lo, 1e-6), 0.0, 1.0)          # bright = near
    rgb = (colormaps['magma'](0.08 + 0.9 * near)[..., :3] * 255).round().astype(np.uint8)
    rgb[~finite] = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[rows]).save(f'{out}_depth.png')

    seg = np.empty((H, W, 3), np.uint8)
    seg[:] = FLOOR
    for i in np.unique(segment[segment >= 0]):
        seg[segment == i] = [int(PALETTE[i % len(PALETTE)][k:k + 2], 16) for k in (1, 3, 5)]
    Image.fromarray(seg[rows]).save(f'{out}_segmentation.png')
    seen = sum(int((segment == i).sum()) >= pv.min_points for i in range(len(shapes)))
    print(f'wrote {out}_depth.png and {out}_segmentation.png: {args.scene}, frame {frame}, {seen} of {len(shapes)} '
          f'objects observed')


if __name__ == '__main__':
    main()
