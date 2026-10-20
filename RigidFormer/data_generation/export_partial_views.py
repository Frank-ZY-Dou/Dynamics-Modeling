"""Partial point clouds of the MOVi scenes for the Hugging Face dataset (configs movi_sphere_partial, movi_a_partial,
movi_b_partial): what the evaluation depth camera of data.partial_view sees, with the ground-truth trajectories.

One row per scene of every split, with
  - the depth camera: data.partial_view's evaluation camera (`eval_camera`; pulled_back: Kubric's camera direction at
    `distance` from `target`), K and the camera-to-world matrix R in Kubric's convention, the resolution and the
    vertical field of view;
  - per observation frame (the last of the 4 warm-up frames of an evaluation rollout, which starts at frame 0: frames
    3, 15 and 30 for step sizes 1, 5 and 10), the depth image (along the optical axis, uint16 PNG in mm, 0 where
    nothing is hit within 65.535 m; the floor is included) and the segmentation (uint8 PNG, object i is i + 1, 0 is
    the floor and the background);
  - per observation frame and object, the observed points: the object's pixels of that depth image back-projected into
    the object's frame (asset units, like the meshes), in farthest-point order, at most max_points. These are the
    points the evaluation observes, before completion (src/data/multi_dataloader.py, _partial_item). Objects with
    fewer than min_points pixels are not observed and have no points;
  - the ground truth: every object's shape, size, and positions and quaternions (w, x, y, z) at every frame, as in
    data/<name>. An observed point u of object i is at positions[t, i] + size[i] * R(quaternions[t, i]) u at frame t.

    python data_generation/export_partial_views.py --dataset RigidFormer-MOVi --name movi_b \\
        --config configs/movi_b_partial.yaml --out RigidFormer-MOVi [--workers 16] [--scenes 4]
"""
import argparse
import io
import json
import sys
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.partial_view import PartialView  # noqa: E402

SPLITS = ('train', 'validation', 'test')
CAMERA = pa.struct([('K', pa.list_(pa.list_(pa.float64(), 3), 3)), ('R', pa.list_(pa.list_(pa.float64(), 4), 4)),
                    ('resolution', pa.list_(pa.int32(), 2)), ('fov_y_deg', pa.float64()),
                    ('distance', pa.float64()), ('target', pa.list_(pa.float64(), 3))])
SCHEMA = pa.schema([
    ('scene', pa.string()), ('split', pa.string()), ('num_objects', pa.int32()),
    ('frames', pa.list_(pa.int32())), ('step_sizes', pa.list_(pa.int32())),
    ('camera', CAMERA),
    ('depth', pa.list_(pa.binary())), ('segmentation', pa.list_(pa.binary())),
    ('num_points', pa.list_(pa.list_(pa.int32()))),
    ('points', pa.list_(pa.list_(pa.list_(pa.list_(pa.float32(), 3))))),
    ('shape', pa.list_(pa.string())), ('size', pa.list_(pa.float64())),
    ('positions', pa.list_(pa.list_(pa.list_(pa.float32(), 3)))),
    ('quaternions', pa.list_(pa.list_(pa.list_(pa.float32(), 4)))),
])
_PV = None


def png(image: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format='PNG', optimize=False, compress_level=6)
    return buf.getvalue()


def _init(partial_cfg: dict):
    global _PV
    _PV = PartialView(partial_cfg, REPO)


def scene_row(args):
    row, frames, steps = args
    pv = _PV
    shapes, sizes = list(row['shape']), np.asarray(row['size'], np.float64)
    P, Q = np.asarray(row['positions'], np.float32), np.asarray(row['quaternions'], np.float32)
    K, R, H, W = pv.camera(row['scene'], False, np.random.default_rng(0))
    depth_png, seg_png, counts, points = [], [], [], []
    for f in frames:
        pos, quat = P[f].astype(np.float64), Q[f].astype(np.float64)
        depth, seg = pv.images(row['scene'], shapes, sizes, pos, quat)
        mm = np.where(np.isfinite(depth) & (depth > 0) & (depth < 65.5355), np.round(depth * 1000.0), 0)
        depth_png.append(png(mm.astype(np.uint16)))
        seg_png.append(png((seg + 1).astype(np.uint8)))
        # the evaluation's observation of this frame (multi_dataloader._partial_item: rng seeded by scene and frame)
        rng = np.random.default_rng(zlib.crc32(f"{row['scene']}:{f}".encode()))
        obs = pv.observe(row['scene'], shapes, sizes, pos, quat, False, rng, camera=(K, R, H, W))
        counts.append([0 if o is None else int(len(o)) for o in obs])
        points.append([np.zeros((0, 3), np.float32) if o is None else o.astype(np.float32) for o in obs])
    return {
        'scene': row['scene'], 'split': row['split'], 'num_objects': len(shapes), 'frames': list(frames),
        'step_sizes': list(steps),
        'camera': {'K': K.tolist(), 'R': R.tolist(), 'resolution': [H, W], 'fov_y_deg': float(pv.cfg['fov_y_deg']),
                   'distance': float(pv.cfg['distance']), 'target': [float(v) for v in pv.cfg['target']]},
        'depth': depth_png, 'segmentation': seg_png, 'num_points': counts,
        'points': [[p.tolist() for p in per_frame] for per_frame in points],
        'shape': shapes, 'size': sizes.tolist(), 'positions': P.tolist(), 'quaternions': Q.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='local copy of the dataset repository (data/<name>/*.parquet)')
    ap.add_argument('--name', required=True, choices=['movi_sphere', 'movi_a', 'movi_b'])
    ap.add_argument('--config', required=True, help='partial-view training config (data.partial_view, data.*)')
    ap.add_argument('--out', required=True, help='dataset repository to write data/<name>_partial/ into')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scenes', type=int, default=0, help='only the first N scenes of every split (quick checks)')
    ap.add_argument('--shard_rows', type=int, default=240, help='scenes per Parquet file')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO / args.config if not Path(args.config).is_absolute() else args.config))
    data = cfg['data']
    # observed points only, without completion
    partial = {k: v for k, v in data['partial_view'].items() if k not in ('completion', 'completion_input_points')}
    warmup = int(cfg['training'].get('metrics_test', {}).get('eval_warmup_frames', 2))
    k = int(partial['observe_index'])
    if k != warmup - 1:
        raise SystemExit(f'observe_index {k} is not the last of the {warmup} evaluation warm-up frames')
    steps = sorted(int(s) for s in data['eval_step_sizes'])
    # evaluation rollouts start at frame 0 (src/training/evaluator.py does not apply data.skip_first_frames), so step
    # size s observes frame k * s
    frames = [k * s for s in steps]
    sizes_file = json.loads((REPO / data['object_scale_file']).read_text())
    out_dir = Path(args.out) / 'data' / f'{args.name}_partial'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{args.name}: observation frames {frames} (step sizes {steps}); '
          f'{PartialView(partial, REPO).describe(train=False)}', flush=True)
    t0 = time.time()
    for split in SPLITS:
        rows = []
        for f in sorted((Path(args.dataset) / 'data' / args.name).glob(f'{split}-*.parquet')):
            rows += pq.read_table(f, columns=['scene', 'split', 'shape', 'size', 'positions', 'quaternions']).to_pylist()
        if args.scenes:
            rows = rows[:args.scenes]
        for r in rows:                                   # sizes must match data.object_scale_file
            ref = sizes_file['scales'][r['scene']]
            if len(ref) != len(r['size']) or not np.allclose(ref, r['size'], rtol=0, atol=1e-9):
                raise SystemExit(f"{r['scene']}: sizes differ from {data['object_scale_file']}")
        with get_context('fork').Pool(args.workers, initializer=_init, initargs=(partial,)) as pool:
            out = pool.map(scene_row, [(r, frames, steps) for r in rows], chunksize=1)
        n_files = max(1, -(-len(out) // args.shard_rows))
        for old in out_dir.glob(f'{split}-*.parquet'):
            old.unlink()
        for j in range(n_files):
            part = out[j * args.shard_rows:(j + 1) * args.shard_rows]
            pq.write_table(pa.Table.from_pylist(part, schema=SCHEMA),
                           out_dir / f'{split}-{j:05d}-of-{n_files:05d}.parquet', compression='zstd', row_group_size=16)
        seen = np.array([c for r in out for per_frame in r['num_points'] for c in per_frame])
        print(f'{split}: {len(out)} scenes, {n_files} file(s); observed objects {np.mean(seen > 0):.1%}, points per '
              f'observed object median {np.median(seen[seen > 0]):.0f}; {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
