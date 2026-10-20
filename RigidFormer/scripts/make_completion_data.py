"""Observations to train and select the completion network of data.partial_view.completion (src/data/shape_completion.py).

Each observed object of a scene, seen by data.partial_view's camera at a frame drawn uniformly, gives one record: what
the network reads (the first --input_points observed points, FPS order, in their view frame, and the viewing geometry)
and what it is scored against (the object's shape, size and pose in that frame; scripts/train_completion.py puts the
target points on the posed collision surface).

    --split train  the training scenes of the config's split file, random training cameras with the augmentations of
                   its data.partial_view (whole objects excepted);
    --split val    the validation scenes and the evaluation camera, for checkpoint selection only.
Test scenes are never read.

    python scripts/make_completion_data.py --config <training config> --split train --per_scene 20 \\
        --out completion/train.npz --workers 32
"""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.partial_view import PartialView, quat_to_rotmat, view_frame  # noqa: E402

SHAPES = ('cone', 'cube', 'cylinder', 'gear', 'sphere', 'sponge', 'spot', 'suzanne', 'teapot', 'torus', 'torus_knot')
_pv = None


def _init(pv_cfg):
    global _pv
    _pv = PartialView(pv_cfg, REPO)


def observe_scene(job):
    """Records of one scene: per observed object its inputs and its pose in the view frame."""
    root, scene, shapes, sizes, train, per_scene, input_points, seed = job
    traj = np.load(Path(root) / scene / 'vis_traj.npy')[0]
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(per_scene):
        t = int(rng.integers(traj.shape[0]))
        pos, quat = traj[t, :, :3].astype(np.float64), traj[t, :, 3:7].astype(np.float64)
        cam = _pv.camera(scene, train, rng)
        local = _pv.observe(scene, shapes, sizes, pos, quat, train, rng, camera=cam)
        R = quat_to_rotmat(quat)
        for i, U in enumerate(local):
            if U is None:
                continue
            X = pos[i] + sizes[i] * U @ R[i].T
            c, Rv, extra = view_frame(X, cam[1][:3, 3])
            v = ((X[:input_points] - c) @ Rv).astype(np.float16)
            recs.append((v, extra, SHAPES.index(shapes[i]), sizes[i], Rv.T @ R[i], (pos[i] - c) @ Rv, t))
    return scene, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='training config: data.dataset_path, split_file, object_scale_file, '
                                                    'partial_view')
    ap.add_argument('--split', choices=['train', 'val'], required=True)
    ap.add_argument('--per_scene', type=int, default=20, help='observations per scene')
    ap.add_argument('--input_points', type=int, default=256,
                    help='observed points kept per object, FPS order (data.partial_view.completion_input_points)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scenes', type=int, default=0,
                    help='only the first N scenes and the first N generated (aug_*) scenes of the split (quick checks)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))['data']
    rel = lambda p: Path(p) if Path(p).is_absolute() else REPO / p   # noqa: E731
    root = rel(cfg['dataset_path'])
    scenes = json.loads(rel(cfg['split_file']).read_text())[f'{args.split}_datasets']
    if args.scenes:
        scenes = scenes[:args.scenes] + [s for s in scenes if s.startswith('aug_')][:args.scenes]
    scales = json.loads(rel(cfg['object_scale_file']).read_text())['scales']
    pv_cfg = {k: v for k, v in cfg['partial_view'].items() if k not in ('completion', 'completion_input_points')}
    pv_cfg['full_object_prob'] = 0.0
    jobs = []
    for n, s in enumerate(scenes):
        shapes = [o['asset_id'] for o in json.loads((root / s / 'conversion.json').read_text())['objects']]
        jobs.append((str(root), s, shapes, np.asarray(scales[s], np.float64), args.split == 'train', args.per_scene,
                     args.input_points, args.seed * 1_000_003 + n))
    with multiprocessing.get_context('fork').Pool(args.workers, initializer=_init, initargs=(pv_cfg,)) as pool:
        out = pool.map(observe_scene, jobs, chunksize=4)
    recs = [(s, r) for s, rs in out for r in rs]
    M, P = len(recs), args.input_points
    pts = np.zeros((M, P, 3), np.float16)
    n = np.zeros(M, np.int16)
    for j, (_, r) in enumerate(recs):
        pts[j, :len(r[0])] = r[0]
        n[j] = len(r[0])
    names = sorted({s for s, _ in recs})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pts=pts, n=n, extra=np.stack([r[1] for _, r in recs]).astype(np.float32),
             shape=np.array([r[2] for _, r in recs], np.int8), size=np.array([r[3] for _, r in recs], np.float32),
             rot=np.stack([r[4] for _, r in recs]).astype(np.float32),
             trans=np.stack([r[5] for _, r in recs]).astype(np.float32),
             frame=np.array([r[6] for _, r in recs], np.int16),
             scene=np.array([names.index(s) for s, _ in recs], np.int32), scene_names=np.array(names),
             shapes=np.array(SHAPES), split=args.split, config=str(args.config))
    per_shape = np.bincount(np.array([r[2] for _, r in recs]), minlength=len(SHAPES))
    print(f'{args.out}: {M} objects from {len(scenes)} {args.split} scenes x {args.per_scene} views; points per object '
          f'median {int(np.median(n))}; per shape ' + ', '.join(f'{s} {c}' for s, c in zip(SHAPES, per_shape)))


if __name__ == '__main__':
    main()
