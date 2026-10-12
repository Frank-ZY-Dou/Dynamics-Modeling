"""Exhaustive check of the anchor and FPS-order caches the loader consumes, for every object of every scene of
MOVi-Sphere/A/B: each cache is recomputed and must be equal.

  anchors/obj_<i>_anchors_fps<K>.npy      written by the converter, K in 1..16       FPS rule index0 on the asset mesh
  anchors/obj_<i>_anchors_fps4_cf.npy     the anchors the configs use                FPS rule centroid_farthest, frame 0
  anchors/obj_<i>_fpsorder_cf.npy         the hierarchical encoder subsets           full FPS order, frame 0

The centroid_farthest caches were generated from the stored (unscaled) frame-0 vertices (pregen_cf_anchors.py,
pregen_fps_order.py); FPS is invariant to the uniform size scaling, and the check also recomputes them on the scaled
vertices (what the loader computes when a cache is missing) and reports any difference.

  python scripts/verify_caches.py --out <report dir> [--workers 48]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'data_generation'))
from src.data.multi_dataloader import MultiObjectDynamics  # noqa: E402
from movi_to_sim_data import ANCHOR_COUNTS, _compute_fps_anchors  # noqa: E402

FPS = MultiObjectDynamics._farthest_point_sample_np
SETS = {'S': ('datasets/MOVi-Sphere', 'MoviSphere'), 'A': ('datasets/MOVi-A', 'MoviA'), 'B': ('datasets/MOVi-B', 'MoviB')}
_assets = {}


def check_scene(args):
    conv, scene, sizes = args
    d = os.path.join(REPO, conv, scene)
    phys = json.load(open(os.path.join(d, 'physics_params.json')))
    p0 = np.load(os.path.join(d, 'vis_traj.npy'))[0, 0, :, :3].astype(np.float64)
    out = {'scene': scene, 'objects': 0, 'bad': [], 'scaled_differs': 0}
    for o in range(len(sizes)):
        shape = phys[f'obj_{o}']['shape']
        if shape not in _assets:
            _assets[shape] = np.asarray(trimesh.load(os.path.join(REPO, 'data_generation', 'objects', shape,
                                                                  'collision_geometry.obj'), process=False).vertices)
        A = _assets[shape].astype(np.float32)
        V = np.load(os.path.join(d, 'frame_00000', f'obj_{o}_pos.npy')).astype(np.float32)
        ad = os.path.join(d, 'anchors')
        for K in ANCHOR_COUNTS:
            if not np.array_equal(np.load(os.path.join(ad, f'obj_{o}_anchors_fps{K}.npy')), _compute_fps_anchors(A, K)):
                out['bad'].append(f'obj {o} anchors_fps{K}')
        cf = FPS(V, 4, rule='centroid_farthest')
        order = FPS(V, V.shape[0], rule='centroid_farthest')
        if not np.array_equal(np.load(os.path.join(ad, f'obj_{o}_anchors_fps4_cf.npy')), cf):
            out['bad'].append(f'obj {o} anchors_fps4_cf')
        if not np.array_equal(np.load(os.path.join(ad, f'obj_{o}_fpsorder_cf.npy')), order):
            out['bad'].append(f'obj {o} fpsorder_cf')
        Vs = (p0[o] + sizes[o] * (V.astype(np.float64) - p0[o])).astype(np.float32)
        if not (np.array_equal(FPS(Vs, 4, rule='centroid_farthest'), cf) and
                np.array_equal(FPS(Vs, Vs.shape[0], rule='centroid_farthest'), order)):
            out['scaled_differs'] += 1
        out['objects'] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    failed = False
    for D, (conv, name) in SETS.items():
        scales = json.load(open(f'data_generation/object_scales_{name}.json'))['scales']
        sp = json.load(open(f'dataset_split_{name}_1200.json'))
        scenes = sp['train_datasets'] + sp['val_datasets'] + sp['test_datasets']
        with Pool(a.workers) as pool:
            res = pool.map(check_scene, [(conv, s, scales[s]) for s in scenes], chunksize=2)
        bad = [(r['scene'], b) for r in res for b in r['bad']]
        text = (f"== {D}: {len(res)} scenes, {sum(r['objects'] for r in res)} objects: converter anchors (K in "
                f"{list(ANCHOR_COUNTS)}), centroid-farthest anchors (K=4) and full FPS orders recomputed; mismatches "
                f"{len(bad)} {bad[:5]}; objects whose FPS on the size-scaled vertices differs: "
                f"{sum(r['scaled_differs'] for r in res)}")
        print(text, flush=True)
        open(os.path.join(a.out, f'{D}.txt'), 'w').write(text + '\n')
        failed = failed or bool(bad)
    if failed:
        raise SystemExit('verify_caches: mismatches found (see the reports)')


if __name__ == '__main__':
    main()
