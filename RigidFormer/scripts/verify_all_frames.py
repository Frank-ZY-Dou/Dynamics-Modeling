"""Exhaustive check of every stored vertex array against the simulated geometry: for every scene of MOVi-Sphere/A/B,
every frame and every object, the stored frame_<t>/obj_<i>_pos.npy must equal R(q_t) v_asset + p_t, with v_asset the
unscaled collision mesh of the object's shape (data_generation/objects/<shape>/collision_geometry.obj) and (p_t, q_t)
the metadata pose (the sizes are applied at load time). Also checks that every array is finite, that vis_traj.npy
equals the metadata poses, and that faces equal the asset faces.

  python scripts/verify_all_frames.py --out <report dir> [--raw <folder with the complete raw scenes>] [--workers 48]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get('MOVI_RAW', os.path.join(REPO, 'raw'))   # <root>/MOVi-{spheres,A,B}-complete/<n>/ (data_generation/README.md)
SETS = {'S': ('MOVi-spheres-complete', 'datasets/MOVi-Sphere', 'dataset_split_MoviSphere_1200.json'),
        'A': ('MOVi-A-complete', 'datasets/MOVi-A', 'dataset_split_MoviA_1200.json'),
        'B': ('MOVi-B-complete', 'datasets/MOVi-B', 'dataset_split_MoviB_1200.json')}
_assets = {}


def asset(shape):
    if shape not in _assets:
        m = trimesh.load(os.path.join(REPO, 'data_generation', 'objects', shape, 'collision_geometry.obj'), process=False)
        _assets[shape] = (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces))
    return _assets[shape]


def quat_to_matrix(q):
    """(T, 4) WXYZ unit quaternions -> (T, 3, 3) (trimesh.transformations.quaternion_matrix, vectorised)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def check_scene(args):
    raw, conv, scene = args
    meta = json.load(open(os.path.join(raw, scene.split('_')[1], 'metadata.json')))
    insts = [i for i in meta['instances'] if (i.get('shape') or i.get('asset_id') or '').lower() != 'floor']
    d = os.path.join(REPO, conv, scene)
    T = int(meta['metadata']['num_frames'])
    vt = np.load(os.path.join(d, 'vis_traj.npy'))[0]
    out = {'scene': scene, 'object_frames': 0, 'max_err': 0.0, 'worst': None, 'bad': [], 'nonfinite': 0}
    for o, inst in enumerate(insts):
        A, F = asset(inst.get('shape') or inst.get('asset_id'))
        P = np.asarray(inst['positions'], dtype=np.float64)
        Q = np.asarray(inst['quaternions'], dtype=np.float64)
        if not np.isfinite(A).all():
            out['bad'].append(f'obj {o}: non-finite asset vertices')
        qn = np.linalg.norm(Q, axis=1)
        if not (np.isfinite(qn).all() and np.abs(qn - 1.0).max() <= 1e-4):
            out['bad'].append(f'obj {o}: metadata quaternions are not unit')
        if not (np.isfinite(P).all() and np.isfinite(Q).all()):
            out['bad'].append(f'obj {o}: non-finite metadata pose')
        if not (np.array_equal(vt[:, o, :3], P.astype(np.float32)) and np.array_equal(vt[:, o, 3:7], Q.astype(np.float32))):
            out['bad'].append(f'obj {o}: vis_traj differs from the metadata pose (or is not finite)')
        if not np.array_equal(np.load(os.path.join(d, 'meshes', f'obj_{o}_faces.npy')), F):
            out['bad'].append(f'obj {o}: faces differ from the asset faces')
        R = quat_to_matrix(Q)
        for t in range(T):
            V = np.load(os.path.join(d, f'frame_{t:05d}', f'obj_{o}_pos.npy'))
            if V.shape != A.shape:
                out['bad'].append(f'obj {o} frame {t}: shape {V.shape} vs asset {A.shape}')
                continue
            if not np.isfinite(V).all():
                out['nonfinite'] += 1
                continue
            err = float(np.abs(V.astype(np.float64) - (A @ R[t].T + P[t])).max())
            if not np.isfinite(err):
                out['bad'].append(f'obj {o} frame {t}: non-finite expected geometry')
                continue
            out['object_frames'] += 1
            if err > out['max_err']:
                out['max_err'], out['worst'] = err, (o, t)
            if err > 1e-5:
                out['bad'].append(f'obj {o} frame {t}: max error {err:.2e} m')
    if len(os.listdir(os.path.join(d))) and len([f for f in os.listdir(d) if f.startswith('frame_')]) != T:
        out['bad'].append('frame directory count differs from num_frames')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw', default=RAW, help='folder of the complete raw scenes: <raw>/MOVi-{spheres,A,B}-complete/<n>/')
    ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    failed = False
    for D, (raw, conv, split_file) in SETS.items():
        sp = json.load(open(split_file))
        scenes = sp['train_datasets'] + sp['val_datasets'] + sp['test_datasets']
        with Pool(a.workers) as pool:
            res = pool.map(check_scene, [(os.path.join(a.raw, raw), conv, s) for s in scenes], chunksize=4)
        n = sum(r['object_frames'] for r in res)
        worst = max(res, key=lambda r: r['max_err'])
        bad = [(r['scene'], b) for r in res for b in r['bad']]
        text = (f"== {D}: {len(res)} scenes, {n} object-frames compared with R(q_t) v_asset + p_t (unscaled asset, metadata "
                f"pose): max abs error {worst['max_err']:.3e} m ({worst['scene']} obj/frame {worst['worst']}); non-finite "
                f"arrays {sum(r['nonfinite'] for r in res)}; problems {len(bad)} {bad[:5]}")
        print(text, flush=True)
        open(os.path.join(a.out, f'{D}.txt'), 'w').write(text + '\n')
        failed = failed or bool(bad) or any(r['nonfinite'] for r in res)
    if failed:
        raise SystemExit('verify_all_frames: problems found (see the reports)')


if __name__ == '__main__':
    main()
