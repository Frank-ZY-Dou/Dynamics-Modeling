"""Data of the interactive demo (demo/index.html) from exported rollouts (scripts/export_rollouts.py).

For each dataset, the chosen test scenes with, per step size, the predicted pose of every object in every frame (the
first frames are the given context frames); and the visual meshes of the shapes they use (positions, the meshes' own
normals, triangles).

    python scripts/build_demo_data.py --rollouts MOVi-A=rollouts/movi_a MOVi-B=rollouts/movi_b \\
        MOVi-Sphere=rollouts/movi_sphere --scenes 6 --out demo/data

Writes <out>/meshes.json, <out>/manifest.json and <out>/<dataset>.json (lower case, e.g. movi-b.json); arrays are
little-endian binary in base64.

Scenes: --scenes N picks, among the test scenes with at least the median amount of motion (summed path length of the
objects), the N with the lowest prediction error. The error of a scene is its position error (m) and rotation error
(deg), averaged over the objects, at the evaluation horizons (physical frames 25, 50, 75, 100, 200, 300 and 400) that
lie on the step-size grid after the context frames, as in the evaluation, each relative to the median over the test
scenes and averaged, so that every horizon and step size weighs the same. The ground truth is the scene's vis_traj.npy in the dataset folder
(--data, default datasets/<dataset name>). --pick MOVi-B=movi_4,movi_40 names the scenes instead.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.utils.rollout_render import load_render_meshes  # noqa: E402

MESHES = os.path.join(REPO, 'data_generation', 'objects')
FRAME_RATE = 240                                     # physical frames per second of the MOVi conversions


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).astype(a.dtype.newbyteorder('<')).tobytes()).decode('ascii')


def load(folder, scene, step):
    d = np.load(os.path.join(folder, f'{scene}_ss{step}.npz'))
    return {k: d[k] for k in d.files}


def quat_angle(q, g):
    """Angle (deg) between rotations given as (w, x, y, z) quaternions."""
    d = np.abs((q * g).sum(-1) / (np.linalg.norm(q, axis=-1) * np.linalg.norm(g, axis=-1)))
    return np.degrees(2 * np.arccos(np.clip(d, 0.0, 1.0)))


HORIZONS = (25, 50, 75, 100, 200, 300, 400)


def scene_errors(folder, scene, steps, context, traj):
    """{(step, horizon): (position error, rotation error)}, averaged over the objects, at each horizon on the step-size
    grid after the context frames (as the evaluation); traj (T, n, 7) is the ground truth (position, (w, x, y, z)) of
    every physical frame."""
    out = {}
    for step in steps:
        r = load(folder, scene, step)
        gt = traj[r['frame']].astype(np.float64)
        if 'object_index' in r:                     # rollouts from partial point clouds: the observed objects
            gt = gt[:, np.asarray(r['object_index'])]
        if np.abs(r['translation'][:context] - gt[:context, :, :3]).max() > 1e-4:
            raise ValueError(f'{folder}/{scene} step size {step}: the context frames differ from the ground truth')
        for h in HORIZONS:
            k = h // step
            if h % step or k < context:             # between two predicted frames, or a context frame
                continue
            pos = np.linalg.norm(r['translation'][k] - gt[k, :, :3], axis=-1).mean()
            rot = quat_angle(r['rotation'][k].astype(np.float64), gt[k, :, 3:]).mean()
            out[(step, h)] = (float(pos), float(rot))
    return out


def select(folder, index, data_dir, n):
    """The n test scenes with the lowest relative error among those with at least the median motion."""
    steps, context = index['step_sizes'], index['warmup_frames']
    motion, errors = {}, {}
    for scene in index['scenes']:
        traj = np.load(os.path.join(data_dir, scene, 'vis_traj.npy'))[0]
        motion[scene] = float(np.linalg.norm(np.diff(traj[:, :, :3], axis=0), axis=-1).sum())
        errors[scene] = scene_errors(folder, scene, steps, context, traj)
    cells = list(errors[index['scenes'][0]])
    med = {c: [np.median([errors[s][c][j] for s in errors]) for j in (0, 1)] for c in cells}
    score = {s: float(np.mean([(e[c][0] / med[c][0] + e[c][1] / med[c][1]) / 2 for c in cells]))
             for s, e in errors.items()}
    busy = [s for s in index['scenes'] if motion[s] >= np.median(list(motion.values()))]
    chosen = sorted(busy, key=lambda s: score[s])[:n]
    for s in chosen:
        e = errors[s]
        print(f'  {s}: error {score[s]:.2f} x median; step size 1 @100 {e[(1, 100)][0] * 100:.1f} cm '
              f'{e[(1, 100)][1]:.1f} deg, @400 {e[(1, 400)][0] * 100:.1f} cm {e[(1, 400)][1]:.1f} deg')
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rollouts', nargs='+', required=True, help='<dataset name>=<export folder>')
    ap.add_argument('--scenes', type=int, default=6, help='scenes per dataset, by lowest prediction error')
    ap.add_argument('--data', nargs='*', default=[], help='<dataset name>=<dataset folder> (default datasets/<name>)')
    ap.add_argument('--pick', nargs='*', default=[], help='<dataset name>=<scene>,<scene>,...')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    picks = {k: v.split(',') for k, v in (a.split('=', 1) for a in args.pick)}
    data = dict(a.split('=', 1) for a in args.data)
    os.makedirs(args.out, exist_ok=True)

    shapes, manifest = set(), []
    for spec in args.rollouts:
        name, folder = spec.split('=', 1)
        index = json.load(open(os.path.join(folder, 'index.json')))
        steps = index['step_sizes']
        scenes = picks.get(name) or select(folder, index, data.get(name, os.path.join(REPO, 'datasets', name)),
                                           args.scenes)
        out = {'name': name, 'frame_rate': FRAME_RATE, 'context_frames': index['warmup_frames'],
               'step_sizes': steps, 'checkpoint_sha256': index['checkpoint_sha256'], 'scenes': []}
        for scene in scenes:
            first = load(folder, scene, steps[0])
            objects = [{'shape': str(s), 'size': round(float(z), 6), 'material': str(m)}
                       for s, z, m in zip(first['shape'], first['size'], first['material'])]
            shapes |= {o['shape'] for o in objects}
            rollouts = {}
            for step in steps:
                r = load(folder, scene, step)
                if [str(s) for s in r['shape']] != [o['shape'] for o in objects]:
                    raise ValueError(f'{folder}/{scene}: objects differ between step sizes')
                q = r['rotation'][..., [1, 2, 3, 0]]                         # (w, x, y, z) -> (x, y, z, w)
                pose = np.concatenate([r['translation'], q], axis=-1).astype(np.float32)   # (K, n, 7)
                rollouts[str(step)] = {'frames': int(len(pose)), 'first_frame': int(r['frame'][0]),
                                       'pose': b64(pose)}
            out['scenes'].append({'id': scene, 'objects': objects, 'rollouts': rollouts})
        path = os.path.join(args.out, f'{name.lower()}.json')
        json.dump(out, open(path, 'w'), separators=(',', ':'))
        manifest.append({'name': name, 'file': os.path.basename(path), 'scenes': scenes})
        print(f'{name}: {len(scenes)} scenes {scenes} -> {path} ({os.path.getsize(path) / 1e6:.2f} MB)')

    meshes = {}
    for s, (V, N, F) in sorted(load_render_meshes(MESHES, sorted(shapes)).items()):
        meshes[s] = {'position': b64(V.astype(np.float32)), 'normal': b64(np.round(N * 127).astype(np.int8)),
                     'index': b64(F.astype(np.uint16 if len(V) < 65536 else np.uint32)),
                     'index_type': 'uint16' if len(V) < 65536 else 'uint32',
                     'radius': round(float(np.linalg.norm(V, axis=1).max()), 6)}
    path = os.path.join(args.out, 'meshes.json')
    json.dump(meshes, open(path, 'w'), separators=(',', ':'))
    json.dump({'datasets': manifest}, open(os.path.join(args.out, 'manifest.json'), 'w'), indent=1)
    print(f'{len(meshes)} meshes -> {path} ({os.path.getsize(path) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
