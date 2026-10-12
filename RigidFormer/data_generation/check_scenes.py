"""Scene completeness: does a MOVi scene's metadata hold every simulated object?

The MOVi generator (HopNet's data/movi_worker.py, from Kubric's movi_ab_worker.py) simulates `num_objects` objects but
writes only the ones visible in at least one rendered frame into metadata.json (instances = visible_foreground_assets).
A never-visible object still collides with the recorded ones; replay_movi_scene.py recovers it.

A scene is complete when the number of recorded (non-floor) instances equals the number of simulated objects, replayed
from the seed the generator used (metadata.seed): RandomState(seed), the CLEVR light jitter (4 x rand(3)), for the
CLEVR camera its jitter (rand(3)), then randint(min_num_objects, max_num_objects + 1). As a consistency check a complete
scene must be free of unexplained impulses: a change of an object's velocity between consecutive frames beyond gravity
(|v_t - v_{t-1} - g dt| > 0.1 m/s) while, at frames t-1 and t, its scaled collision mesh is more than 3 cm above the
floor and its axis-aligned box more than 3 cm from every other object's box (box gaps never exceed mesh gaps).

Check every scene of a split (complete, no unexplained impulse):

  python data_generation/check_scenes.py --raw raw/MOVi-B-complete --converted datasets/MOVi-B \
      --scales data_generation/object_scales_MoviB.json --split dataset_split_MoviB_1200.json
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

DV = 0.1      # m/s, velocity change between consecutive frames beyond gravity
GAP = 0.03    # m, no floor or exported object within this distance at frames t-1 and t


def replay_num_objects(meta):
    """Number of objects the generator simulated, from the seed it used (see the module docstring).

    Kubric's setup replaces a false-valued flags.seed with a random one and records the seed actually used in
    metadata.seed: the replay uses that one, and a set flags.seed must agree with it.
    """
    flags = meta['flags']
    seed = meta['metadata']['seed']
    if flags.get('seed') and flags['seed'] != seed:
        raise ValueError(f"flags.seed {flags['seed']} differs from metadata.seed {seed}")
    rng = np.random.RandomState(seed=seed)
    for _ in range(4):                  # kb.assets.utils.get_clevr_lights(rng=rng): jitter of 4 lights
        rng.rand(3)
    if flags.get('camera') == 'clevr':  # camera position jitter
        rng.rand(3)
    return int(rng.randint(flags['min_num_objects'], flags['max_num_objects'] + 1))


def exported_instances(meta):
    return [i for i in meta['instances'] if (i.get('shape') or i.get('asset_id') or '').lower() != 'floor']


def unexplained_impulses(meta, scene_dir, sizes, dv=DV, gap=GAP):
    """[(object, frame, |dv| m/s, floor gap m, object gap m)] for the exported objects of one converted scene."""
    insts = exported_instances(meta)
    dt = 1.0 / meta['metadata']['frame_rate']
    g = np.array([0.0, 0.0, -10.0])
    vt = np.load(os.path.join(scene_dir, 'vis_traj.npy'))[0].astype(np.float64)
    n = len(insts)
    # every frame's axis-aligned boxes: the frame-0 mesh moved rigidly by the vis_traj poses (the converted vertices are
    # exactly this motion) and scaled by the metadata size
    lo, hi = np.zeros((len(vt), n, 3)), np.zeros((len(vt), n, 3))
    for o in range(n):
        V0 = np.load(os.path.join(scene_dir, 'frame_00000', f'obj_{o}_pos.npy')).astype(np.float64)
        R = Rotation.from_quat(vt[:, o, [4, 5, 6, 3]]).as_matrix()          # WXYZ -> scipy XYZW
        C = sizes[o] * (V0 - vt[0, o, :3]) @ R[0]                             # object frame, scaled
        W = vt[:, o, None, :3] + np.einsum('tij,nj->tni', R, C)
        lo[:, o], hi[:, o] = W.min(1), W.max(1)
    events = []
    for o, inst in enumerate(insts):
        v = np.asarray(inst['velocities'], dtype=np.float64)
        res = np.linalg.norm(v[1:] - v[:-1] - g * dt, axis=1)
        for t in (np.nonzero(res > dv)[0] + 1).tolist():
            floor_gap, obj_gap = np.inf, np.inf
            for tt in (t - 1, t):
                floor_gap = min(floor_gap, lo[tt, o, 2])
                for k in range(n):
                    if k != o:
                        obj_gap = min(obj_gap, np.linalg.norm(np.maximum(0.0, np.maximum(lo[tt, k] - hi[tt, o],
                                                                                         lo[tt, o] - hi[tt, k]))))
            if floor_gap > gap and obj_gap > gap:
                events.append((o, t, float(res[t - 1]), float(floor_gap), float(obj_gap)))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='complete raw scenes <raw>/<n>/metadata.json')
    ap.add_argument('--converted', required=True, help='converted root with movi_<n>/')
    ap.add_argument('--scales', required=True, help='object-size file (build_object_scales.py)')
    ap.add_argument('--split', required=True, help='split file')
    a = ap.parse_args()
    split = json.load(open(a.split))
    scales = json.load(open(a.scales))['scales']
    problems, n = [], 0
    for key in ('train_datasets', 'val_datasets', 'test_datasets'):
        for sc in split[key]:
            meta = json.load(open(os.path.join(a.raw, sc.split('_', 1)[1], 'metadata.json')))
            n_exp, n_sim = len(exported_instances(meta)), replay_num_objects(meta)
            events = unexplained_impulses(meta, os.path.join(a.converted, sc), scales[sc])
            if n_exp != n_sim:
                problems.append(f'{sc}: {n_exp} recorded objects, {n_sim} simulated')
            if events:
                problems.append(f'{sc}: unexplained impulses {events[:3]}')
            n += 1
    print(f'{a.split}: {n} scenes, {len(problems)} problems' + ''.join(f'\n  {p}' for p in problems[:20]))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
