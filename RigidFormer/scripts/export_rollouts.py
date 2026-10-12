"""Rollouts of a checkpoint as rigid object poses, for visualization.

Runs the evaluation of scripts/eval.sh in this process (same configuration, data, rollout code and metrics) on the test
scenes of one dataset and records every rollout: for each scene and step size, the pose of every object in each
predicted frame, fitted with the Kabsch algorithm to the predicted vertices and the object's collision mesh at its size
(the predictions are rigid; the largest fitting residual is recorded and must stay below 1 mm). The pose maps the
object's own frame (its collision and visual meshes, unit size) to the world: x = R (size v) + t.

Output, per scene and step size, <out>/<scene>_ss<step>.npz:
    rotation      float32 (frames, objects, 4)   (w, x, y, z)
    translation   float32 (frames, objects, 3)   m
    frame         int32   (frames,)              physical frame of each predicted frame (k * step size)
    shape, material (objects,) str; size (objects,) float32
and <out>/index.json (scenes, step sizes, warm-up frames, residuals, checkpoint SHA-256) and the evaluation record
<out>/eval.txt. The first `warmup_frames` frames of every rollout are the given context frames.

    python scripts/export_rollouts.py --checkpoint checkpoints/rigidformer_movib.pth --config configs/movi_b.yaml \\
        --target B --out rollouts/movi_b [--gpu 0]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import socket
import sys
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, REPO)
MESHES = os.path.join(REPO, 'data_generation', 'objects')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True, help='the config the checkpoint was trained with')
    ap.add_argument('--target', required=True, choices=['S', 'A', 'B'], help='test scenes of MOVi-Sphere, MOVi-A or MOVi-B')
    ap.add_argument('--out', required=True, help='new output folder')
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f'{args.out} exists: export into a new folder')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    os.environ.update({'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1', 'MASTER_ADDR': '127.0.0.1',
                       'MASTER_PORT': str(port)})

    import train                                      # src/train.py
    from eval_cross_dataset import build_target_config

    tag = f'export_{args.target}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    cfg = build_target_config(yaml.safe_load(open(args.config)), args.target, tag)
    warmup = int(cfg['training'].get('metrics_test', {}).get('eval_warmup_frames', 2))   # as the evaluator reads it
    tmp_cfg = os.path.join(REPO, 'configs', f'_tmp_{tag}.yaml')
    yaml.safe_dump(cfg, open(tmp_cfg, 'w'), sort_keys=False)

    from src.utils.rollout_render import collision_vertices, fit_poses
    data_path = cfg['data']['dataset_path']
    data_path = data_path if os.path.isabs(data_path) else os.path.join(REPO, data_path)
    records = {}
    rollout = train.CUT3RDynamicsModel.rollout

    def recording_rollout(self, initial_batch, num_steps, dt=0.01, return_extras=True, warmup_frames=2):
        out = rollout(self, initial_batch, num_steps, dt=dt, return_extras=return_extras, warmup_frames=warmup_frames)
        if warmup_frames != warmup:                   # only the evaluation's rollouts
            return out
        pred = out['obj_positions'].float().cpu().numpy()                 # (B, K, objects, vertices, 3)
        vmask = initial_batch['valid_vtx_mask'][:, 0].cpu().numpy().astype(bool)
        omask = initial_batch['valid_obj_mask'].cpu().numpy().astype(bool)
        steps = initial_batch['skip_step_size'].cpu().numpy()
        for b, name in enumerate(initial_batch['dataset_name']):
            n = int(omask[b].sum())
            if not omask[b, :n].all():
                raise ValueError(f'{name}: objects are not the first {n} slots')
            info = json.load(open(os.path.join(data_path, name, 'physics_params.json')))
            objects = [info[f'obj_{i}'] for i in range(n)]
            K = pred.shape[1]
            quat, trans, resid = np.zeros((K, n, 4), np.float32), np.zeros((K, n, 3), np.float32), 0.0
            for i, o in enumerate(objects):
                ref = float(o['size']) * collision_vertices(MESHES, o['shape'])
                if int(vmask[b, i].sum()) != len(ref) or not vmask[b, i, :len(ref)].all():
                    raise ValueError(f"{name}: object {i} has {int(vmask[b, i].sum())} vertices, its {o['shape']} "
                                     f"collision mesh {len(ref)}")
                P, r = fit_poses(ref, pred[b, :, i, :len(ref)])
                quat[:, i] = Rotation.from_matrix(P[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
                trans[:, i] = P[:, :3, 3]
                resid = max(resid, r)
            if resid > 1e-3:
                raise ValueError(f'{name}: predicted vertices are {resid:.2e} m from a rigid motion of the objects')
            step = int(round(float(steps[b])))
            records[(name, step)] = {'rotation': quat, 'translation': trans,
                                     'frame': (np.arange(K) * step).astype(np.int32), 'residual': resid,
                                     'shape': np.array([o['shape'] for o in objects]),
                                     'material': np.array([o['material'] for o in objects]),
                                     'size': np.array([o['size'] for o in objects], np.float32)}
        return out

    train.CUT3RDynamicsModel.rollout = recording_rollout
    sys.argv = ['train.py', '--config', tmp_cfg, '--checkpoint', args.checkpoint, '--inference', '--use_multi_dataset',
                '--cuda-graph']
    try:
        train.main()
    finally:
        os.remove(tmp_cfg)

    os.makedirs(args.out)
    for (name, step), rec in sorted(records.items()):
        np.savez(os.path.join(args.out, f'{name}_ss{step}.npz'),
                 **{k: v for k, v in rec.items() if k != 'residual'})
    run = sorted(glob.glob(os.path.join(REPO, 'exps', 'logs_multi', tag, 'run_*')))[-1]
    shutil.copyfile(os.path.join(run, 'eval.txt'), os.path.join(args.out, 'eval.txt'))
    index = {'target': args.target, 'config': os.path.basename(args.config),
             'checkpoint_sha256': sha256(args.checkpoint), 'warmup_frames': warmup,
             'scenes': sorted({n for n, _ in records}), 'step_sizes': sorted({s for _, s in records}),
             'max_residual': max(r['residual'] for r in records.values())}
    json.dump(index, open(os.path.join(args.out, 'index.json'), 'w'), indent=1)
    print(f"{len(records)} rollouts -> {args.out}; largest Kabsch residual {index['max_residual']:.2e} m")


if __name__ == '__main__':
    main()
