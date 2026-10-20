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
With a partial-view config (data.partial_view), the objects are the observed ones, and also
    object_index  int32   (objects,)             the object's index in the scene
    points        float32 (sum of counts, 3)     observed points in the object's frame (unit size), object by object
    points_count  int32   (objects,)
and with a completion network (data.partial_view.completion), whose points join the observed ones in the model's input
(the farthest-point subset of both), `points` are what the camera saw and
    added_points  float32 (sum of counts, 3)     the points the completion network predicted, in the same frames
    added_count   int32   (objects,)
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
import zlib
from pathlib import Path

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
    ap.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                    help='CUDA_VISIBLE_DEVICES for the run (default: the one already set, e.g. by Slurm, else 0)')
    ap.add_argument('--split', default='test', choices=['test', 'val'],
                    help='val: the validation scenes, evaluated with the test protocol (as eval_cross_dataset.py --split val)')
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
    if args.split == 'val':
        src_split = cfg['data']['split_file']
        sp = json.load(open(src_split if os.path.isabs(src_split) else os.path.join(REPO, src_split)))
        sp['test_datasets'] = list(sp['val_datasets'])
        sp['evaluated_partition'] = 'val'
        val_split = os.path.join(REPO, 'configs', f'_tmp_{tag}_valsplit.json')
        json.dump(sp, open(val_split, 'w'), indent=1)
        cfg['data']['split_file'] = val_split
    warmup = int(cfg['training'].get('metrics_test', {}).get('eval_warmup_frames', 2))   # as the evaluator reads it
    tmp_cfg = os.path.join(REPO, 'configs', f'_tmp_{tag}.yaml')
    yaml.safe_dump(cfg, open(tmp_cfg, 'w'), sort_keys=False)

    from src.utils.rollout_render import collision_vertices, fit_poses
    data_path = cfg['data']['dataset_path']
    data_path = data_path if os.path.isabs(data_path) else os.path.join(REPO, data_path)
    pv_cfg = {k: v for k, v in cfg['data'].get('partial_view', {}).items()}
    camera_only = completer = scales = None
    if pv_cfg.get('enable') and pv_cfg.get('completion'):        # what the camera saw, and what the network added
        from src.data.object_scale import ObjectScale
        from src.data.partial_view import PartialView
        camera_only = PartialView({**pv_cfg, 'completion': None}, Path(REPO))
        completer = PartialView(pv_cfg, Path(REPO)).completer
        sf = cfg['data']['object_scale_file']
        scales = ObjectScale(sf if os.path.isabs(sf) else os.path.join(REPO, sf)).scales
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
            if 'partial_obj_index' in initial_batch:                  # data.partial_view inputs
                records[(name, int(round(float(steps[b]))))] = partial_record(initial_batch, b, pred[b], info)
                continue
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

    def partial_record(batch, b, pred, info):
        """Record of a rollout from partial point clouds, for its observed objects. The Kabsch motion from the observed
        points at the observation frame (ground truth) to their predictions in every frame moves the object's collision
        mesh at the observation frame, as the evaluator scores it; the poses are fitted to the moved mesh. Also holds
        the observed points in each object's frame (unit size), for drawing."""
        from src.data.partial_view import kabsch
        observed = np.asarray(batch['partial_obj_index'][b], np.int64)
        k = int(batch['partial_obs_index'][b])
        full = batch['partial_full_obs'][b]
        vmask = batch['valid_vtx_mask'][b].cpu().numpy().astype(bool)                # (T0, observed, points)
        K, n = pred.shape[0], len(observed)
        objects = [info[f'obj_{i}'] for i in observed]
        quat, trans, resid, points = np.zeros((K, n, 4), np.float32), np.zeros((K, n, 3), np.float32), 0.0, []
        for j, (i, o) in enumerate(zip(observed, objects)):
            m = vmask[k, j]
            R, t = kabsch(pred[k, j, m].astype(np.float64), pred[:, j, m].astype(np.float64))
            frames = np.einsum('tab,nb->tna', R, np.asarray(full[i], np.float64)) + t[:, None, :]
            P, r = fit_poses(float(o['size']) * collision_vertices(MESHES, o['shape']), frames)
            quat[:, j] = Rotation.from_matrix(P[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
            trans[:, j] = P[:, :3, 3]
            resid = max(resid, r)
            points.append((pred[k, j, m].astype(np.float64) - P[k, :3, 3]) @ P[k, :3, :3] / float(o['size']))
        if resid > 1e-3:
            raise ValueError(f'{batch["dataset_name"][b]}: the observation-frame meshes are {resid:.2e} m from the '
                             f'collision meshes at their poses')
        step = int(round(float(batch['skip_step_size'][b])))
        rec = {'rotation': quat, 'translation': trans, 'frame': (np.arange(K) * step).astype(np.int32),
               'residual': resid, 'shape': np.array([o['shape'] for o in objects]),
               'material': np.array([o['material'] for o in objects]),
               'size': np.array([o['size'] for o in objects], np.float32), 'object_index': observed.astype(np.int32)}
        if camera_only is not None:
            inputs = points
            # observation frame from the sample's frame ids: evaluation samples start at frame 0, not skip_first_frames
            points, added = camera_and_completion(batch['dataset_name'][b], int(batch['frame_ids'][b][k]), observed)
            for j, (u, c, a) in enumerate(zip(inputs, points, added)):    # the input is a subset of the two
                ref = np.concatenate([c, a])
                gap = max(float(np.sqrt(((ref - x) ** 2).sum(1)).min()) for x in u)
                if gap > 5e-3:
                    raise ValueError(f'{batch["dataset_name"][b]}: object {int(observed[j])}: an input point is '
                                     f'{gap:.2e} from the camera and completion points (not the evaluation\'s observation)')
            rec['added_points'] = np.concatenate(added).astype(np.float32)
            rec['added_count'] = np.array([len(u) for u in added], np.int32)
        rec['points'] = np.concatenate(points).astype(np.float32)
        rec['points_count'] = np.array([len(u) for u in points], np.int32)
        return rec

    def camera_and_completion(name, t_obs, observed):
        """The evaluation's observation (same camera, frame and random stream as the data loader's), split into the
        points the camera saw and those the completion network predicted: one array per observed object, in the
        object's frame (unit size)."""
        from src.data.partial_view import quat_to_rotmat
        scene = os.path.join(data_path, name)
        shapes = [o['asset_id'] for o in json.load(open(os.path.join(scene, 'conversion.json')))['objects']]
        sizes = np.asarray(scales[name], np.float64)
        traj = np.load(os.path.join(scene, 'vis_traj.npy'))[0]
        pos, quat = traj[t_obs, :, :3].astype(np.float64), traj[t_obs, :, 3:7].astype(np.float64)
        rng = np.random.default_rng(zlib.crc32(f'{name}:{t_obs}'.encode()))
        seen = camera_only.observe(name, shapes, sizes, pos, quat, False, rng)
        _, Rc, _, _ = camera_only.camera(name, False, rng)
        Rq = quat_to_rotmat(quat)
        ids = [i for i in range(len(shapes)) if seen[i] is not None]
        pred = completer.complete([pos[i] + sizes[i] * seen[i] @ Rq[i].T for i in ids], Rc[:3, 3])
        added = {i: (Y - pos[i]) @ Rq[i] / sizes[i] for i, Y in zip(ids, pred)}
        if any(seen[i] is None for i in observed):
            raise ValueError(f'{name}: the camera-only observation misses objects the evaluation observed')
        return [seen[i] for i in observed], [added[i] for i in observed]

    train.CUT3RDynamicsModel.rollout = recording_rollout
    sys.argv = ['train.py', '--config', tmp_cfg, '--checkpoint', args.checkpoint, '--inference', '--use_multi_dataset',
                '--cuda-graph']
    try:
        train.main()
    finally:
        os.remove(tmp_cfg)
        if args.split == 'val':
            os.remove(cfg['data']['split_file'])

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
