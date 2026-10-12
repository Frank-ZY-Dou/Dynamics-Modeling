"""Resting height of MOVi-Sphere objects at the end of step-size-1 rollouts: predicted vs. ground-truth height.

Runs the standard test evaluation (src/train.py --inference, step size 1 only) with the evaluator's per-scene metric
function wrapped: for every object, the predicted origin height at the last evaluated frame (recovered from the predicted
vertices by the evaluator's HopNet shape matching, exact for a rigidly moved mesh; the vertex mean is not the origin)
minus the ground-truth origin height (vis_traj), split by ground-truth size (small / large spheres) for objects that rest
on the ground in the ground truth at that frame (centre height within 2 cm of the resting height of its size).
Only each object's own vertices are used (configs with a fixed max_vertices_per_object pad every object).

  CUDA_VISIBLE_DEVICES=0 python scripts/rest_height_check.py --config <config> --checkpoint <ckpt> --out <dir>
(the config's own evaluation setup is used; eval_step_sizes is forced to [1] and videos are off)
"""
import argparse
import atexit
import json
import os
import sys
import tempfile

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def measure(ev, pred_single, dataset_name, skip_step_size):
    """Per object at the last evaluated frame: ground-truth and predicted origin height, the predicted lowest vertex,
    and the horizontal origin error. The predicted origin comes from the evaluator's HopNet shape matching (exact for a
    rigidly moved mesh); the vertex mean is not the origin (the 64-vertex sphere's vertex mean is 22.5 mm off it)."""
    import torch
    import roma
    from pathlib import Path
    import src.training.evaluator as E
    root = Path(ev.config['data']['dataset_path']) / dataset_name
    traj = np.load(root / 'vis_traj.npy')[0][::skip_step_size]
    T_eval = min(pred_single.shape[0], traj.shape[0])
    n_obj = traj.shape[1]
    t = T_eval - 1
    refs, faces = [], []
    for o in range(n_obj):
        p0 = root / 'frame_00000' / f'obj_{o}_pos.npy'
        V0 = np.load(p0)
        if ev._object_scale is not None:
            V0 = ev._object_scale.apply(p0, V0)
        q = traj[0, o, 3:7]
        R0 = roma.unitquat_to_rotmat(torch.tensor([q[1], q[2], q[3], q[0]]).float()[None])[0].numpy()
        refs.append((V0 - traj[0, o, :3]) @ R0)
        faces.append(np.load(root / 'meshes' / f'obj_{o}_faces.npy'))
    origins, _ = E._hopnet_object_poses(pred_single[t:t + 1, :n_obj], refs, faces, n_obj)
    out = []
    for o in range(n_obj):
        # only the object's own vertices: configs with a fixed max_vertices_per_object pad every object
        V = pred_single[t, o, :refs[o].shape[0]]
        out.append({'scene': dataset_name, 'obj': o, 'frame': t, 'gt_z': float(traj[t, o, 2]),
                    'pred_z': float(origins[0, o, 2]), 'pred_min_z': float(V[:, 2].min()),
                    'xy_err': float(np.linalg.norm(origins[0, o, :2] - traj[t, o, :2]))})
    return out


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    own, rest = ap.parse_known_args()
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    os.makedirs(own.out, exist_ok=True)
    cfg = yaml.safe_load(open(own.config))
    cfg['data']['eval_step_sizes'] = [1]
    cfg.setdefault('inference', {}).update({'mode': True, 'generate_videos': False, 'compute_metrics': True})
    cfg['training']['enable_validation'] = False
    cfg['experiment_tag'] = f"rest_height_{os.getpid()}"
    tmp = tempfile.NamedTemporaryFile('w', suffix='.yaml', dir=os.path.join(REPO, 'configs'), prefix='_tmp_rest_', delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()

    import src.training.evaluator as E  # noqa: E402
    import src.train as T  # noqa: E402
    records = []
    orig = E.Evaluator._compute_rmse_for_sample

    def wrapped(self, pred_single, dataset_name, eval_dataset, skip_step_size, physical_checkpoints, **kw):
        records.extend(measure(self, pred_single, dataset_name, skip_step_size))
        return orig(self, pred_single, dataset_name, eval_dataset, skip_step_size, physical_checkpoints, **kw)
    E.Evaluator._compute_rmse_for_sample = wrapped

    def report():
        os.unlink(tmp.name)
        if not records:
            return
        lines = [f"{len(records)} objects at the last evaluated frame (step size 1)"]
        for name, lo, hi in (('small (resting centre ~0.35 m)', 0.33, 0.37), ('large (resting centre ~0.70 m)', 0.68, 0.72)):
            sel = [r for r in records if lo <= r['gt_z'] <= hi]
            if not sel:
                continue
            dz = np.array([r['pred_z'] - r['gt_z'] for r in sel])
            lines.append(f"{name}: {len(sel)} resting objects; predicted minus ground-truth centre height: mean {dz.mean():+.4f} m, "
                         f"median {np.median(dz):+.4f} m, |dz| > 0.05 m for {int((np.abs(dz) > 0.05).sum())}; "
                         f"predicted lowest vertex below z=0 for {sum(r['pred_min_z'] < 0 for r in sel)}; "
                         f"horizontal error median {np.median([r['xy_err'] for r in sel]):.3f} m")
        text = '\n'.join(lines)
        print(text, flush=True)
        open(os.path.join(own.out, 'report.txt'), 'w').write(text + '\n')
        json.dump(records, open(os.path.join(own.out, 'records.json'), 'w'))
    atexit.register(report)
    sys.argv = [os.path.join(REPO, 'src', 'train.py'), '--config', tmp.name, '--inference'] + rest
    T.main()


if __name__ == '__main__':
    main()
