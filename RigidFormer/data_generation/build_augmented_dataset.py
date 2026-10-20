"""A training set of MOVi-B plus augmented scenes (data_generation/augment_movi_scenes.py), MOVi-B itself unchanged.

  1. converts every augmented scene <raw>/<scene>/metadata.json into the packed training layout under <out>/<scene>
     (movi_to_sim_data.py), and checks that its frame-0 vertices are the unscaled collision meshes under the frame-0
     poses;
  2. links every MOVi-B scene into <out>, so <out> is one dataset folder;
  3. writes <out>/object_scales.json: MOVi-B's sizes plus those of the augmented scenes (with their fingerprints and
     frame-0 radii, as data_generation/build_object_scales.py), for data.object_scale_file;
  4. writes <out>/dataset_split.json: train = MOVi-B's train scenes + the augmented scenes; val and test = MOVi-B's,
     unchanged. Augmented scenes never enter val or test, and their names never collide with MOVi-B's.

    python data_generation/build_augmented_dataset.py --raw raw/MOVi-B-aug --movi datasets/MOVi-B \\
        --movi_scales data_generation/object_scales_MoviB.json --movi_split dataset_split_MoviB_1200.json \\
        --assets <dataset repository>/assets/kubasic --out datasets/MOVi-B-aug [--kinds movi:1200,toss:600,slide:600]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from movi_to_sim_data import prepare_from_metadata, write_conversion


def quat_rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def convert(job):
    """Convert one augmented scene; return its (name, sizes, fingerprint, radius0)."""
    raw_dir, out_dir, assets = (Path(p) for p in job)
    meta = json.loads((raw_dir / 'metadata.json').read_text())
    if meta['flags'].get('generator') != 'augment_movi_scenes':
        raise ValueError(f'{raw_dir}: not written by augment_movi_scenes.py')
    conv = prepare_from_metadata(meta, raw_dir.name, assets, source=f'augment_movi_scenes {raw_dir.name}',
                                 metadata_sha256=None)
    if not conv.manifest['complete']:
        raise ValueError(f'{raw_dir.name}: incomplete conversion')
    if not out_dir.exists():
        write_conversion(out_dir, conv, verbose=False, packed=True)
    traj = np.load(out_dir / 'vis_traj.npy')[0]
    frame0 = np.load(out_dir / 'frames.npy', mmap_mode='r')[0].astype(np.float64)
    counts = np.load(out_dir / 'frames_counts.npy')
    V0 = np.split(frame0, np.cumsum(counts)[:-1])
    radius0 = []
    for i, (inst, V) in enumerate(zip(meta['instances'], V0)):
        p, q = traj[0, i, :3].astype(np.float64), traj[0, i, 3:7].astype(np.float64)
        expected = np.asarray(conv.series[i].mesh.vertices, np.float64) @ quat_rotmat(q).T + p
        if np.abs(expected - V).max() > 1e-3:
            raise ValueError(f'{raw_dir.name} object {i}: frame-0 vertices are not the unscaled mesh at the frame-0 pose')
        radius0.append(float(np.linalg.norm(V - p, axis=1).max()))
    sizes = [float(inst['size']) for inst in meta['instances']]
    return raw_dir.name, sizes, traj[0, 0, :3].astype(np.float64).tolist(), radius0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='augment_movi_scenes.py output with <scene>/metadata.json')
    ap.add_argument('--movi', required=True, help='MOVi-B training layout (movi_<n>/)')
    ap.add_argument('--movi_scales', required=True, help="MOVi-B's object scale file")
    ap.add_argument('--movi_split', required=True, help="MOVi-B's split file")
    ap.add_argument('--assets', required=True, help='mesh library with <shape>/collision_geometry.obj')
    ap.add_argument('--out', required=True)
    ap.add_argument('--kinds', default='', help='kind:count,... to use (default: every augmented scene)')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    raw, movi, out = Path(args.raw), Path(args.movi).resolve(), Path(args.out)
    scenes = sorted(p.name for p in raw.iterdir() if (p / 'metadata.json').exists())
    if args.kinds:
        want = {f'aug_{k}_{i}' for spec in args.kinds.split(',') for k, n in [spec.split(':')] for i in range(int(n))}
        missing = sorted(want - set(scenes))
        if missing:
            raise SystemExit(f'{len(missing)} requested scenes not generated, e.g. {missing[:3]}')
        scenes = sorted(want)
    split = json.loads(Path(args.movi_split).read_text())
    movi_scenes = set(split['train_datasets']) | set(split['val_datasets']) | set(split['test_datasets'])
    clash = sorted(set(scenes) & movi_scenes)
    if clash or any(not s.startswith('aug_') for s in scenes):
        raise SystemExit(f'augmented scene names collide with MOVi-B or are malformed: {clash[:3]}')
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(convert, [(raw / s, out / s, args.assets) for s in scenes], chunksize=8))
    for s in sorted(movi_scenes):
        link = out / s
        if not link.exists():
            os.symlink(movi / s, link)
    ms = json.loads(Path(args.movi_scales).read_text())
    scales = {'source': f"{ms.get('source', 'MOVi-B')} + augment_movi_scenes", 'converted': out.name,
              'note': ms.get('note', ''), 'scales': dict(ms['scales']), 'fingerprint': dict(ms.get('fingerprint') or {}),
              'radius0': dict(ms['radius0'])}
    for name, sizes, fp, r0 in rows:
        scales['scales'][name], scales['fingerprint'][name], scales['radius0'][name] = sizes, fp, r0
    (out / 'object_scales.json').write_text(json.dumps(scales))
    aug = [name for name, *_ in rows]
    new_split = {
        'metadata': dict(split.get('metadata', {}), n_train=len(split['train_datasets']) + len(aug),
                         source=f"{split.get('metadata', {}).get('source', 'MOVi-B')}; train + {len(aug)} augmented "
                                f"scenes (augment_movi_scenes.py), val and test unchanged"),
        'scene_set': split.get('scene_set'),
        'train_datasets': list(split['train_datasets']) + aug,
        'val_datasets': list(split['val_datasets']),
        'test_datasets': list(split['test_datasets']),
    }
    assert not set(aug) & (set(new_split['val_datasets']) | set(new_split['test_datasets']))
    (out / 'dataset_split.json').write_text(json.dumps(new_split, indent=1))
    kinds = {}
    for name in aug:
        kinds[name.split('_')[1]] = kinds.get(name.split('_')[1], 0) + 1
    print(f'{out}: {len(aug)} augmented scenes {kinds} + {len(movi_scenes)} MOVi-B scenes linked; train '
          f'{len(new_split["train_datasets"])}, val {len(new_split["val_datasets"])}, test {len(new_split["test_datasets"])}')


if __name__ == '__main__':
    main()
