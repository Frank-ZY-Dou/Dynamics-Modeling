"""Write the training layout from the Hugging Face dataset made by export_hf_dataset.py.

Every row (one scene) is converted with movi_to_sim_data.py's own code (prepare_from_metadata, write_conversion), so a
scene is identical to the conversion of its raw metadata: per-frame vertices of the objects' collision meshes, faces,
physics parameters, the pose trajectory and the FPS anchor caches. Before writing, the local copy is checked against its
manifest.json (every listed file present with its SHA-256, no other Parquet file, the scene counts of every split, unique
scenes). The output folders are the ones the configs read and the size files name:

    <out>/MOVi-Sphere   movi_sphere   configs/movi_sphere.yaml
    <out>/MOVi-A        movi_a        configs/movi_a.yaml
    <out>/MOVi-B        movi_b        configs/movi_b.yaml

    python data_generation/materialize_hf_dataset.py --dataset <local copy of the dataset repository> \\
        --out datasets --workers 16
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from movi_to_sim_data import prepare_from_metadata, write_conversion  # noqa: E402

FOLDERS = {'movi_sphere': 'MOVi-Sphere', 'movi_a': 'MOVi-A', 'movi_b': 'MOVi-B'}
TRAJ = (('positions', 3), ('quaternions', 4), ('velocities', 3), ('angular_velocities', 3))
FRAME_ANNOT = (('image_positions', 2),)
# the columns the conversion reads (not the camera, the image annotations or the video)
CONVERT_COLUMNS = ['scene', 'seed', 'num_frames', 'frame_rate', 'step_rate', 'gravity', 'generator', 'num_objects',
                   'shape', 'size', 'material', 'mass', 'friction', 'restitution', 'recovered'] + [k for k, _ in TRAJ]
SHAPES =('cone', 'cube', 'cylinder', 'gear', 'sphere', 'sponge', 'spot', 'suzanne', 'teapot', 'torus', 'torus_knot')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(ds, cfgs):
    """The local copy must be the published dataset: the files manifest.json lists for these configurations and the
    meshes (the videos and the other configurations are not needed), with their SHA-256, no other Parquet file, and in
    every split the listed number of scenes, each once, every row in the file of its split."""
    if not (ds / 'manifest.json').is_file():
        raise FileNotFoundError(f'{ds / "manifest.json"} missing: not a complete copy of the dataset repository')
    man = json.loads((ds / 'manifest.json').read_text())
    needed = {rel: info for rel, info in man['files'].items()
              if rel.startswith('assets/') or any(rel.startswith(f'data/{cfg}/') for cfg in cfgs)}
    for rel, info in needed.items():
        path = ds / rel
        if not path.is_file():
            raise FileNotFoundError(f'{path} (listed in manifest.json) is missing: incomplete download')
        if sha256(path) != info['sha256']:
            raise ValueError(f'{path}: SHA-256 differs from manifest.json (damaged or altered)')
    for cfg in cfgs:
        listed = {rel for rel in man['files'] if rel.startswith(f'data/{cfg}/')}
        present = {str(p.relative_to(ds)) for p in (ds / 'data' / cfg).glob('*.parquet')}
        if listed != present:
            raise ValueError(f'{cfg}: Parquet files differ from manifest.json: missing {sorted(listed - present)}, '
                             f'unlisted {sorted(present - listed)}')
        scenes, counts = set(), Counter()
        for rel in sorted(listed):
            t = pq.read_table(ds / rel, columns=['scene', 'split'])
            for scene, split in zip(t.column('scene').to_pylist(), t.column('split').to_pylist()):
                if not Path(rel).name.startswith(f'{split}-'):
                    raise ValueError(f'{rel}: scene {scene} of split {split} in another split\'s file')
                if scene in scenes:
                    raise ValueError(f'{cfg}: scene {scene} appears twice')
                scenes.add(scene)
                counts[split] += 1
        want = {split: v['scenes'] for split, v in man['configs'][cfg]['splits'].items()}
        if dict(counts) != want:
            raise ValueError(f'{cfg}: scenes per split {dict(counts)}, manifest.json lists {want}')


def rows(path, columns=None):
    """The rows of one Parquet file (all columns, or `columns`), the per-frame columns as arrays: trajectories and image
    positions (frames, objects, d) float32, visibility (frames, objects) int32."""
    table = pq.read_table(path, columns=columns)
    frame_cols = [(k, d) for k, d in TRAJ + FRAME_ANNOT + (('visibility', None),) if k in table.column_names]
    plain = table.drop_columns([k for k, _ in frame_cols]).to_pylist()
    for key, d in frame_cols:
        col = table.column(key).combine_chunks()
        frame_off = col.offsets.to_numpy()
        frames = col.flatten()
        obj_off = frames.offsets.to_numpy()
        values = frames.flatten()
        values = (values if d is None else values.flatten()).to_numpy(zero_copy_only=False)
        values = values if d is None else values.reshape(-1, d)
        for r, rec in enumerate(plain):
            f0, f1 = frame_off[r], frame_off[r + 1]
            shape = (f1 - f0, rec['num_objects']) + (() if d is None else (d,))
            rec[key] = values[obj_off[f0]:obj_off[f1]].reshape(shape)
    return plain


def metadata(rec):
    """The Kubric metadata fields the converter reads, from one row."""
    T, n = rec['num_frames'], rec['num_objects']
    for key, d in TRAJ:
        if rec[key].shape != (T, n, d):
            raise ValueError(f"{rec['scene']}: {key} has shape {rec[key].shape}")
    k = rec['recovered'].count(False)
    if any(rec['recovered'][:k]):
        raise ValueError(f"{rec['scene']}: a recovered object before a recorded one")
    return {
        'metadata': {'num_frames': T, 'frame_rate': rec['frame_rate'], 'step_rate': rec['step_rate'], 'seed': rec['seed'],
                     'gravity': rec['gravity']},
        'flags': dict(rec['generator'], seed=rec['seed']),
        'instances': [{'asset_id': rec['shape'][i], 'shape': rec['shape'][i], 'size': rec['size'][i],
                       'material': rec['material'][i], 'mass': rec['mass'][i], 'friction': rec['friction'][i],
                       'restitution': rec['restitution'][i],
                       **{key: rec[key][:, i] for key, _ in TRAJ}} for i in range(n)],
    }


def materialize(job):
    rec, out_root, assets, source = job
    conv = prepare_from_metadata(metadata(rec), rec['scene'], Path(assets), source=f"{source} {rec['scene']}",
                                 metadata_sha256=None)
    if not conv.manifest['complete']:
        raise ValueError(f"{rec['scene']}: not a complete scene")
    write_conversion(Path(out_root) / rec['scene'], conv, verbose=False)
    return rec['scene']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='local copy of the dataset repository (data/, assets/)')
    ap.add_argument('--out', required=True, help='folder for the converted datasets (the datasets/ folder of the code)')
    ap.add_argument('--configs', default='movi_sphere,movi_a,movi_b')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    ds = Path(args.dataset)
    assets = ds / 'assets' / 'kubasic'
    missing = [d for d in SHAPES if not (assets / d / 'collision_geometry.obj').is_file()]
    if missing:
        raise FileNotFoundError(f'{assets}: no collision mesh for {missing}')
    cfgs = [c.strip() for c in args.configs.split(',') if c.strip()]
    unknown = [c for c in cfgs if c not in FOLDERS]
    if unknown:
        raise ValueError(f'unknown configs {unknown}; choose from {sorted(FOLDERS)}')
    for cfg in cfgs:
        if (Path(args.out) / FOLDERS[cfg]).exists():
            raise FileExistsError(f'{Path(args.out) / FOLDERS[cfg]} exists: materialize into a new folder')
    check_dataset(ds, cfgs)
    for cfg in cfgs:
        out_root = Path(args.out) / FOLDERS[cfg]
        if out_root.exists():
            raise FileExistsError(f'{out_root} exists: materialize into a new folder')
        files = sorted((ds / 'data' / cfg).glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'no Parquet files in {ds / "data" / cfg}')
        out_root.mkdir(parents=True)
        n = 0
        with ProcessPoolExecutor(args.workers) as pool:
            for path in files:
                jobs = [(rec, str(out_root), str(assets), f'{ds.name}/{path.relative_to(ds)}')
                        for rec in rows(path, CONVERT_COLUMNS)]
                for _ in pool.map(materialize, jobs):
                    n += 1
        print(f'{cfg}: {n} scenes -> {out_root}', flush=True)


if __name__ == '__main__':
    main()
