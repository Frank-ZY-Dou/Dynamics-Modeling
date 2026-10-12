"""Check the Hugging Face dataset against the complete raw scenes and the meshes it was exported from.

For each scene configuration, the dataset copy must hold every scene of the split file exactly once, in its split, and
every row must equal what export_hf_dataset.py makes of the scene's raw metadata: the generator settings and physics
constants, per object the shape, size, material, mass, friction, restitution, colour, labels and recovered flag, the
positions, quaternions, velocities and angular velocities in every frame (float32, bit for bit), the camera, the
visibility, image positions and 2D boxes; the scene's video must be the raw rgba.mp4, byte for byte. The `meshes` and
`point_clouds` tables must equal a fresh computation from data_generation/objects, and assets/ must hold those meshes.

    python scripts/verify_hf_export.py --dataset <local copy of the dataset repository> --raw_root raw
"""
import argparse
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'data_generation'))
from export_hf_dataset import CONFIGS, FRAME_ANNOT, SPLITS, TRAJ, scene_record, shape_tables  # noqa: E402
from materialize_hf_dataset import rows  # noqa: E402


def same(key, ref, got):
    if key in dict(TRAJ + FRAME_ANNOT):
        return got.dtype == np.float32 and np.array_equal(ref, got)
    if key == 'visibility':
        return got.dtype == np.int32 and np.array_equal(ref, got)
    return ref == got


def check_scenes(ds, raw_root, cfg):
    raw, split_file, scale_file = CONFIGS[cfg]
    split = json.load(open(REPO / split_file))
    scales = json.load(open(REPO / 'data_generation' / scale_file))['scales']
    want = {sc: name for name, key in SPLITS for sc in split[key]}
    seen, problems = set(), []
    for path in sorted((ds / 'data' / cfg).glob('*.parquet')):
        for row in rows(path):
            sc = row['scene']
            if sc in seen or want.get(sc) != row['split']:
                problems.append(f'{sc}: duplicate or not in split {row["split"]}')
                continue
            seen.add(sc)
            scene_dir = raw_root / raw / sc.split('_')[1]
            ref = scene_record(json.load(open(scene_dir / 'metadata.json')), sc, row['split'], scales[sc], cfg)
            if set(ref) != set(row):
                problems.append(f'{sc}: fields {sorted(set(ref) ^ set(row))} differ')
            problems += [f'{sc}: {k} differs' for k, v in ref.items() if k in row and not same(k, v, row[k])]
            video = ds / row['video']
            if not video.is_file() or not filecmp.cmp(video, scene_dir / 'rgba.mp4', shallow=False):
                problems.append(f'{sc}: {row["video"]} is not the raw rgba.mp4')
    missing = sorted(set(want) - seen)
    print(f'{cfg}: {len(seen)} rows checked, {len(missing)} scenes missing, {len(problems)} problems'
          + ''.join(f'\n  {p}' for p in (problems + missing)[:10]), flush=True)
    return not problems and not missing


def check_shapes(ds):
    meshes, points = shape_tables()
    problems = []
    for cfg, table in (('meshes', meshes), ('point_clouds', points)):
        files = sorted((ds / 'data' / cfg).glob('*.parquet'))
        if len(files) != 1 or not pq.read_table(files[0]).equals(table):
            problems.append(f'{cfg}: differs from a fresh computation')
    for shape in meshes.column('shape').to_pylist():
        src = REPO / 'data_generation' / 'objects' / shape / 'collision_geometry.obj'
        dst = ds / 'assets' / 'kubasic' / shape / 'collision_geometry.obj'
        if not dst.is_file() or not filecmp.cmp(src, dst, shallow=False):
            problems.append(f'assets/kubasic/{shape}: not the KuBasic collision mesh')
    for r in points.to_pylist():
        n = np.asarray(r['normals'], np.float64)
        if len(r['points']) != r['num_points'] or not np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6):
            problems.append(f"point_clouds {r['shape']} {r['sampling']} {r['num_points']}: bad points or normals")
    print(f'meshes: {meshes.num_rows} shapes, point_clouds: {points.num_rows} rows, {len(problems)} problems'
          + ''.join(f'\n  {p}' for p in problems[:10]), flush=True)
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='local copy of the dataset repository')
    ap.add_argument('--raw_root', required=True, help='folder with MOVi-spheres-complete, MOVi-A-complete, MOVi-B-complete')
    ap.add_argument('--configs', default='movi_sphere,movi_a,movi_b')
    args = ap.parse_args()
    ds, ok = Path(args.dataset), True
    for cfg in [c.strip() for c in args.configs.split(',') if c.strip()]:
        ok &= check_scenes(ds, Path(args.raw_root), cfg)
    ok &= check_shapes(ds)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
