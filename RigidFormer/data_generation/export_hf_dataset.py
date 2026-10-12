"""Export MOVi-Sphere / MOVi-A / MOVi-B as a Hugging Face dataset.

Scenes (configs movi_sphere, movi_a, movi_b; Parquet, one row per scene). Each row holds one scene of the complete raw
data (assemble_complete_raw.py: the never-visible objects the generator's metadata lacks recovered by
replay_movi_scene.py): the generator settings, the physics constants, per object its KuBasic shape, size, material,
mass, friction, restitution, colour, labels and whether it was recovered, the trajectories `positions`, `quaternions`
(WXYZ), `velocities` and `angular_velocities` as [frame][object][3|4] float32 (the Kubric values are float32 numbers,
so this is lossless), the camera, and the annotations of the rendered video: per frame and object the number of visible
pixels and the projected centre, per object the 2D boxes of the frames it is visible in. Objects are in the metadata
order, which is the obj_<i> order of the converted data. The splits are dataset_split_*_1200.json (960/120/120). The
rendered videos are copied to videos/<config>/<scene>.mp4.

Geometry. The 12 KuBasic collision meshes are copied to assets/kubasic/. Config `meshes` holds them as arrays (vertices,
faces, area-weighted vertex normals); config `point_clouds` holds, per object shape, the mesh vertices (the points
RigidFormer reads) and surface samples of 128 to 4096 points with their normals.

`materialize_hf_dataset.py` turns the scenes back into the training layout, identical to movi_to_sim_data.py's output.

    python data_generation/export_hf_dataset.py --raw_root raw --out RigidFormer-MOVi
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'data_generation'))
from check_scenes import exported_instances, replay_num_objects  # noqa: E402

# config name: complete raw scenes, split file, size file
CONFIGS = {
    'movi_sphere': ('MOVi-spheres-complete', 'dataset_split_MoviSphere_1200.json', 'object_scales_MoviSphere.json'),
    'movi_a': ('MOVi-A-complete', 'dataset_split_MoviA_1200.json', 'object_scales_MoviA.json'),
    'movi_b': ('MOVi-B-complete', 'dataset_split_MoviB_1200.json', 'object_scales_MoviB.json'),
}
SPLITS = (('train', 'train_datasets'), ('validation', 'val_datasets'), ('test', 'test_datasets'))
TRAJ = (('positions', 3), ('quaternions', 4), ('velocities', 3), ('angular_velocities', 3))
FRAME_ANNOT = (('image_positions', 2),)          # float32 [frame][object][d], like the trajectories
GENERATOR_KEYS = ('objects_set', 'camera', 'background', 'min_num_objects', 'max_num_objects')
OBJECT_SHAPES = ('cone', 'cube', 'cylinder', 'gear', 'sphere', 'sponge', 'spot', 'suzanne', 'teapot', 'torus',
                 'torus_knot')                   # the KuBasic objects; `dome` is the floor
SURFACE_POINTS = (128, 256, 512, 768, 1024, 2048, 4096)

CAMERA = pa.struct([
    ('focal_length', pa.float64()), ('sensor_width', pa.float64()), ('field_of_view', pa.float64()),
    ('K', pa.list_(pa.list_(pa.float64(), 3), 3)), ('R', pa.list_(pa.list_(pa.float64(), 4), 4)),
    ('positions', pa.list_(pa.list_(pa.float32(), 3))), ('quaternions', pa.list_(pa.list_(pa.float32(), 4))),
])
SCHEMA = pa.schema([
    ('scene', pa.string()), ('seed', pa.int32()), ('split', pa.string()),
    ('num_frames', pa.int32()), ('frame_rate', pa.int32()), ('step_rate', pa.int32()),
    ('gravity', pa.list_(pa.float64(), 3)), ('floor_friction', pa.float64()), ('floor_restitution', pa.float64()),
    ('generator', pa.struct([('objects_set', pa.string()), ('camera', pa.string()), ('background', pa.string()),
                             ('min_num_objects', pa.int32()), ('max_num_objects', pa.int32())])),
    ('num_objects', pa.int32()),
    ('shape', pa.list_(pa.string())), ('size', pa.list_(pa.float64())), ('material', pa.list_(pa.string())),
    ('mass', pa.list_(pa.float64())), ('friction', pa.list_(pa.float64())), ('restitution', pa.list_(pa.float64())),
    ('color', pa.list_(pa.list_(pa.float64(), 3))), ('recovered', pa.list_(pa.bool_())),
] + [(k, pa.list_(pa.list_(pa.list_(pa.float32(), d)))) for k, d in TRAJ] + [
    ('size_label', pa.list_(pa.string())), ('color_label', pa.list_(pa.string())),
    ('resolution', pa.list_(pa.int32(), 2)), ('background_color', pa.string()), ('camera', CAMERA),
    ('visibility', pa.list_(pa.list_(pa.int32()))),
] + [(k, pa.list_(pa.list_(pa.list_(pa.float32(), d)))) for k, d in FRAME_ANNOT] + [
    ('bbox_frames', pa.list_(pa.list_(pa.int32()))), ('bboxes', pa.list_(pa.list_(pa.list_(pa.float32(), 4)))),
    ('video', pa.string()),
])
MESH_SCHEMA = pa.schema([
    ('shape', pa.string()), ('num_vertices', pa.int32()), ('num_faces', pa.int32()),
    ('vertices', pa.list_(pa.list_(pa.float64(), 3))), ('faces', pa.list_(pa.list_(pa.int32(), 3))),
    ('vertex_normals', pa.list_(pa.list_(pa.float32(), 3))),
])
POINT_SCHEMA = pa.schema([
    ('shape', pa.string()), ('sampling', pa.string()), ('num_points', pa.int32()),
    ('points', pa.list_(pa.list_(pa.float32(), 3))), ('normals', pa.list_(pa.list_(pa.float32(), 3))),
])


def f32_array(values, shape, what):
    """float32 array of values that are float32 numbers (stored losslessly), with the expected shape."""
    a64 = np.asarray(values, np.float64)
    a32 = a64.astype(np.float32)
    if a64.shape != shape or not np.array_equal(a32.astype(np.float64), a64):
        raise ValueError(f'{what} is not a float32 {shape} array')
    return a32


def scene_record(meta, scene, split, sizes, cfg):
    insts = exported_instances(meta)
    n = len(insts)
    if n != len(meta['instances']):             # (no floor instance in MOVi: object k is instance k)
        raise ValueError(f'{scene}: a floor instance in the metadata')
    if n != replay_num_objects(meta):
        raise ValueError(f'{scene}: {n} objects, {replay_num_objects(meta)} simulated (not a complete scene)')
    if len(sizes) != n or not np.array_equal(np.asarray([i['size'] for i in insts], np.float64), np.asarray(sizes)):
        raise ValueError(f'{scene}: object sizes differ from the size file')
    added = set(meta.get('replay', {}).get('added_instances', []))
    md, cam = meta['metadata'], meta['camera']
    T = int(md['num_frames'])
    rec = {
        'scene': scene, 'seed': int(md['seed']), 'split': split,
        'num_frames': T, 'frame_rate': int(md['frame_rate']), 'step_rate': int(md['step_rate']),
        'gravity': [float(g) for g in md['gravity']],
        'floor_friction': float(meta['flags']['floor_friction']), 'floor_restitution': float(meta['flags']['floor_restitution']),
        'generator': {k: meta['flags'][k] for k in GENERATOR_KEYS},
        'num_objects': n,
        'shape': [str(i.get('shape') or i['asset_id']) for i in insts],
        'size': [float(i['size']) for i in insts], 'material': [str(i['material']) for i in insts],
        'mass': [float(i['mass']) for i in insts], 'friction': [float(i['friction']) for i in insts],
        'restitution': [float(i['restitution']) for i in insts],
        'color': [[float(c) for c in i['color']] for i in insts],
        'recovered': [k in added for k in range(n)],
        'size_label': [i.get('size_label') for i in insts], 'color_label': [i.get('color_label') for i in insts],
        'resolution': [int(r) for r in md['resolution']], 'background_color': str(md['background']),
        'camera': {'focal_length': float(cam['focal_length']), 'sensor_width': float(cam['sensor_width']),
                   'field_of_view': float(cam['field_of_view']),
                   'K': np.asarray(cam['K'], np.float64).tolist(), 'R': np.asarray(cam['R'], np.float64).tolist(),
                   'positions': f32_array(cam['positions'], (T, 3), f'{scene}: camera positions').tolist(),
                   'quaternions': f32_array(cam['quaternions'], (T, 4), f'{scene}: camera quaternions').tolist()},
        'video': f'videos/{cfg}/{scene}.mp4',
    }
    if np.asarray(cam['K']).shape != (3, 3) or np.asarray(cam['R']).shape != (4, 4):
        raise ValueError(f'{scene}: camera K or R has an unexpected shape')
    for label in ('size_label', 'color_label'):
        if not all(v is None or isinstance(v, str) for v in rec[label]):
            raise ValueError(f'{scene}: {label} is not a string')
    for key, d in TRAJ + FRAME_ANNOT:
        rec[key] = np.stack([f32_array(i[key], (T, d), f'{scene}: {key}') for i in insts], axis=1)   # (T, n, d)
    vis = np.stack([np.asarray(i['visibility']) for i in insts], axis=1)                          # (T, n)
    if vis.shape != (T, n) or vis.dtype.kind not in 'iu' or vis.min() < 0 or vis.max() >= 2 ** 31:
        raise ValueError(f'{scene}: visibility is not a ({T}, {n}) array of pixel counts')
    rec['visibility'] = vis.astype(np.int32)
    rec['bbox_frames'], rec['bboxes'] = [], []
    for k, i in enumerate(insts):
        frames = [int(f) for f in i['bbox_frames']]
        boxes = f32_array(i['bboxes'], (len(frames), 4), f'{scene}: bboxes of object {k}') if frames else np.zeros((0, 4), np.float32)
        if frames != sorted(set(frames)) or (frames and (frames[0] < 0 or frames[-1] >= T)) or \
                frames != np.nonzero(vis[:, k])[0].tolist():
            raise ValueError(f'{scene}: bbox_frames of object {k} are not the frames it is visible in')
        rec['bbox_frames'].append(frames)
        rec['bboxes'].append(boxes.tolist())
    return rec


def nested(arrs, inner):
    """[frame][object][d] list array from (T, n, d) arrays (inner: the flat value type; d=None for (T, n) arrays)."""
    flat = pa.array(np.concatenate([a.reshape(-1) for a in arrs]), inner)
    d = None if arrs[0].ndim == 2 else arrs[0].shape[2]
    values = flat if d is None else pa.FixedSizeListArray.from_arrays(flat, d)
    obj_off = np.concatenate([[0], np.cumsum([a.shape[1] for a in arrs for _ in range(a.shape[0])])])
    mid = pa.ListArray.from_arrays(pa.array(obj_off, pa.int32()), values)
    frame_off = np.concatenate([[0], np.cumsum([a.shape[0] for a in arrs])])
    return pa.ListArray.from_arrays(pa.array(frame_off, pa.int32()), mid)


def to_table(records):
    """Arrow table; the per-frame columns are built from flat buffers (nested Python lists would be slow)."""
    cols = {}
    for f in SCHEMA:
        if f.name in dict(TRAJ + FRAME_ANNOT):
            cols[f.name] = nested([r[f.name] for r in records], pa.float32()).cast(f.type)
        elif f.name == 'visibility':
            cols[f.name] = nested([r[f.name] for r in records], pa.int32()).cast(f.type)
        else:
            cols[f.name] = pa.array([r[f.name] for r in records], f.type)
    return pa.table(cols, schema=SCHEMA)


def load_mesh(shape):
    """The KuBasic collision mesh as the converter reads it (vertex order kept)."""
    mesh = trimesh.load((REPO / 'data_generation' / 'objects' / shape / 'collision_geometry.obj').as_posix(), process=False)
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def face_normals(V, F):
    """Unnormalized face normals: cross products, of length twice the face area."""
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def vertex_normals(V, F):
    """Unit vertex normals: the area-weighted mean of the normals of the faces using the vertex."""
    acc = np.zeros_like(V)
    fn = face_normals(V, F)
    for c in range(3):
        np.add.at(acc, F[:, c], fn)
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    if not (norm > 0).all():
        raise ValueError('a vertex without a face')
    return acc / norm


def surface_samples(V, F, n):
    """n points uniform in area on the mesh surface and their unit face normals; numpy.random.default_rng(n)."""
    rng = np.random.default_rng(n)
    fn = face_normals(V, F)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    face = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1.0 - s, s * (1.0 - r2), s * r2], axis=1)                    # barycentric weights
    pts = np.einsum('nk,nkd->nd', w, V[F[face]])
    nrm = fn[face] / np.linalg.norm(fn[face], axis=1, keepdims=True)
    return pts, nrm


def shape_tables():
    """`meshes` (every KuBasic collision mesh) and `point_clouds` (per object shape) tables."""
    meshes, points = [], []
    for shape in sorted(p.name for p in (REPO / 'data_generation' / 'objects').iterdir()
                        if (p / 'collision_geometry.obj').is_file()):
        V, F = load_mesh(shape)
        vn = vertex_normals(V, F)
        meshes.append({'shape': shape, 'num_vertices': len(V), 'num_faces': len(F), 'vertices': V.tolist(),
                       'faces': F.astype(np.int32).tolist(), 'vertex_normals': vn.astype(np.float32).tolist()})
        if shape not in OBJECT_SHAPES:
            continue
        points.append({'shape': shape, 'sampling': 'vertices', 'num_points': len(V),
                       'points': V.astype(np.float32).tolist(), 'normals': vn.astype(np.float32).tolist()})
        for n in SURFACE_POINTS:
            pts, nrm = surface_samples(V, F, n)
            points.append({'shape': shape, 'sampling': 'surface', 'num_points': n,
                           'points': pts.astype(np.float32).tolist(), 'normals': nrm.astype(np.float32).tolist()})
    if sorted(r['shape'] for r in points if r['sampling'] == 'vertices') != sorted(OBJECT_SHAPES):
        raise FileNotFoundError('a KuBasic object mesh is missing from data_generation/objects')
    return pa.Table.from_pylist(meshes, schema=MESH_SCHEMA), pa.Table.from_pylist(points, schema=POINT_SCHEMA)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', required=True, help='folder with MOVi-spheres-complete, MOVi-A-complete, MOVi-B-complete')
    ap.add_argument('--out', required=True, help='new staging folder of the dataset repository')
    ap.add_argument('--scenes_per_file', type=int, default=240)
    ap.add_argument('--row_group', type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} exists: export into a new folder')
    manifest = {'configs': {}, 'files': {}}

    def add(path, **info):
        manifest['files'][str(path.relative_to(out))] = {'sha256': sha256(path), **info}

    for cfg, (raw, split_file, scale_file) in CONFIGS.items():
        split = json.load(open(REPO / split_file))
        scales = json.load(open(REPO / 'data_generation' / scale_file))['scales']
        manifest['configs'][cfg] = {'split_file': split_file, 'size_file': scale_file, 'splits': {}}
        for name, key in SPLITS:
            scenes = split[key]
            files = []
            for s0 in range(0, len(scenes), args.scenes_per_file):
                chunk = scenes[s0:s0 + args.scenes_per_file]
                recs = [scene_record(json.load(open(Path(args.raw_root) / raw / sc.split('_')[1] / 'metadata.json')),
                                     sc, name, scales[sc], cfg) for sc in chunk]
                for sc, rec in zip(chunk, recs):
                    video = out / rec['video']
                    video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(args.raw_root) / raw / sc.split('_')[1] / 'rgba.mp4', video)
                    add(video)
                files.append((chunk, to_table(recs)))
            nf = len(files)
            for k, (chunk, table) in enumerate(files):
                path = out / 'data' / cfg / f'{name}-{k:05d}-of-{nf:05d}.parquet'
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, path, compression='zstd', row_group_size=args.row_group)
                add(path, scenes=len(chunk))
            manifest['configs'][cfg]['splits'][name] = {
                'scenes': len(scenes), 'objects': int(sum(t.column('num_objects').to_numpy().sum() for _, t in files)),
                'recovered_objects': int(sum(sum(sum(r) for r in t.column('recovered').to_pylist()) for _, t in files))}
            print(cfg, name, manifest['configs'][cfg]['splits'][name], flush=True)
    meshes, points = shape_tables()
    for cfg, table in (('meshes', meshes), ('point_clouds', points)):
        path = out / 'data' / cfg / f'{cfg}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression='zstd')
        add(path, rows=table.num_rows)
        manifest['configs'][cfg] = {'rows': table.num_rows}
    for shape in meshes.column('shape').to_pylist():
        dst = out / 'assets' / 'kubasic' / shape / 'collision_geometry.obj'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / 'data_generation' / 'objects' / shape / 'collision_geometry.obj', dst)
        add(dst)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    shutil.copyfile(REPO / 'data_generation' / 'hf_dataset_card.md', out / 'README.md')   # the dataset card
    print('wrote', out)


if __name__ == '__main__':
    main()
