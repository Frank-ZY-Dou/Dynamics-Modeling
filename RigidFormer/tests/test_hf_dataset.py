"""The Hugging Face dataset (data_generation/export_hf_dataset.py) and its way back to the training layout
(materialize_hf_dataset.py), on real MOVi scenes (skipped when the raw data is absent):
- a row holds the metadata values exactly (float64 scalars, float32 trajectories, the recovered flags, the camera, the
  visibility, image positions and 2D boxes, the labels) and names its video;
- a row written to Parquet and read back converts to exactly what the converter makes of the raw metadata: vertices,
  faces, anchors, trajectory, physics parameters and manifest, for scenes with and without recovered objects;
- without the objects flagged `recovered`, a row converts to exactly the conversion of HopNet's metadata;
- the export refuses an incomplete scene and sizes that differ from the size file;
- the `meshes` and `point_clouds` tables: the collision meshes as the converter reads them, the vertex point cloud equal
  to the converted vertices at the identity pose, surface samples reproducible, with unit normals."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'data_generation'))
RAW = os.environ.get('MOVI_RAW', os.path.join(REPO, 'raw'))   # <root>/MOVi-{spheres,A,B}[-complete]/<n>/ (README)
ASSETS = Path(REPO) / 'data_generation' / 'objects'
HOPNET_RAW = {'movi_sphere': 'MOVi-spheres', 'movi_a': 'MOVi-A', 'movi_b': 'MOVi-B'}
# scenes with recovered objects: MOVi-Sphere 749 (1), MOVi-B 203 (2, a test scene); and complete ones
CASES = [('movi_sphere', 'movi_749'), ('movi_sphere', 'movi_4'), ('movi_a', 'movi_4'), ('movi_a', 'movi_1200'),
         ('movi_b', 'movi_203'), ('movi_b', 'movi_7')]

pq = pytest.importorskip('pyarrow.parquet')
from export_hf_dataset import (CONFIGS, FRAME_ANNOT, OBJECT_SHAPES, SURFACE_POINTS, TRAJ, load_mesh,  # noqa: E402
                               scene_record, shape_tables, to_table)
from materialize_hf_dataset import metadata, rows  # noqa: E402
from movi_to_sim_data import _prepare, prepare_from_metadata  # noqa: E402


def _raw_dir(cfg, scene, hopnet=False):
    d = Path(RAW) / (HOPNET_RAW[cfg] if hopnet else CONFIGS[cfg][0]) / scene.split('_')[1]
    if not (d / 'metadata.json').is_file():
        pytest.skip(f'raw scene {d} not available')
    return d


def _rows(cfg, scenes, tmp_path):
    scales = json.load(open(Path(REPO) / 'data_generation' / CONFIGS[cfg][2]))['scales']
    recs = [scene_record(json.load(open(_raw_dir(cfg, sc) / 'metadata.json')), sc, 'test', scales[sc], cfg)
            for sc in scenes]
    path = tmp_path / f'{cfg}.parquet'
    pq.write_table(to_table(recs), path, compression='zstd', row_group_size=16)
    return rows(path)


def _same_conversion(conv, ref):
    assert np.array_equal(conv.traj, ref.traj) and conv.traj.dtype == ref.traj.dtype
    assert json.dumps(conv.physics) == json.dumps(ref.physics)
    assert len(conv.faces) == len(ref.faces) and all(np.array_equal(a, b) for a, b in zip(conv.faces, ref.faces))
    assert conv.anchors.keys() == ref.anchors.keys()
    assert all(np.array_equal(conv.anchors[k], ref.anchors[k]) for k in ref.anchors)
    for t in sorted({0, 1, conv.T // 2, conv.T - 1}):
        for a, b in zip(conv.frame_vertices(t), ref.frame_vertices(t)):
            assert a.dtype == b.dtype and np.array_equal(a, b)
    m, r = dict(conv.manifest), dict(ref.manifest)
    for k in ('source', 'metadata_sha256'):
        m.pop(k)
        r.pop(k)
    assert m == r


@pytest.mark.parametrize('cfg,scene', CASES)
def test_row_holds_the_metadata(cfg, scene, tmp_path):
    rec = _rows(cfg, [scene], tmp_path)[0]
    meta = json.load(open(_raw_dir(cfg, scene) / 'metadata.json'))
    insts = meta['instances']
    assert rec['num_objects'] == len(insts) and rec['seed'] == meta['metadata']['seed']
    for k in ('size', 'mass', 'friction', 'restitution'):
        assert rec[k] == [float(i[k]) for i in insts]
    assert rec['color'] == [[float(c) for c in i['color']] for i in insts]
    added = meta.get('replay', {}).get('added_instances', [])
    assert rec['recovered'] == [k in added for k in range(len(insts))]
    assert (scene in ('movi_749', 'movi_203') and cfg != 'movi_a') == bool(added)
    for key, d in TRAJ + FRAME_ANNOT:
        assert rec[key].dtype == np.float32 and rec[key].shape == (rec['num_frames'], len(insts), d)
        for i, inst in enumerate(insts):
            assert np.array_equal(rec[key][:, i].astype(np.float64), np.asarray(inst[key], np.float64))
    assert rec['visibility'].dtype == np.int32
    assert np.array_equal(rec['visibility'], np.stack([np.asarray(i['visibility']) for i in insts], axis=1))
    for i, inst in enumerate(insts):
        assert rec['bbox_frames'][i] == list(inst['bbox_frames'])
        assert rec['bboxes'][i] == [[float(c) for c in b] for b in inst['bboxes']]
        assert rec['bbox_frames'][i] == np.nonzero(rec['visibility'][:, i])[0].tolist()
        if rec['recovered'][i]:                                 # a recovered object is never in view
            assert not rec['bbox_frames'][i] and not rec['visibility'][:, i].any()
    assert rec['size_label'] == [i.get('size_label') for i in insts]
    assert rec['color_label'] == [i.get('color_label') for i in insts]
    cam = meta['camera']
    assert rec['camera']['K'] == cam['K'] and rec['camera']['R'] == cam['R']
    assert rec['camera']['positions'] == cam['positions'] and rec['camera']['quaternions'] == cam['quaternions']
    assert rec['camera']['field_of_view'] == cam['field_of_view'] and rec['camera']['focal_length'] == cam['focal_length']
    assert rec['resolution'] == meta['metadata']['resolution'] and rec['background_color'] == meta['metadata']['background']
    assert rec['video'] == f'videos/{cfg}/{scene}.mp4'


@pytest.mark.parametrize('cfg,scene', CASES)
def test_roundtrip_equals_conversion(cfg, scene, tmp_path):
    rec = _rows(cfg, [scene], tmp_path)[0]
    conv = prepare_from_metadata(metadata(rec), scene, ASSETS, source='test', metadata_sha256=None)
    assert conv.manifest['complete']
    _same_conversion(conv, _prepare(_raw_dir(cfg, scene), ASSETS))


@pytest.mark.parametrize('cfg,scene', [('movi_sphere', 'movi_749'), ('movi_b', 'movi_203'), ('movi_b', 'movi_7')])
def test_without_recovered_objects_is_hopnet_metadata(cfg, scene, tmp_path):
    rec = _rows(cfg, [scene], tmp_path)[0]
    meta = metadata(rec)
    k = rec['recovered'].count(False)
    meta['instances'] = meta['instances'][:k]
    conv = prepare_from_metadata(meta, scene, ASSETS, source='test', metadata_sha256=None)
    assert conv.manifest['complete'] == (not any(rec['recovered']))
    _same_conversion(conv, _prepare(_raw_dir(cfg, scene, hopnet=True), ASSETS))


def test_export_refuses_incomplete_scene_and_wrong_sizes():
    meta = json.load(open(_raw_dir('movi_b', 'movi_203', hopnet=True) / 'metadata.json'))
    scales = json.load(open(Path(REPO) / 'data_generation' / CONFIGS['movi_b'][2]))['scales']
    with pytest.raises(ValueError, match='complete'):
        scene_record(meta, 'movi_203', 'test', scales['movi_203'][:len(meta['instances'])], 'movi_b')
    meta = json.load(open(_raw_dir('movi_b', 'movi_7') / 'metadata.json'))
    sizes = list(scales['movi_7'])
    sizes[0] = np.nextafter(sizes[0], 2.0)
    with pytest.raises(ValueError, match='size'):
        scene_record(meta, 'movi_7', 'test', sizes, 'movi_b')


def test_materialize_checks_the_manifest(tmp_path):
    """materialize_hf_dataset.py refuses an incomplete, altered or extended copy before writing anything."""
    import hashlib
    import shutil
    from materialize_hf_dataset import check_dataset
    ds = tmp_path / 'ds'
    (ds / 'data' / 'movi_b').mkdir(parents=True)
    scales = json.load(open(Path(REPO) / 'data_generation' / CONFIGS['movi_b'][2]))['scales']
    recs = [scene_record(json.load(open(_raw_dir('movi_b', sc) / 'metadata.json')), sc, 'test', scales[sc], 'movi_b')
            for sc in ('movi_4', 'movi_40')]
    pq.write_table(to_table(recs), ds / 'data' / 'movi_b' / 'test-00000-of-00001.parquet')
    shutil.copytree(ASSETS, ds / 'assets' / 'kubasic')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = {str(p.relative_to(ds)): {'sha256': sha(p)} for p in ds.rglob('*') if p.is_file()}
    man = {'configs': {'movi_b': {'splits': {'test': {'scenes': 2}}}}, 'files': files}
    (ds / 'manifest.json').write_text(json.dumps(man))
    check_dataset(ds, ['movi_b'])
    mesh = ds / 'assets' / 'kubasic' / 'cone' / 'collision_geometry.obj'
    keep = mesh.read_bytes()
    mesh.unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        check_dataset(ds, ['movi_b'])
    mesh.write_bytes(keep + b'\n')
    with pytest.raises(ValueError, match='SHA-256'):
        check_dataset(ds, ['movi_b'])
    mesh.write_bytes(keep)
    shutil.copy(ds / 'data' / 'movi_b' / 'test-00000-of-00001.parquet', ds / 'data' / 'movi_b' / 'extra.parquet')
    with pytest.raises(ValueError, match='unlisted'):
        check_dataset(ds, ['movi_b'])
    (ds / 'data' / 'movi_b' / 'extra.parquet').unlink()
    man['configs']['movi_b']['splits']['test']['scenes'] = 3
    (ds / 'manifest.json').write_text(json.dumps(man))
    with pytest.raises(ValueError, match='scenes per split'):
        check_dataset(ds, ['movi_b'])


def test_shape_tables():
    """`meshes`: every KuBasic collision mesh as the converter reads it; `point_clouds`: per object shape the vertex point
    cloud (the converted vertices at the identity pose) and reproducible surface samples with unit normals."""
    from movi_to_sim_data import _apply_pose_vertices
    import trimesh
    meshes, points = shape_tables()
    again = shape_tables()
    assert meshes.equals(again[0]) and points.equals(again[1])
    assert sorted(meshes.column('shape').to_pylist()) == sorted(OBJECT_SHAPES + ('dome',))
    for r in meshes.to_pylist():
        V, F = load_mesh(r['shape'])
        assert np.array_equal(np.asarray(r['vertices']), V) and np.array_equal(np.asarray(r['faces']), F)
        assert (r['num_vertices'], r['num_faces']) == V.shape[:1] + F.shape[:1]
    rows_ = points.to_pylist()
    assert len(rows_) == len(OBJECT_SHAPES) * (1 + len(SURFACE_POINTS))
    for r in rows_:
        pts, nrm = np.asarray(r['points']), np.asarray(r['normals'])
        assert pts.shape == nrm.shape == (r['num_points'], 3)
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)
        if r['sampling'] == 'vertices':
            mesh = trimesh.load((ASSETS / r['shape'] / 'collision_geometry.obj').as_posix(), process=False)
            ref = _apply_pose_vertices(mesh, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])).astype(np.float32)
            assert np.array_equal(pts.astype(np.float32), ref)
        else:
            assert r['num_points'] in SURFACE_POINTS
            V, F = load_mesh(r['shape'])
            lo, hi = V.min(0) - 1e-6, V.max(0) + 1e-6
            assert ((pts >= lo) & (pts <= hi)).all()
