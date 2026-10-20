"""Data and evaluation integrity checks on the real MOVi data (skipped when the data is absent):
- every split holds HopNet's benchmark lists with every scene complete; the recovered objects of the complete raw scenes
  leave the recorded ones unchanged, and the size files match every converted scene; the configured scene set must
  match the split file;
- HopNet's shape-matching estimator returns the ground-truth origin for a perfect prediction (the area-weighted
  surface centroid does not: a MOVi-B floor of ~1.8 cm);
- the loader refuses missing frames, per-frame vertex counts that differ from frame 0, and noise augmentation it would
  silently ignore; the size file refuses vertex data that already carries the size;
- the converter reproduces the stored data bit for bit, marks incomplete scenes, and reuses an output only when its
  manifest matches the source metadata."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# <root>/MOVi-{spheres,A,B}/<n>/ (HopNet's raw scenes) and <root>/MOVi-{spheres,A,B}-complete/<n>/ (data_generation/README.md)
RAW = os.environ.get('MOVI_RAW', os.path.join(REPO, 'raw'))
SETS = {'S': ('MOVi-Sphere', 'MoviSphere', 'MOVi-spheres'), 'A': ('MOVi-A', 'MoviA', 'MOVi-A'),
        'B': ('MOVi-B', 'MoviB', 'MOVi-B')}

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, 'datasets', 'MOVi-B')), reason='MOVi data not available')


def _split(name):
    return json.load(open(os.path.join(REPO, f'dataset_split_{name}_1200.json')))


@pytest.mark.parametrize('key', ['S', 'A', 'B'])
def test_splits_and_complete_data(key):
    """The splits are disjoint 960/120/120 lists of complete scenes; HopNet's metadata lacks objects in exactly 1
    MOVi-Sphere and 48 MOVi-B scenes, whose complete raw scenes keep the recorded objects unchanged and add the replayed
    ones; the converted data of those scenes is complete, and the size file matches every converted scene."""
    conv, name, raw = SETS[key]
    sp = _split(name)
    assert sp['scene_set'] == 'complete'
    lists = [sp[k] for k in ('train_datasets', 'val_datasets', 'test_datasets')]
    assert [len(x) for x in lists] == [960, 120, 120] and len(set().union(*map(set, lists))) == 1200
    scenes = sum(lists, [])
    data = os.path.join(REPO, 'datasets', conv)
    from data_generation.check_scenes import exported_instances, replay_num_objects
    if os.path.isdir(os.path.join(RAW, raw)) and os.path.isdir(os.path.join(RAW, raw + '-complete')):
        lacking = []
        for sc in scenes:
            n = sc.split('_')[1]
            orig = json.load(open(os.path.join(RAW, raw, n, 'metadata.json')))
            full = json.load(open(os.path.join(RAW, raw + '-complete', n, 'metadata.json')))
            assert len(exported_instances(full)) == replay_num_objects(full), sc
            assert full['instances'][:len(orig['instances'])] == orig['instances'], sc
            if len(exported_instances(orig)) < replay_num_objects(orig):
                lacking.append(sc)
        assert len(lacking) == {'S': 1, 'A': 0, 'B': 48}[key]
        assert {'S': {'movi_749'}, 'A': set(), 'B': {'movi_522', 'movi_1103', 'movi_203', 'movi_301', 'movi_924'}}[key] <= set(lacking)
        for sc in lacking:
            man = json.load(open(os.path.join(data, sc, 'conversion.json')))
            assert man['complete'] and man['exported_objects'] == man['simulated_objects'], sc
    from src.data.object_scale import ObjectScale, check_object_counts
    osc = ObjectScale(os.path.join(REPO, 'data_generation', f'object_scales_{name}.json'))
    assert osc.converted == os.path.basename(os.path.realpath(data))     # the loader's dataset-identity check
    check_object_counts(osc, data, scenes)


def test_scene_set_must_match_split_file():
    from src.data.multi_dataloader import check_scene_set
    cfg = {'data': {'split_file': os.path.join(REPO, 'dataset_split_MoviB_1200.json'), 'scene_set': 'complete'}}
    assert check_scene_set(cfg) == 'complete'
    for wrong in ('benchmark', None):
        if wrong is None:
            del cfg['data']['scene_set']                       # default: benchmark
        else:
            cfg['data']['scene_set'] = wrong
        with pytest.raises(ValueError):
            check_scene_set(cfg)


def _perfect_prediction(data, scene, scale, steps):
    """Ground-truth (scaled) vertices at the given frames as a prediction array (T, Nobj, Nv, 3)."""
    d = os.path.join(data, scene)
    n = len([f for f in os.listdir(os.path.join(d, 'frame_00000')) if f.endswith('_pos.npy')])
    load = lambda t: [scale.apply(Path(d) / f'frame_{t:05d}' / f'obj_{o}_pos.npy',
                                  np.load(os.path.join(d, f'frame_{t:05d}', f'obj_{o}_pos.npy'))) for o in range(n)]
    f0 = load(0)
    nv = max(v.shape[0] for v in f0)
    pred = np.zeros((max(steps) + 1, n, nv, 3), dtype=np.float32)
    for o in range(n):
        pred[:, o, :f0[o].shape[0]] = f0[o]
    for s in steps:
        for o, V in enumerate(load(s)):
            pred[s, o, :V.shape[0]] = V
    return pred


def test_hopnet_estimator_is_exact():
    import src.training.evaluator as E
    from src.data.object_scale import ObjectScale
    data = os.path.join(REPO, 'datasets', 'MOVi-B')
    cfg = {'data': {'dataset_path': data, 'object_scale_file': 'data_generation/object_scales_MoviB.json'}}
    scale = ObjectScale(os.path.join(REPO, 'data_generation', 'object_scales_MoviB.json'))
    cps = [25, 100, 400]
    for scene in _split('MoviB')['test_datasets'][:2]:
        pred = _perfect_prediction(data, scene, scale, cps)
        res = {}
        for est in ('hopnet', 'area_centroid'):
            ev = E.Evaluator.__new__(E.Evaluator)
            ev.config, ev.device, ev.position_estimator, ev.eval_warmup_frames, ev._object_scale = cfg, 'cpu', est, 4, scale
            res[est] = ev._compute_rmse_for_sample(pred, scene, None, 1, cps)
        for cp in cps:
            assert res['hopnet'][cp] < 1e-5, (scene, cp, res['hopnet'][cp])
            assert res['hopnet'][f'rotation_deg_{cp}'] < 0.05
            assert res['area_centroid'][cp] > 1e-3                # the surface-centroid floor of MOVi-B objects


def _scene_copy(tmp_path, scene='movi_7', drop_frame=None, truncate=None):
    """A copy of one MOVi-B scene in tmp_path (frames as symlinks; anchors, meshes, physics copied so nothing is
    written into the dataset) and a split file listing it as the only test scene."""
    src = Path(REPO) / 'datasets' / 'MOVi-B' / scene
    root = tmp_path / 'data'
    dst = root / scene
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.name.startswith('frame_'):
            if drop_frame is not None and p.name == f'frame_{drop_frame:05d}':
                continue
            if truncate is not None and p.name == f'frame_{truncate:05d}':
                (dst / p.name).mkdir()
                for f in p.iterdir():
                    V = np.load(f)
                    np.save(dst / p.name / f.name, V[:-1] if f.name == 'obj_0_pos.npy' else V)
                continue
            os.symlink(p, dst / p.name)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name)
        else:
            shutil.copy(p, dst / p.name)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train_datasets': [], 'val_datasets': [], 'test_datasets': [scene]}))
    return str(root), str(split)


def _dataset(root, split, **kw):
    from src.data.multi_dataloader import MultiObjectDynamics
    args = dict(data_path=root, sequence_length=8, stride=1, skip_step_size=1, split='test', split_file=split,
                max_vertices_per_object=None, max_objects_per_frame=16, anchor_k=4, random_step_size={'enable': False},
                augmentation={'enable': False}, anchor_fps_start='centroid_farthest', fps_order=True)
    args.update(kw)
    return MultiObjectDynamics(**args)


def test_loader_refuses_missing_frame(tmp_path):
    root, split = _scene_copy(tmp_path, drop_frame=3)
    with pytest.raises(ValueError, match='without gaps'):
        _dataset(root, split)


def test_loader_refuses_vertex_count_change(tmp_path):
    root, split = _scene_copy(tmp_path, truncate=2)
    ds = _dataset(root, split)
    ds[5]                                                       # frames 5..12: unaffected
    with pytest.raises(ValueError, match='vertex counts'):
        ds[0]                                                   # frames 0..7 include the truncated frame 2


def test_loader_refuses_ignored_noise(tmp_path):
    root, split = _scene_copy(tmp_path)
    for blk in ('random_walk', 'independent'):
        with pytest.raises(NotImplementedError):
            _dataset(root, split, augmentation={blk: {'enable': True}})


def test_sidecar_refuses_scaled_vertex_data(tmp_path):
    from src.data.object_scale import ObjectScale, check_object_counts
    src = os.path.join(REPO, 'data_generation', 'object_scales_MoviB.json')
    osc = ObjectScale(src)
    scene = _split('MoviB')['test_datasets'][0]
    data = os.path.join(REPO, 'datasets', 'MOVi-B')
    check_object_counts(osc, data, [scene])
    osc.radius0[scene] = osc.radius0[scene] * osc.scales[scene]      # as if the stored vertices carried the size
    with pytest.raises(ValueError, match='radius0'):
        check_object_counts(osc, data, [scene])
    d = json.load(open(src))
    del d['radius0']
    (tmp_path / 'old.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='radius0'):
        ObjectScale(str(tmp_path / 'old.json'))


@pytest.mark.skipif(not os.path.isdir(os.path.join(RAW, 'MOVi-B')), reason='raw MOVi-B metadata not available')
def test_converter_reproduces_and_marks_incomplete(tmp_path):
    from data_generation.movi_to_sim_data import process_sample
    assets = Path(REPO) / 'data_generation' / 'objects'
    out = tmp_path / 'out'
    out.mkdir()
    for n, complete in (('7', True), ('522', False)):
        process_sample(Path(RAW) / 'MOVi-B' / n, out, assets)
        new, old = out / f'movi_{n}', Path(REPO) / 'datasets' / 'MOVi-B' / f'movi_{n}'
        man = json.loads((new / 'conversion.json').read_text())
        assert man['complete'] is complete and man['num_frames'] == 480
        for t in (0, 1, 239, 479):
            for f in (new / f'frame_{t:05d}').iterdir():
                assert np.array_equal(np.load(f), np.load(old / f'frame_{t:05d}' / f.name))
        traj = np.load(new / 'vis_traj.npy')
        assert np.array_equal(traj, np.load(old / 'vis_traj.npy')[:, :, :traj.shape[2]])   # recorded objects first
        for f in (new / 'anchors').iterdir():
            assert np.array_equal(np.load(f), np.load(old / 'anchors' / f.name))
    process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)          # matching manifest: reused
    man = json.loads((out / 'movi_7' / 'conversion.json').read_text())
    man['metadata_sha256'] = '0' * 64
    (out / 'movi_7' / 'conversion.json').write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match='manifest differs'):
        process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)
    (out / 'movi_7' / 'conversion.json').unlink()
    with pytest.raises(RuntimeError, match='without a conversion manifest'):
        process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)


def test_ground_handler_is_identity_on_ground_truth():
    """The configs of MOVi-Sphere, MOVi-A and MOVi-B, each on its own data with its own size file: the ground handler
    (lift below z = buffer) leaves the deepest simulated floor impacts of the data unchanged (-5.65 mm at most); with
    buffer 0 it would lift them."""
    import torch
    import yaml
    from src.data.object_scale import ObjectScale
    from src.utils.intersection_handler import handle_intersections
    deepest = {'S': [('movi_851', 5, 165), ('movi_250', 3, 207)], 'A': [('movi_376', 0, 186)], 'B': [('movi_541', 4, 225)]}
    configs = {'S': 'movi_sphere.yaml', 'A': 'movi_a.yaml', 'B': 'movi_b.yaml'}
    for key, cases in deepest.items():
        cfg = yaml.safe_load(open(os.path.join(REPO, 'configs', configs[key])))
        handler = cfg['model']['intersection_handling']
        scale = ObjectScale(os.path.join(REPO, cfg['data']['object_scale_file']))
        for scene, obj, t in cases:
            path = Path(REPO) / cfg['data']['dataset_path'] / scene / f'frame_{t:05d}' / f'obj_{obj}_pos.npy'
            V = torch.from_numpy(scale.apply(path, np.load(path)).astype(np.float32))[None, None]
            assert float(V[..., 2].min()) < -0.001                          # a real penetration of the floor plane
            ones_o, ones_v = torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, V.shape[2], dtype=torch.bool)
            assert torch.equal(handle_intersections(V, ones_o, ones_v, handler), V)
            lifted = handle_intersections(V, ones_o, ones_v, dict(handler, ground=dict(handler['ground'], buffer=0.0)))
            assert not torch.equal(lifted, V)


def test_sidecar_builder_refuses_scaled_vertex_data(tmp_path):
    """build_object_scales.py establishes the scale convention from the asset meshes: a converted scene whose stored
    vertices already carry the size is refused (a sidecar rebuilt from it would scale them a second time)."""
    from data_generation.build_object_scales import scene_sizes
    from src.data.object_scale import ObjectScale
    raw = os.path.join(RAW, 'MOVi-spheres')
    if not os.path.isdir(raw):
        pytest.skip('raw MOVi-Sphere metadata not available')
    scene = 'movi_4'
    src = Path(REPO) / 'datasets' / 'MOVi-Sphere' / scene
    scene_sizes(raw, str(src.parent), scene)                               # the stored (unscaled) data is accepted
    dst = tmp_path / 'conv' / scene
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.name != 'frame_00000':
            os.symlink(p, dst / p.name)
    (dst / 'frame_00000').mkdir()
    osc = ObjectScale(os.path.join(REPO, 'data_generation', 'object_scales_MoviSphere.json'))
    for f in (src / 'frame_00000').iterdir():
        np.save(dst / 'frame_00000' / f.name, osc.apply(f, np.load(f)))      # as if re-converted with the size
    with pytest.raises(RuntimeError, match='not the unscaled asset mesh'):
        scene_sizes(raw, str(dst.parent), scene)


@pytest.mark.skipif(not os.path.isdir(os.path.join(RAW, 'MOVi-B')), reason='raw MOVi-B metadata not available')
def test_converter_reuse_checks_outputs_and_assets(tmp_path):
    from data_generation.movi_to_sim_data import process_sample
    assets = Path(REPO) / 'data_generation' / 'objects'
    out = tmp_path / 'out'
    out.mkdir()
    process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)
    victim = out / 'movi_7' / 'frame_00100' / 'obj_2_pos.npy'
    V = np.load(victim)
    for bad in (None, V + np.float32(0.01), np.where(np.arange(V.shape[0])[:, None] == 0, np.nan, V).astype(V.dtype), V[:-1]):
        if bad is None:
            victim.unlink()                                              # missing file
        else:
            np.save(victim, bad)                                         # 10 mm shift, a NaN, one vertex less
        with pytest.raises(RuntimeError, match='frame 100'):
            process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)
    np.save(victim, V)
    process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)               # restored: reused
    phys = out / 'movi_7' / 'physics_params.json'
    good = phys.read_text()
    phys.write_text(json.dumps({k: {} for k in json.loads(good)}))
    with pytest.raises(RuntimeError, match='physics_params'):
        process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)
    phys.write_text(good)
    man_path = out / 'movi_7' / 'conversion.json'
    man = json.loads(man_path.read_text())
    man['objects'][0]['mesh_sha256'] = '0' * 64
    man_path.write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match='manifest differs'):
        process_sample(Path(RAW) / 'MOVi-B' / '7', out, assets)


def test_point_cloud_mode_refused(tmp_path):
    root, split = _scene_copy(tmp_path)
    with pytest.raises(NotImplementedError):
        _dataset(root, split, if_point_cloud={'enable': True})


def test_seed_replay_uses_the_recorded_seed():
    from data_generation.check_scenes import replay_num_objects
    flags = {'seed': 0, 'camera': 'random', 'min_num_objects': 3, 'max_num_objects': 10}
    n1 = replay_num_objects({'flags': flags, 'metadata': {'seed': 1}})
    assert n1 == replay_num_objects({'flags': dict(flags, seed=1), 'metadata': {'seed': 1}})
    with pytest.raises(ValueError):
        replay_num_objects({'flags': dict(flags, seed=2), 'metadata': {'seed': 1}})


def test_protocol_names_the_evaluated_partition(tmp_path):
    import src.training.evaluator as E
    sp = _split('MoviB')
    sp['test_datasets'], sp['evaluated_partition'] = list(sp['val_datasets']), 'val'
    (tmp_path / 'val.json').write_text(json.dumps(sp))
    ev = E.Evaluator.__new__(E.Evaluator)
    ev.position_estimator = 'hopnet'
    ev.warmup_from_step1, ev._partial_view = False, None          # the defaults of Evaluator.__init__
    ev.config = {'data': {'split_file': str(tmp_path / 'val.json'), 'scene_set': 'complete'}}
    assert '114 val scenes' in ev._protocol_text('test', 114)
    ev.config['data']['split_file'] = os.path.join(REPO, 'dataset_split_MoviB_1200.json')
    assert '120 test scenes' in ev._protocol_text('test', 120)


def test_rest_height_measure_is_exact_for_a_perfect_prediction():
    """scripts/rest_height_check.py recovers the object origin by shape matching (the vertex mean of the 64-vertex
    sphere is 22.5 mm off the origin)."""
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import rest_height_check as RH
    import src.training.evaluator as E
    from src.data.object_scale import ObjectScale
    data = os.path.join(REPO, 'datasets', 'MOVi-Sphere')
    scale = ObjectScale(os.path.join(REPO, 'data_generation', 'object_scales_MoviSphere.json'))
    ev = E.Evaluator.__new__(E.Evaluator)
    ev.config, ev._object_scale = {'data': {'dataset_path': data}}, scale
    scene = _split('MoviSphere')['test_datasets'][0]
    d = Path(data) / scene
    n = len([f for f in os.listdir(d / 'frame_00000') if f.endswith('_pos.npy')])
    Vs = [scale.apply(d / 'frame_00479' / f'obj_{o}_pos.npy', np.load(d / 'frame_00479' / f'obj_{o}_pos.npy')) for o in range(n)]
    pred = np.zeros((480, n, max(v.shape[0] for v in Vs) + 5, 3), dtype=np.float32)
    for o, V in enumerate(Vs):
        pred[479, o, :V.shape[0]] = V
    for r in RH.measure(ev, pred, scene, 1):
        assert abs(r['pred_z'] - r['gt_z']) < 1e-5 and r['xy_err'] < 1e-5


def test_non_finite_values_are_refused(tmp_path):
    """NaN must not pass the size-sidecar checks (a NaN comparison is false)."""
    from src.data.object_scale import ObjectScale, check_object_counts
    from data_generation.build_object_scales import scene_sizes
    src = os.path.join(REPO, 'data_generation', 'object_scales_MoviSphere.json')
    d = json.load(open(src))
    scene = 'movi_4'
    bad = json.loads(json.dumps(d))
    bad['radius0'][scene][0] = float('nan')
    (tmp_path / 'nan.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='non-finite'):
        ObjectScale(str(tmp_path / 'nan.json'))
    # a NaN vertex in the stored frame 0: refused by the loader check and by the sidecar builder
    srcd = Path(REPO) / 'datasets' / 'MOVi-Sphere' / scene
    dst = tmp_path / 'conv' / scene
    dst.mkdir(parents=True)
    for p in srcd.iterdir():
        if p.name != 'frame_00000':
            os.symlink(p, dst / p.name)
    (dst / 'frame_00000').mkdir()
    for f in (srcd / 'frame_00000').iterdir():
        V = np.load(f)
        if f.name == 'obj_0_pos.npy':
            V = V.copy()
            V[0, 0] = np.nan
        np.save(dst / 'frame_00000' / f.name, V)
    with pytest.raises(ValueError):
        check_object_counts(ObjectScale(src), str(dst.parent), [scene])
    raw = os.path.join(RAW, 'MOVi-spheres')
    if os.path.isdir(raw):
        with pytest.raises(RuntimeError):
            scene_sizes(raw, str(dst.parent), scene)


def test_verify_all_frames_fails_on_problems(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import verify_all_frames as VF

    class FakePool:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, fn, items, chunksize=1):
            return [{'scene': 'x', 'object_frames': 1, 'max_err': 0.01, 'worst': (0, 0), 'bad': ['obj 0 frame 0: 1e-2'],
                     'nonfinite': 1}]
    monkeypatch.setattr(VF, 'Pool', FakePool)
    monkeypatch.setattr(sys, 'argv', ['verify_all_frames.py', '--out', str(tmp_path)])
    with pytest.raises(SystemExit):
        VF.main()


def test_loader_computes_missing_caches_like_pregen(tmp_path):
    """With the size sidecar active, a missing FPS cache is computed from the stored (unscaled) frame-0 vertices, so it
    equals the pre-generated cache (on the scaled float32 vertices the exact ties of symmetric meshes break differently)."""
    scene = 'movi_7'
    src = Path(REPO) / 'datasets' / 'MOVi-B' / scene
    sidecar = os.path.join(REPO, 'data_generation', 'object_scales_MoviB.json')
    root = tmp_path / json.load(open(sidecar))['converted']           # the converted-dataset name the sidecar records
    dst = root / scene
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.name == 'anchors':
            (dst / 'anchors').mkdir()
            for f in p.iterdir():
                if not f.name.endswith('_cf.npy'):
                    shutil.copy(f, dst / 'anchors' / f.name)
        elif p.name.startswith('frame_'):
            os.symlink(p, dst / p.name)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name)
        else:
            shutil.copy(p, dst / p.name)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train_datasets': [], 'val_datasets': [], 'test_datasets': [scene]}))
    _dataset(str(root), str(split), object_scale_file=sidecar)
    for f in sorted((src / 'anchors').glob('obj_*_anchors_fps4_cf.npy')) + sorted((src / 'anchors').glob('obj_*_fpsorder_cf.npy')):
        assert np.array_equal(np.load(dst / 'anchors' / f.name), np.load(f)), f.name


def test_float_faces_refused(tmp_path):
    root, split = _scene_copy(tmp_path)
    f = Path(root) / 'movi_7' / 'meshes' / 'obj_0_faces.npy'
    F = np.load(f).astype(np.float64)
    F[0, 0] = -0.5
    np.save(f, F)
    with pytest.raises(ValueError, match='dtype'):
        _dataset(root, split)
