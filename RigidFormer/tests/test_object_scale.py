"""data.object_scale_file (src/data/object_scale.py) on the real MOVi-Sphere data (skipped when the data is absent):
- without a scale file the dataset returns exactly the stored vertices;
- with it every object satisfies v' - p_t = size (v - p_t) about the vis_traj origin, the stored (cached) arrays are
  left untouched, and the anchor indices do not change;
- resting objects of both sizes end up with their lowest vertex just above the floor (0 < z < 0.01 m), where the
  unscaled meshes sit at ~ -0.15 m (small) and ~ +0.2 m (large)."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'datasets', 'MOVi-Sphere')
SPLIT = os.path.join(REPO, 'dataset_split_MoviSphere_1200.json')
SCALES = os.path.join(REPO, 'data_generation', 'object_scales_MoviSphere.json')

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason='MOVi-Sphere data not available')


def _dataset(scale_file):
    from src.data.multi_dataloader import MultiObjectDynamics
    return MultiObjectDynamics(
        data_path=DATA, sequence_length=8, stride=999999, skip_step_size=1, split='test', split_file=SPLIT,
        max_vertices_per_object=None, max_objects_per_frame=16, anchor_k=4, random_step_size={'enable': False},
        augmentation={'enable': False}, anchor_fps_start='centroid_farthest', fps_order=True,
        object_scale_file=scale_file)


def test_scale_about_origin_and_floor():
    from pathlib import Path
    from src.data.object_scale import ObjectScale
    osc = ObjectScale(SCALES)
    scene = json.load(open(SPLIT))['test_datasets'][0]
    sdir = Path(DATA) / scene
    vt = np.load(sdir / 'vis_traj.npy')[0]
    sizes = osc.scales[scene]
    lows = {0.7: [], 1.4: []}
    for t in (0, 240, 479):
        for i in range(len(sizes)):
            path = sdir / f'frame_{t:05d}' / f'obj_{i}_pos.npy'
            V = np.load(path)
            V0 = V.copy()
            W = osc.apply(path, V)
            assert np.array_equal(V, V0)                               # input untouched
            p = vt[t, i, :3].astype(np.float64)
            assert np.allclose(W - p, sizes[i] * (V.astype(np.float64) - p), atol=1e-5)
            if t == 479 and vt[t, i, 2] < 0.5 * sizes[i] + 0.01:           # centre within 1 cm of floor contact
                lows[round(float(sizes[i]), 1)].append(float(W[:, 2].min()))
    assert sum(len(v) for v in lows.values()) > 0
    for s, v in lows.items():
        for z in v:
            assert 0.0 < z < 0.01, (s, z)


def test_dataset_default_unchanged_and_anchors_invariant():
    plain = _dataset(None)
    scaled = _dataset(SCALES)
    a, b = plain[0], scaled[0]
    assert a['dataset_name'] == b['dataset_name']
    assert np.array_equal(a['anchor_idx'].numpy(), b['anchor_idx'].numpy())
    # plain dataset = the stored vertices, bit for bit
    from pathlib import Path
    sdir = Path(DATA) / a['dataset_name']
    fid = int(a['frame_ids'][0])
    V = np.load(sdir / f'frame_{fid:05d}' / 'obj_0_pos.npy')
    assert np.array_equal(a['obj_positions'][0, 0, :V.shape[0]].numpy(), V)
    # scaled dataset differs by the metadata scale about the origin
    vt = np.load(sdir / 'vis_traj.npy')[0]
    s = json.load(open(SCALES))['scales'][a['dataset_name']][0]
    p = vt[fid, 0, :3]
    assert np.allclose(b['obj_positions'][0, 0, :V.shape[0]].numpy() - p, s * (V - p), atol=1e-5)


def test_sidecar_of_another_dataset_is_refused():
    other = os.path.join(REPO, 'data_generation', 'object_scales_MoviA.json')
    with pytest.raises(ValueError):
        _dataset(other)


def test_object_count_check():
    from src.data.object_scale import ObjectScale, check_object_counts
    osc = ObjectScale(os.path.join(REPO, 'data_generation', 'object_scales_MoviA.json'))
    scenes = json.load(open(SPLIT))['test_datasets'][:50]
    with pytest.raises(ValueError):
        check_object_counts(osc, DATA, scenes)          # MOVi-A sizes against MOVi-Sphere scenes
    check_object_counts(ObjectScale(SCALES), DATA, scenes)
