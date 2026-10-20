"""The packed layout (frames.npy per scene, src/data/scene_frames.py) holds exactly the per-file vertices:
- on a synthetic scene written in both layouts, load_frame, frame_dirs and num_frames agree, bit for bit;
- on MOVi-B (skipped unless the per-file copy datasets/MOVi-B and a packed copy, PACKED_MOVI_B=<folder>, exist):
  random frames are bit-identical, and the loader returns identical evaluation samples from both."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data.scene_frames import frame_dirs, is_packed, load_frame, num_frames  # noqa: E402

DATA = os.path.join(REPO, 'datasets', 'MOVi-B')
PACKED = os.environ.get('PACKED_MOVI_B', '')
SPLIT = os.path.join(REPO, 'dataset_split_MoviB_1200.json')
SCALES = os.path.join(REPO, 'data_generation', 'object_scales_MoviB.json')


def test_synthetic_scene_both_layouts(tmp_path):
    rng = np.random.default_rng(0)
    counts = [5, 12, 3]
    frames = [[rng.normal(size=(n, 3)).astype(np.float32) for n in counts] for _ in range(7)]
    per_file, packed = tmp_path / 'per_file' / 'movi_1', tmp_path / 'packed' / 'movi_1'
    for t, objs in enumerate(frames):
        (per_file / f'frame_{t:05d}').mkdir(parents=True)
        for i, V in enumerate(objs):
            np.save(per_file / f'frame_{t:05d}' / f'obj_{i}_pos.npy', V)
    packed.mkdir(parents=True)
    np.save(packed / 'frames.npy', np.stack([np.concatenate(objs) for objs in frames]))
    np.save(packed / 'frames_counts.npy', np.array(counts))
    assert not is_packed(per_file) and is_packed(packed)
    assert num_frames(per_file) == num_frames(packed) == 7
    assert [p.name for p in frame_dirs(per_file)] == [p.name for p in frame_dirs(packed)]
    for t in range(7):
        a, b = load_frame(per_file, t), load_frame(packed, t)
        assert len(a) == len(b) == 3
        for x, y in zip(a, b):
            assert x.dtype == y.dtype == np.float32 and np.array_equal(x, y)
    with pytest.raises(IndexError):
        load_frame(packed, 7)


@pytest.mark.skipif(not (os.path.isdir(DATA) and PACKED and os.path.isdir(PACKED)),
                    reason='per-file MOVi-B and PACKED_MOVI_B needed')
def test_movi_b_packed_equals_per_file():
    rng = np.random.default_rng(1)
    scenes = sorted(p.name for p in Path(PACKED).iterdir() if p.is_dir())
    for sc in rng.choice(scenes, 20, replace=False):
        assert num_frames(Path(DATA) / sc) == num_frames(Path(PACKED) / sc)
        for t in rng.choice(num_frames(Path(PACKED) / sc), 5, replace=False):
            for x, y in zip(load_frame(Path(DATA) / sc, int(t)), load_frame(Path(PACKED) / sc, int(t))):
                assert np.array_equal(x, y)


@pytest.mark.skipif(not (os.path.isdir(DATA) and PACKED and os.path.isdir(PACKED)),
                    reason='per-file MOVi-B and PACKED_MOVI_B needed')
def test_loader_samples_identical():
    from src.data.multi_dataloader import MultiObjectDynamics

    def dataset(path):
        return MultiObjectDynamics(
            data_path=path, sequence_length=8, stride=999999, skip_step_size=5, split='test', split_file=SPLIT,
            max_vertices_per_object=None, max_objects_per_frame=16, anchor_k=4, random_step_size={'enable': False},
            augmentation={'enable': False}, anchor_fps_start='centroid_farthest', fps_order=True,
            object_scale_file=SCALES)
    a, b = dataset(DATA), dataset(PACKED)
    assert len(a) == len(b)
    for idx in (0, 7, 59, len(a) - 1):
        sa, sb = a[idx], b[idx]
        for key in ('obj_positions', 'valid_vtx_mask', 'ref_obj_positions', 'anchor_idx', 'vtx_rank', 'frame_ids'):
            assert np.array_equal(sa[key].numpy(), sb[key].numpy()), key
