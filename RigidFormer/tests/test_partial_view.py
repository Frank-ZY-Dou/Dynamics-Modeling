"""data.partial_view (src/data/partial_view.py):
- quat_to_rotmat gives rotations; kabsch recovers a rigid motion exactly, and a perfect rollout of the observed points
  moves the full mesh onto its ground truth (the evaluation's mapping from partial points to full meshes);
- look_at puts the target at the image centre; farthest_point_order starts at the point farthest from the centroid,
  its prefixes are shorter FPS orders, and farthest_point_orders matches it cloud by cloud;
- a ray-cast object is seen only on faces turned to the camera, every observed point lies on its surface, and its far
  side stays unseen;
- boundary erosion removes points but keeps the object, and depth noise moves points along their rays;
- on MOVi-B (skipped when the data are absent): observed points move rigidly with their objects over the sample, the
  anchors are the first FPS points, and evaluation samples carry the observation index and the full meshes."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data.partial_view import (PartialView, farthest_point_order, kabsch, look_at,  # noqa: E402
                                   quat_to_rotmat)

DATA = os.path.join(REPO, 'datasets', 'MOVi-B')
SPLIT = os.path.join(REPO, 'dataset_split_MoviB_1200.json')
SCALES = os.path.join(REPO, 'data_generation', 'object_scales_MoviB.json')
CAMERAS = os.path.join(REPO, 'data_generation', 'scene_cameras_MoviB.json')


def _random_rotation(rng):
    q = rng.normal(size=4)
    return quat_to_rotmat(q / np.linalg.norm(q))


def test_quat_to_rotmat_is_a_rotation():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(100, 4))
    R = quat_to_rotmat(q / np.linalg.norm(q, axis=1, keepdims=True))
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0)
    # 90 degrees about z, (w, x, y, z)
    Rz = quat_to_rotmat(np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    assert np.allclose(Rz @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_kabsch_and_full_mesh_mapping():
    rng = np.random.default_rng(1)
    full = rng.normal(size=(300, 3))                          # full mesh at the observation frame
    seen = full[rng.choice(300, 40, replace=False)]           # observed subset
    Rs = np.stack([_random_rotation(rng) for _ in range(6)])
    ts = rng.normal(size=(6, 3)) * 3
    moved = np.einsum('tij,nj->tni', Rs, seen) + ts[:, None]  # a perfect rollout of the observed points
    R, t = kabsch(seen, moved)
    assert np.allclose(R, Rs, atol=1e-10) and np.allclose(t, ts, atol=1e-10)
    pred_full = np.einsum('tij,nj->tni', R, full) + t[:, None]
    assert np.allclose(pred_full, np.einsum('tij,nj->tni', Rs, full) + ts[:, None], atol=1e-9)
    # a reflection of the points must not be returned as a rotation
    R1, _ = kabsch(seen, (seen * [1, 1, -1])[None])
    assert np.linalg.det(R1[0]) > 0


def test_look_at_and_fps():
    K, R = look_at([10.0, -4.0, 6.0], [0.0, 0.0, 1.5], 58.0, 480, 640)
    Xc = (np.array([0.0, 0.0, 1.5]) - R[:3, 3]) @ R[:3, :3]
    h = Xc @ K.T
    assert Xc[2] < 0 and np.allclose(h[:2] / h[2], [320, 240])
    P = np.random.default_rng(2).normal(size=(500, 3))
    order = farthest_point_order(P, 64)
    assert len(set(order.tolist())) == 64
    assert order[0] == np.argmax(((P - P.mean(0)) ** 2).sum(1))
    # prefix property: the first 16 of a 64-point order are the 16-point order
    assert np.array_equal(farthest_point_order(P, 16), order[:16])
    # several clouds in one pass: the same orders as one by one
    from src.data.partial_view import farthest_point_orders
    clouds = [P[:500], P[:37] * 2.0, P[100:400] + 1.0]
    for c, o in zip(clouds, farthest_point_orders(clouds, 64)):
        assert np.array_equal(o, farthest_point_order(c, 64))


@pytest.mark.parametrize('shape', ['cube', 'teapot', 'torus_knot'])
def test_observation_is_on_the_visible_surface(tmp_path, shape):
    import open3d as o3d
    import trimesh
    from src.data.partial_view import visual_mesh
    cams = {'scenes': {'movi_x': {'K': np.eye(3).tolist(), 'R': np.eye(4).tolist(), 'resolution': [256, 256]}}}
    cams['scenes']['movi_x']['R'][0][3] = 5.0                               # Kubric camera on +x of the origin
    (tmp_path / 'cams.json').write_text(json.dumps(cams))
    pv = PartialView({'enable': True, 'camera_file': str(tmp_path / 'cams.json'), 'eval_camera': 'pulled_back',
                      'distance': 8.0, 'target': [0.0, 0.0, 0.0], 'max_points': 4096, 'min_points': 16},
                     Path(REPO))
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    size = 1.3
    U = pv.observe('movi_x', [shape], np.array([size]), np.zeros((1, 3)), q[None], train=False, rng=rng)[0]
    assert U is not None and U.shape[0] > 300
    V, F = visual_mesh(pv.objects_dir, shape)
    mesh = trimesh.Trimesh(V, F, process=False)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    closest = scene.compute_closest_points(o3d.core.Tensor(U.astype(np.float32)))
    dist = np.linalg.norm(closest['points'].numpy() - U, axis=1)
    tri = closest['primitive_ids'].numpy().astype(np.int64)
    assert dist.max() < 1e-3                       # on the surface (asset units; float32 ray casting at ~8 m)
    Rq = quat_to_rotmat(q)
    eye = np.array([8.0, 0.0, 0.0])                                         # pulled back along +x to distance 8
    facing = ((mesh.face_normals[tri] @ Rq.T) * (eye - size * U @ Rq.T)).sum(axis=1)
    assert np.mean(facing > 0) > 0.99                                       # seen on faces turned to the camera
    # the farthest point from the camera is nearer than the mesh's own farthest point (the back stays unseen)
    far_seen = np.linalg.norm(eye - size * U @ Rq.T, axis=1).max()
    far_mesh = np.linalg.norm(eye - size * V @ Rq.T, axis=1).max()
    assert far_seen < far_mesh


def test_train_augmentations(tmp_path):
    """Boundary erosion removes points but keeps the object; depth noise moves points along their rays."""
    cams = {'scenes': {'movi_x': {'K': np.eye(3).tolist(), 'R': np.eye(4).tolist(), 'resolution': [256, 256]}}}
    cams['scenes']['movi_x']['R'][0][3] = 5.0
    (tmp_path / 'cams.json').write_text(json.dumps(cams))
    base = {'enable': True, 'camera_file': str(tmp_path / 'cams.json'), 'train_camera': 'pulled_back', 'distance': 8.0,
            'target': [0.0, 0.0, 0.0], 'max_points': 100000, 'min_points': 16, 'train_resolutions': [[480, 640]]}
    q = np.array([[0.9, 0.1, 0.3, 0.2]])
    q /= np.linalg.norm(q)
    args = ('movi_x', ['teapot'], np.array([1.0]), np.zeros((1, 3)), q)
    clean = PartialView(base, Path(REPO)).observe(*args, train=True, rng=np.random.default_rng(0))[0]
    eroded = PartialView({**base, 'boundary_erosion_px': 2}, Path(REPO)).observe(*args, train=True,
                                                                               rng=np.random.default_rng(0))[0]
    assert 0 < eroded.shape[0] < clean.shape[0]
    noisy = PartialView({**base, 'depth_noise_std': 1e-3}, Path(REPO)).observe(*args, train=True,
                                                                             rng=np.random.default_rng(0))[0]
    assert noisy.shape == clean.shape                                       # the same rays hit
    import open3d as o3d
    from src.data.partial_view import visual_mesh
    V, F = visual_mesh(str(Path(REPO) / 'data_generation' / 'objects'), 'teapot')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    off = [np.abs(scene.compute_signed_distance(o3d.core.Tensor(P.astype(np.float32))).numpy()) for P in (clean, noisy)]
    assert np.median(off[0]) < 1e-3                                         # on the surface
    assert 0.005 < np.median(off[1]) < 0.2                                  # sigma = 1e-3 * 8^2 m along the rays


@pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(CAMERAS)), reason='MOVi-B data or cameras not available')
def test_movi_b_points_move_rigidly():
    from src.data.multi_dataloader import MultiObjectDynamics
    ds = MultiObjectDynamics(
        data_path=DATA, sequence_length=8, stride=999999, skip_step_size=1, split='test', split_file=SPLIT,
        max_vertices_per_object=None, max_objects_per_frame=16, anchor_k=4, random_step_size={'enable': False},
        augmentation={'enable': False}, anchor_fps_start='centroid_farthest', fps_order=True,
        object_scale_file=SCALES,
        partial_view={'enable': True, 'camera_file': 'data_generation/scene_cameras_MoviB.json'})
    for idx in range(3):
        s = ds[idx]
        X = s['obj_positions'].numpy().astype(np.float64)
        m = s['valid_vtx_mask'].numpy()
        assert m.all(axis=0).any(axis=-1).all()                             # every kept object has points
        for j in range(X.shape[1]):
            P = X[:, j, m[0, j]]
            d0 = np.linalg.norm(P[0, :, None] - P[0, None], axis=-1)
            for t in range(1, X.shape[0]):
                assert np.abs(np.linalg.norm(P[t, :, None] - P[t, None], axis=-1) - d0).max() < 1e-4
        assert np.array_equal(s['anchor_idx'].numpy(), np.tile(np.arange(4), (X.shape[1], 1)))
        assert len(s['partial_full_obs']) >= X.shape[1] and s['partial_obs_index'] == 3
