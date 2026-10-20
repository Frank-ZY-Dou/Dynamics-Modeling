"""data.partial_view.completion (src/data/shape_completion.py):
- the view frame has its origin at the centroid, z up and x horizontal towards the camera, with the documented extra
  inputs, and turning the scene about the vertical turns the completed points with it;
- the network ignores padded points, and its output does not depend on the order of the observed points;
- surface points lie on the collision mesh and are reproducible for a given seed;
- with completion, every observed object has exactly max_points points (at most max_points without), and whole
  objects are the collision-mesh vertices."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data.partial_view import PartialView, view_frame  # noqa: E402
from src.data.shape_completion import Completer, CompletionNet, surface_points  # noqa: E402

OBJECTS = os.path.join(REPO, 'data_generation', 'objects')
CAMERAS = os.path.join(REPO, 'data_generation', 'scene_cameras_MoviB.json')


def _yaw(a):
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    net = CompletionNet(n_out=64, latent=32)
    path = tmp_path_factory.mktemp('completion') / 'c.pth'
    torch.save({'net': {'n_out': 64, 'latent': 32}, 'state_dict': net.state_dict()}, path)
    return str(path)


def test_view_frame():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)) + [2.0, -1.0, 1.0]
    cam = np.array([10.0, 4.0, 6.0])
    c, R, extra = view_frame(X, cam)
    assert np.allclose(c, X.mean(0)) and np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], [0, 0, 1])
    d = (cam - c) @ R                                       # the camera lies in the x > 0 half of the x-z plane
    assert abs(d[1]) < 1e-9 and d[0] > 0
    assert np.isclose(extra[0], c[2]) and np.isclose(extra[1], d[2] / np.linalg.norm(d))
    assert np.isclose(extra[2], np.linalg.norm(d) / 10.0)


def test_network_padding_and_order():
    torch.manual_seed(0)
    net = CompletionNet(n_out=32, latent=16).eval()
    pts = torch.randn(1, 20, 3)
    extra = torch.randn(1, 3)
    out = net(pts, torch.ones(1, 20, dtype=torch.bool), extra)
    padded = torch.cat([pts, 100 * torch.randn(1, 7, 3)], dim=1)
    mask = torch.zeros(1, 27, dtype=torch.bool)
    mask[:, :20] = True
    assert out.shape == (1, 32, 3)
    assert torch.allclose(out, net(padded, mask, extra), atol=1e-5)
    perm = torch.randperm(20)
    assert torch.allclose(out, net(pts[:, perm], torch.ones(1, 20, dtype=torch.bool), extra), atol=1e-5)


def test_surface_points_on_mesh():
    import open3d as o3d
    import trimesh
    m = trimesh.load(os.path.join(OBJECTS, 'cube', 'collision_geometry.obj'), force='mesh', process=False)
    P = surface_points(OBJECTS, 'cube', 500, seed=3)
    assert P.shape == (500, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
    assert scene.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy().max() < 1e-4   # float32
    assert np.array_equal(P, surface_points(OBJECTS, 'cube', 500, seed=3))


def test_completion_turns_with_the_scene(checkpoint):
    comp = Completer(checkpoint, input_points=40)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3)) * 0.3 + [1.0, 2.0, 0.6]
    cam = np.array([8.0, -3.0, 5.0])
    Y = comp.complete([X], cam)[0]
    Rz = _yaw(0.7)
    Y2 = comp.complete([X @ Rz.T], Rz @ cam)[0]
    assert Y.shape == (64, 3) and np.allclose(Y2, Y @ Rz.T, atol=1e-4)


def test_partial_view_with_completion(checkpoint):
    cfg = {'camera_file': CAMERAS, 'completion': checkpoint, 'completion_input_points': 64, 'max_points': 96}
    pv = PartialView(cfg, Path(REPO))
    pos = np.array([[0.0, 0.0, 0.5], [1.5, 0.5, 0.5]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0], [np.cos(0.3), 0.0, 0.0, np.sin(0.3)]])
    sizes = np.array([1.0, 0.8])
    rng = np.random.default_rng(2)
    cam = pv.camera('movi_1', True, rng)
    with_c = pv.observe('movi_1', ['cube', 'sphere'], sizes, pos, quat, True, np.random.default_rng(5), camera=cam)
    pv.completer = None
    without = pv.observe('movi_1', ['cube', 'sphere'], sizes, pos, quat, True, np.random.default_rng(5), camera=cam)
    for U, V in zip(with_c, without):
        assert U is not None and V is not None
        assert len(U) == 96 and len(V) <= 96
    whole = pv.whole_objects(['cube', 'sphere'])
    assert [len(U) for U in whole] == [51, 64]
