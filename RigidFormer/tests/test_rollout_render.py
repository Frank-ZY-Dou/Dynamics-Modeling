"""Geometry of the rollout renderer (src/utils/rollout_render.py): pose fitting, OBJ loading, colours. No GPU needed."""
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.rollout_render import fit_poses, linear_rgb, load_obj


def test_fit_poses_recovers_rigid_motions():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 3))
    R = Rotation.random(8, random_state=1).as_matrix()
    t = rng.normal(size=(8, 3)) * 3
    frames = np.einsum('tij,vj->tvi', R, reference) + t[:, None]
    poses, resid = fit_poses(reference, frames)
    assert resid < 1e-9
    assert np.allclose(poses[:, :3, :3], R, atol=1e-9) and np.allclose(poses[:, :3, 3], t, atol=1e-9)
    assert np.allclose(poses[:, 3], [0, 0, 0, 1])


def test_fit_poses_reports_non_rigid_frames():
    reference = np.random.default_rng(2).normal(size=(30, 3))
    _, resid = fit_poses(reference, (reference * 1.1)[None])
    assert resid > 1e-2


def test_load_obj_keeps_hard_edges(tmp_path):
    # a unit square folded along its diagonal; the file gives each face its own normal at the shared edge
    path = tmp_path / 'fold.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 1\n'
                    'vn 0 0 1\nvn 0 -1 1\n'
                    'f 1//1 2//1 3//1\nf 1//2 3//2 4//2\n')
    V, N, F = load_obj(str(path))
    assert len(V) == 6 and F.shape == (2, 3)            # vertices 1 and 3 appear once per face
    assert np.allclose(N[F[0]], [0, 0, 1]) and np.allclose(N[F[1]], np.array([1, -1, 1]) / np.sqrt(3))
    assert np.allclose(V[F[1]], [[0, 0, 0], [1, 1, 0], [0, 1, 1]])


def test_load_obj_weights_normals_by_face_area(tmp_path):
    # a large flat triangle and a small one tilted by 27 degrees share a smooth vertex: its normal follows the
    # large face
    path = tmp_path / 'bevel.obj'
    path.write_text('v 0 0 0\nv 10 0 0\nv 0 10 0\nv 0 -0.1 0.05\nv 0.1 0 0\nvn 0 0 1\n'
                    'f 1//1 2//1 3//1\nf 1//1 4//1 5//1\n')
    V, N, F = load_obj(str(path))
    shared = int(np.flatnonzero((V == 0).all(axis=1))[0])
    assert np.degrees(np.arccos(N[shared] @ [0, 0, 1])) < 0.1


def test_load_obj_triangulates_polygons(tmp_path):
    path = tmp_path / 'quad.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\n')
    V, N, F = load_obj(str(path))
    assert len(V) == 4 and F.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_linear_rgb():
    assert np.allclose(linear_rgb('#ffffff'), 1) and np.allclose(linear_rgb('#000000'), 0)
    assert np.allclose(linear_rgb('#808080'), 0.2158605, atol=1e-6)
