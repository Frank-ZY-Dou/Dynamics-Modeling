"""Partial point clouds of MOVi objects observed by a simulated depth camera (data.partial_view).

A pinhole depth camera ray-casts the posed visual meshes of a scene (data_generation/objects/<shape>/
visual_geometry.obj, the meshes rendered by Kubric). The points where rays hit object i, i.e. its segment of the
depth image back-projected to 3D, form its observation. Observed points are stored in the object frame (asset units)
and moved with the ground-truth pose to every frame of a sample, so each point keeps its identity over time: the
input is partial, the supervision is the ground-truth motion of the same surface points, and the camera is used only
at the observation frame.

Cameras follow the Kubric convention: they look along their -z axis, K maps camera coordinates to pixels and R is the
camera-to-world matrix. With the original Kubric camera, the ray cast reproduces the `image_positions` and
`visibility` recorded by Kubric (scripts/check_depth_camera.py).

Config (data.partial_view):
    enable: true
    camera_file: data_generation/scene_cameras_MoviB.json   # Kubric camera of every scene (export_scene_cameras.py)
    eval_camera: pulled_back    # scene camera direction, moved back to `distance`; or scene (the Kubric camera)
    train_camera: random        # azimuth, elevation and distance drawn per sample; or pulled_back / scene
    train_azimuth_jitter_deg: 0.0  # train split, pulled_back camera: rotate the direction about the vertical
                                # through `target` by ~ U(-j, j) degrees, keeping its elevation
    distance: 12.0              # metres from `target`
    target: [0.0, 0.0, 1.5]
    fov_y_deg: 58.0
    resolution: [480, 640]      # (H, W) of the evaluation camera
    train_resolutions: [[480, 640]]
    random_azimuth_deg: [0.0, 360.0]
    random_elevation_deg: [20.0, 70.0]
    random_distance: [10.0, 14.0]
    observe_index: 3            # sample frame observed by the camera (the last warm-up frame)
    max_points: 512             # at most this many points per object (farthest-point subsample)
    min_points: 16              # objects with fewer observed points count as unobserved
    depth_noise_std: 0.0        # train split: depth noise std (m) at 1 m, scaled by depth^2
    boundary_erosion_px: 0      # train split: drop pixels within this distance of another segment
    occluder_prob: 0.0          # train split: probability that 1-3 random discs hide the pixels behind them
    occluder_radius: [0.03, 0.12]  # disc radius as a fraction of the image height
    random_max_points: null     # train split: [lo, hi]; keep the first n ~ U{lo..hi} FPS points of each object
    full_object_prob: 0.0       # train split: probability of using the whole collision meshes (vertices in FPS
                                # order, <= max_points) instead of the observed points
    completion: null            # CompletionNet checkpoint (src/data/shape_completion.py): add predicted points on
                                # the hidden side; the union is FPS-subsampled to max_points
    completion_input_points: 256   # observed points read by the completion network (an FPS prefix)
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULTS = {
    'eval_camera': 'pulled_back',
    'train_camera': 'random',
    'train_azimuth_jitter_deg': 0.0,
    'distance': 12.0,
    'target': [0.0, 0.0, 1.5],
    'fov_y_deg': 58.0,
    'resolution': [480, 640],
    'train_resolutions': [[480, 640]],
    'random_azimuth_deg': [0.0, 360.0],
    'random_elevation_deg': [20.0, 70.0],
    'random_distance': [10.0, 14.0],
    'observe_index': 3,
    'max_points': 512,
    'min_points': 16,
    'depth_noise_std': 0.0,
    'boundary_erosion_px': 0,
    'occluder_prob': 0.0,
    'occluder_radius': [0.03, 0.12],
    'random_max_points': None,
    'full_object_prob': 0.0,
    'completion': None,
    'completion_input_points': 256,
}


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4) in (w, x, y, z) order."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def farthest_point_orders(clouds: Sequence[np.ndarray], k: int) -> List[np.ndarray]:
    """Batched farthest-point sampling of clouds (N_i, 3): min(k, N_i) indices per cloud in sampling order, starting
    at the point farthest from the centroid (ties: lowest index), so the first m indices are an m-point FPS subset.
    Distances are computed in float32."""
    sizes = np.array([c.shape[0] for c in clouds])
    n, rows = int(sizes.max()), np.arange(len(clouds))
    X, Y, Z = (np.zeros((len(clouds), n), dtype=np.float32) for _ in range(3))
    valid = np.zeros((len(clouds), n), dtype=bool)
    for i, c in enumerate(clouds):
        c = np.asarray(c, dtype=np.float32)
        X[i, :sizes[i]], Y[i, :sizes[i]], Z[i, :sizes[i]] = c[:, 0], c[:, 1], c[:, 2]
        valid[i, :sizes[i]] = True
    d = np.zeros((len(clouds), n), dtype=np.float32)
    buf = np.empty_like(d)
    for A in (X, Y, Z):
        np.subtract(A, (A.sum(axis=1) / sizes)[:, None].astype(np.float32), out=buf)
        d += buf * buf
    d[~valid] = -np.inf
    sel = np.zeros((len(clouds), min(int(k), n)), dtype=np.int64)
    sel[:, 0] = np.argmax(d, axis=1)
    d.fill(np.inf)
    d[~valid] = -np.inf
    dd = np.empty_like(d)
    for i in range(1, sel.shape[1]):
        last = sel[:, i - 1]
        np.subtract(X, X[rows, last][:, None], out=buf)
        np.multiply(buf, buf, out=dd)
        for A in (Y, Z):
            np.subtract(A, A[rows, last][:, None], out=buf)
            buf *= buf
            dd += buf
        np.minimum(d, dd, out=d)
        d[rows, last] = -np.inf
        sel[:, i] = np.argmax(d, axis=1)
    return [sel[i, :min(int(k), int(sizes[i]))] for i in range(len(clouds))]


def farthest_point_order(P: np.ndarray, k: int) -> np.ndarray:
    """farthest_point_orders of a single cloud."""
    return farthest_point_orders([P], k)[0]


def kabsch(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transforms (R (..., 3, 3), t (..., 3)) minimising |R a + t - b| from points A (N, 3) to B (..., N, 3)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ca = A.mean(axis=0)
    cb = B.mean(axis=-2)
    H = np.einsum('ni,...nj->...ij', A - ca, B - cb[..., None, :])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('...ij,...jk->...ik', U, Vt)))
    D = np.zeros(H.shape)
    D[..., 0, 0] = 1.0
    D[..., 1, 1] = 1.0
    D[..., 2, 2] = d
    R = np.einsum('...ji,...jk,...lk->...il', Vt, D, U)       # V D U^T
    t = cb - np.einsum('...ij,j->...i', R, ca)
    return R, t


def look_at(eye: Sequence[float], target: Sequence[float], fov_y_deg: float, height: int, width: int):
    """Kubric-convention camera at `eye` aimed at `target` (z up): K (pixels) and camera-to-world R (4, 4)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = 0.5 * height / np.tan(np.radians(fov_y_deg) / 2)
    K = np.array([[f, 0.0, -width / 2], [0.0, -f, -height / 2], [0.0, 0.0, -1.0]])
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    R = np.eye(4)
    R[:3, 0], R[:3, 1], R[:3, 2], R[:3, 3] = right, up, back, eye
    return K, R


def view_frame(points: np.ndarray, camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """View frame of an observed cloud X (n, 3; world, z up, floor at z = 0) seen from camera centre (3,).

    Returns the origin (3,), i.e. the centroid; the rotation R (3, 3) whose columns are the frame axes in world
    coordinates (z up, x horizontal towards the camera); and the extra inputs (centroid height, sine of the camera
    elevation, camera distance / 10). View coordinates are (X - origin) @ R."""
    c = np.asarray(points, np.float64).mean(axis=0)
    d = np.asarray(camera, np.float64) - c
    h = float(np.hypot(d[0], d[1]))
    x = np.array([d[0] / h, d[1] / h, 0.0]) if h > 1e-9 else np.array([1.0, 0.0, 0.0])
    R = np.stack([x, np.cross([0.0, 0.0, 1.0], x), [0.0, 0.0, 1.0]], axis=1)
    dist = float(np.linalg.norm(d))
    return c, R, np.array([c[2], d[2] / max(dist, 1e-9), dist / 10.0])


@lru_cache(maxsize=None)
def visual_mesh(objects_dir: str, shape: str) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices (asset units) and faces of a KuBasic visual mesh."""
    import trimesh
    m = trimesh.load(Path(objects_dir) / shape / 'visual_geometry.obj', force='mesh', process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


@lru_cache(maxsize=None)
def collision_points(objects_dir: str, shape: str, k: int) -> np.ndarray:
    """At most k vertices (asset units) of a KuBasic collision mesh in FPS order (the full-object input)."""
    import trimesh
    V = np.asarray(trimesh.load(Path(objects_dir) / shape / 'collision_geometry.obj', force='mesh',
                                process=False).vertices, dtype=np.float64)
    return V[farthest_point_order(V, min(k, len(V)))]


class PartialView:
    """Observations of a scene by the configured depth camera (see the module docstring)."""

    def __init__(self, config: dict, repo_root: Path):
        unknown = sorted(set(config) - set(DEFAULTS) - {'enable', 'camera_file'})
        if unknown:
            raise ValueError(f'data.partial_view: unknown keys {unknown}')
        self.cfg = {**DEFAULTS, **{k: v for k, v in config.items() if k != 'enable'}}
        for key in ('eval_camera', 'train_camera'):
            if self.cfg[key] not in ('pulled_back', 'scene', 'random'):
                raise ValueError(f"data.partial_view.{key} must be pulled_back, scene or random; got {self.cfg[key]}")
        if self.cfg['eval_camera'] == 'random':
            raise ValueError('data.partial_view.eval_camera must be deterministic (pulled_back or scene)')
        self.objects_dir = str(repo_root / 'data_generation' / 'objects')
        cam_file = self.cfg.get('camera_file')
        if not cam_file:
            raise ValueError('data.partial_view.camera_file is required (see data_generation/export_scene_cameras.py)')
        cam_path = Path(cam_file) if Path(cam_file).is_absolute() else repo_root / cam_file
        self.cameras = json.loads(cam_path.read_text())['scenes']
        self.observe_index = int(self.cfg['observe_index'])
        self.max_points = int(self.cfg['max_points'])
        self.min_points = int(self.cfg['min_points'])
        rmp = self.cfg['random_max_points']
        if rmp is not None and not (len(rmp) == 2 and self.min_points <= int(rmp[0]) <= int(rmp[1]) <= self.max_points):
            raise ValueError(f'data.partial_view.random_max_points must be [lo, hi] with min_points <= lo <= hi <= '
                             f'max_points; got {rmp}')
        self.completer = None
        if self.cfg['completion']:
            from .shape_completion import Completer
            path = Path(self.cfg['completion'])
            self.completer = Completer(path if path.is_absolute() else repo_root / path,
                                       int(self.cfg['completion_input_points']))
        # Import Open3D in the process that builds the dataset: DataLoader workers forked after PyTorch has started its
        # OpenMP threads hang when they import it themselves.
        import open3d  # noqa: F401

    def describe(self, train: bool) -> str:
        c = self.cfg
        cam = c['train_camera'] if train else c['eval_camera']
        res = c['train_resolutions'] if train else [c['resolution']]
        aug = ''
        if train:
            aug = (f"; augmentation: depth noise {c['depth_noise_std']}, erosion {c['boundary_erosion_px']} px, "
                   f"occluders p={c['occluder_prob']} r={c['occluder_radius']}, random points {c['random_max_points']}, "
                   f"whole objects p={c['full_object_prob']}")
        comp = (f"; completion {c['completion']} (reads {c['completion_input_points']} points)"
                if c['completion'] else '')
        if train and cam == 'pulled_back' and float(c['train_azimuth_jitter_deg']) > 0:
            cam += f" (azimuth +-{c['train_azimuth_jitter_deg']} deg)"
        return (f"partial view: {cam} camera, resolutions {res}, fov_y {c['fov_y_deg']} deg, distance "
                f"{c['random_distance'] if cam == 'random' else c['distance']} m, observed at sample frame "
                f"{self.observe_index}, <= {self.max_points} points per object, objects with < {self.min_points} "
                f"points not observed{aug}{comp}")

    def camera(self, scene: str, train: bool, rng: np.random.Generator):
        """(K, R, height, width) of the camera that observes `scene`."""
        c = self.cfg
        kind = c['train_camera'] if train else c['eval_camera']
        if kind == 'scene':
            rec = self.cameras[scene]
            H, W = rec.get('resolution', [256, 256])
            return np.asarray(rec['K'], np.float64), np.asarray(rec['R'], np.float64), int(H), int(W)
        target = np.asarray(c['target'], dtype=np.float64)
        if train:
            H, W = c['train_resolutions'][rng.integers(len(c['train_resolutions']))]
        else:
            H, W = c['resolution']
        if kind == 'pulled_back':
            direction = np.asarray(self.cameras[scene]['R'], np.float64)[:3, 3] - target
            direction /= np.linalg.norm(direction)
            jitter = float(c['train_azimuth_jitter_deg'])
            if train and jitter > 0:
                a = np.radians(rng.uniform(-jitter, jitter))
                direction = np.array([np.cos(a) * direction[0] - np.sin(a) * direction[1],
                                      np.sin(a) * direction[0] + np.cos(a) * direction[1], direction[2]])
            dist = float(c['distance'])
        else:
            az = np.radians(rng.uniform(*c['random_azimuth_deg']))
            el = np.radians(rng.uniform(*c['random_elevation_deg']))
            direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            dist = float(rng.uniform(*c['random_distance']))
        K, R = look_at(target + dist * direction, target, float(c['fov_y_deg']), int(H), int(W))
        return K, R, int(H), int(W)

    def whole_objects(self, shapes: Sequence[str]) -> List[np.ndarray]:
        """Every object as its whole collision mesh (vertices, FPS order, <= max_points), the full-object input."""
        return [collision_points(self.objects_dir, s, self.max_points) for s in shapes]

    def observe(self, scene: str, shapes: Sequence[str], sizes: np.ndarray, positions: np.ndarray,
                quaternions: np.ndarray, train: bool, rng: np.random.Generator,
                camera: Optional[tuple] = None) -> List[Optional[np.ndarray]]:
        """Observed points of every object in its own frame (asset units, (n_i, 3), FPS order), None if unobserved.

        positions (N, 3) and quaternions (N, 4; w, x, y, z) are the object poses at the observation frame and sizes
        (N,) the metadata sizes: point u of object i is at positions[i] + sizes[i] R(quaternions[i]) u in the world.
        camera is (K, R, height, width) as returned by camera(), drawn here if None. With completion enabled, the
        predicted points are added.
        """
        import open3d as o3d
        if train and float(self.cfg['full_object_prob']) > 0 and rng.random() < float(self.cfg['full_object_prob']):
            return self.whole_objects(shapes)
        K, Rc, H, W = self.camera(scene, train, rng) if camera is None else camera
        Rq = quat_to_rotmat(quaternions)
        scene_rc = o3d.t.geometry.RaycastingScene()
        Kinv = np.linalg.inv(K)
        cover = np.zeros((H, W), dtype=bool)          # rays are cast only inside the objects' projected boxes
        for i, shape in enumerate(shapes):
            V, F = visual_mesh(self.objects_dir, shape)
            Vw = positions[i] + sizes[i] * V @ Rq[i].T
            scene_rc.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
            Xc = (Vw - Rc[:3, 3]) @ Rc[:3, :3]
            front = Xc[:, 2] < 0
            if front.any():
                h = Xc[front] @ K.T
                uv = h[:, :2] / h[:, 2:3]
                x0, y0 = np.clip(np.floor(uv.min(axis=0)), 0, [W, H]).astype(int)
                x1, y1 = np.clip(np.ceil(uv.max(axis=0)), 0, [W, H]).astype(int)
                cover[y0:y1, x0:x1] = True
        out: List[Optional[np.ndarray]] = [None] * len(shapes)
        if not cover.any():
            return out
        y0, y1 = np.flatnonzero(cover.any(axis=1))[[0, -1]] + [0, 1]
        x0, x1 = np.flatnonzero(cover.any(axis=0))[[0, -1]] + [0, 1]
        inside = cover[y0:y1, x0:x1].reshape(-1)
        u, v = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = np.stack([u.reshape(-1)[inside], v.reshape(-1)[inside], np.ones(int(inside.sum()))], -1) @ Kinv.T @ Rc[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(Rc[:3, 3], d.shape), d], axis=1).astype(np.float32)
        d = rays[:, 3:].astype(np.float64)                             # directions as cast (float32)
        ans = scene_rc.cast_rays(o3d.core.Tensor(rays), nthreads=1)
        t_hit = ans['t_hit'].numpy().astype(np.float64)
        ids = ans['geometry_ids'].numpy().astype(np.int64)
        ids[~np.isfinite(t_hit)] = -1
        if train and int(self.cfg['boundary_erosion_px']) > 0:
            img = np.full(inside.shape, -1, dtype=np.int64)            # pixels outside all boxes are background
            img[inside] = ids
            img = img.reshape(y1 - y0, x1 - x0)
            keep = np.ones_like(img, dtype=bool)
            for _ in range(int(self.cfg['boundary_erosion_px'])):
                edge = np.zeros_like(keep)
                edge[1:] |= img[1:] != img[:-1]
                edge[:-1] |= img[:-1] != img[1:]
                edge[:, 1:] |= img[:, 1:] != img[:, :-1]
                edge[:, :-1] |= img[:, :-1] != img[:, 1:]
                keep &= ~edge
                img = np.where(keep, img, -2)
            ids = np.where(keep.reshape(-1)[inside], ids, -1)
        if train and float(self.cfg['occluder_prob']) > 0 and rng.random() < float(self.cfg['occluder_prob']):
            pu, pv = u.reshape(-1)[inside], v.reshape(-1)[inside]          # pixel centres of the cast rays
            for _ in range(int(rng.integers(1, 4))):                        # 1-3 discs in front of all objects
                cx, cy, r = rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(*self.cfg['occluder_radius']) * H
                ids[(pu - cx) ** 2 + (pv - cy) ** 2 < r * r] = -1
        if train and float(self.cfg['depth_noise_std']) > 0:
            hits = np.isfinite(t_hit)
            t_hit[hits] += rng.normal(size=int(hits.sum())) * float(self.cfg['depth_noise_std']) * t_hit[hits] ** 2
        seen, clouds = [], []
        for i in range(len(shapes)):
            hit = ids == i
            if int(hit.sum()) < self.min_points:
                continue
            Xw = rays[0, :3].astype(np.float64) + t_hit[hit, None] * d[hit]
            U = (Xw - positions[i]) @ Rq[i] / sizes[i]                 # object frame, asset units
            if U.shape[0] > 2 * self.max_points:                         # random subsample of dense segments before FPS
                U = U[np.sort(rng.choice(U.shape[0], 2 * self.max_points, replace=False))]
            seen.append(i)
            clouds.append(U)
        if clouds:
            rmp = self.cfg['random_max_points'] if train else None
            for i, U, order in zip(seen, clouds, farthest_point_orders(clouds, self.max_points)):
                if rmp is not None:                                         # an FPS prefix still covers the segment
                    order = order[:max(self.min_points, int(rng.integers(int(rmp[0]), int(rmp[1]) + 1)))]
                out[i] = U[order]
        if self.completer is not None and seen:
            pred = self.completer.complete([positions[i] + sizes[i] * out[i] @ Rq[i].T for i in seen], Rc[:3, 3])
            union = [np.concatenate([out[i], (Y - positions[i]) @ Rq[i] / sizes[i]]) for i, Y in zip(seen, pred)]
            for i, U, order in zip(seen, union, farthest_point_orders(union, self.max_points)):
                out[i] = U[order]
        return out

    def images(self, scene: str, shapes: Sequence[str], sizes: np.ndarray, positions: np.ndarray,
               quaternions: np.ndarray):
        """Depth image and segmentation of the evaluation camera, for display. depth (H, W): distance along the
        optical axis (m), including the floor (z = 0), inf where nothing is hit; segment (H, W): index of the object
        hit, -1 for floor and background. Same ray cast as observe(), over the whole image and without training
        augmentations."""
        import open3d as o3d
        K, Rc, H, W = self.camera(scene, False, np.random.default_rng(0))
        Rq = quat_to_rotmat(quaternions)
        scene_rc = o3d.t.geometry.RaycastingScene()
        for i, shape in enumerate(shapes):
            V, F = visual_mesh(self.objects_dir, shape)
            Vw = positions[i] + sizes[i] * V @ Rq[i].T
            scene_rc.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([u.reshape(-1), v.reshape(-1), np.ones(H * W)], -1) @ np.linalg.inv(K).T @ Rc[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(Rc[:3, 3], d.shape), d], axis=1).astype(np.float32)
        ans = scene_rc.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans['t_hit'].numpy().astype(np.float64)
        segment = ans['geometry_ids'].numpy().astype(np.int64)
        segment[~np.isfinite(t_hit)] = -1
        down = d[:, 2] < -1e-9                                           # floor (z = 0) where no object is hit
        t_floor = np.where(down, -Rc[2, 3] / np.where(down, d[:, 2], -1.0), np.inf)
        t = np.where(np.isfinite(t_hit), t_hit, t_floor)
        forward = -Rc[:3, 2]                                             # Kubric cameras look along -z
        depth = t * (d @ forward)
        return depth.reshape(H, W), segment.reshape(H, W)
