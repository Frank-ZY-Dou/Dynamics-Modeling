"""Check the simulated depth camera of data.partial_view against what Kubric recorded.

With every scene's own Kubric camera, the ray cast of the posed visual meshes must reproduce
  1. `image_positions`: the object origins projected into the image, and
  2. `visibility`: the number of pixels each object covers,
which tests the camera convention, the poses, the sizes, the meshes and the occlusions together.

    python scripts/check_depth_camera.py --dataset RigidFormer-MOVi --config movi_b --scenes 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data.partial_view import quat_to_rotmat, visual_mesh  # noqa: E402


def segment(K, R, res, meshes):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    for V, F in meshes:
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    H, W = res
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3) @ np.linalg.inv(K).T @ R[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.concatenate([np.broadcast_to(R[:3, 3], d.shape), d], axis=1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    ids = ans['geometry_ids'].numpy().astype(np.int64)
    ids[~np.isfinite(ans['t_hit'].numpy())] = -1
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='local copy of the dataset repository (data/, assets/)')
    ap.add_argument('--config', default='movi_b', choices=['movi_sphere', 'movi_a', 'movi_b'])
    ap.add_argument('--scenes', type=int, default=20, help='first test scenes to check')
    ap.add_argument('--frames', type=int, nargs='+', default=[0, 3, 100, 300])
    args = ap.parse_args()
    objects_dir = str(Path(__file__).resolve().parents[1] / 'data_generation' / 'objects')
    rows = []
    for f in sorted((Path(args.dataset) / 'data' / args.config).glob('test-*.parquet')):
        rows += pq.read_table(f, columns=['scene', 'shape', 'size', 'positions', 'quaternions', 'camera', 'visibility',
                                          'image_positions', 'resolution']).to_pylist()
    proj, rel = [], []
    for r in rows[:args.scenes]:
        K, R = np.asarray(r['camera']['K'], np.float64), np.asarray(r['camera']['R'], np.float64)
        P, Q = np.asarray(r['positions'], np.float64), np.asarray(r['quaternions'], np.float64)
        vis, ipos = np.asarray(r['visibility']), np.asarray(r['image_positions'], np.float64)
        for t in args.frames:
            h = ((P[t] - R[:3, 3]) @ R[:3, :3]) @ K.T
            proj.append(np.abs(h[:, :2] / h[:, 2:3] - ipos[t]).max())
            Rq = quat_to_rotmat(Q[t])
            meshes = []
            for i, (shape, size) in enumerate(zip(r['shape'], r['size'])):
                V, F = visual_mesh(objects_dir, shape)
                meshes.append((P[t, i] + size * V @ Rq[i].T, F))
            ids = segment(K, R, tuple(r['resolution']), meshes)
            for i in range(len(meshes)):
                if vis[t, i] >= 50:
                    rel.append(abs(int((ids == i).sum()) - int(vis[t, i])) / vis[t, i])
    rel = np.asarray(rel)
    print(f'{args.config}, {min(args.scenes, len(rows))} test scenes, frames {args.frames}')
    print(f'  projected origins vs image_positions: max error {max(proj):.4f} px')
    print(f'  object pixels vs visibility ({rel.size} objects with >= 50 px): relative difference median '
          f'{np.median(rel):.3%}, 95th percentile {np.percentile(rel, 95):.3%}, max {rel.max():.3%}')
    ok = max(proj) < 0.01 and np.percentile(rel, 95) < 0.01
    print('  PASS' if ok else '  FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
