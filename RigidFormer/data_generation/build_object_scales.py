"""Per-object metadata scale of converted MOVi scenes, as a small JSON sidecar (scene -> list of sizes).

movi_to_sim_data.py writes each object's vertices as R v_asset + p and never applies the metadata `size`, so the
stored meshes do not have the simulated dimensions (MOVi-Sphere: every sphere is ~1.0 m across while the simulated
radii are 0.35 m and 0.7 m). With `data.object_scale_file` pointing at this JSON the loader scales every vertex
array about the object origin: v' = p_t + size (v - p_t), p_t from vis_traj.npy (identical to the metadata positions).

For every converted scene movi_<n> this reads <raw>/<n>/metadata.json, keeps the non-floor instances in the same order
as the converter, and checks that their count equals the number of obj_<i>_pos.npy files in frame_00000, that
vis_traj.npy holds exactly the metadata positions, and that every object's stored frame-0 vertices are the UNSCALED
asset mesh (--assets/<shape>/collision_geometry.obj) under the frame-0 metadata pose: the scale convention is
established from the asset geometry, so a sidecar can never be built for vertex data that already carries the size.

  python data_generation/build_object_scales.py --raw raw/MOVi-spheres-complete \
      --converted datasets/MOVi-Sphere --out data_generation/object_scales_MoviSphere.json
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objects')
_asset_cache = {}


def asset_vertices(assets, shape):
    key = (assets, shape)
    if key not in _asset_cache:
        m = trimesh.load(os.path.join(assets, shape, 'collision_geometry.obj'), process=False)
        if not isinstance(m, trimesh.Trimesh):
            raise RuntimeError(f'{shape}: the collision mesh is not a single mesh')
        _asset_cache[key] = np.asarray(m.vertices, dtype=np.float64)
    return _asset_cache[key]


def scene_sizes(raw_root, conv_root, scene, assets=ASSETS):
    n = scene.split('_', 1)[1]
    meta = json.load(open(os.path.join(raw_root, n, 'metadata.json')))
    insts = [i for i in (meta.get('instances') or meta.get('objects') or [])
             if (i.get('shape') or i.get('asset_id') or '').lower() != 'floor']
    frame0 = os.path.join(conv_root, scene, 'frame_00000')
    n_obj = len([f for f in os.listdir(frame0) if f.startswith('obj_') and f.endswith('_pos.npy')])
    if len(insts) != n_obj:
        raise RuntimeError(f'{scene}: {len(insts)} metadata instances vs {n_obj} converted objects')
    sizes = []
    for inst in insts:
        s = inst['size']
        if isinstance(s, (list, tuple)):
            if not np.allclose(s, s[0]):
                raise RuntimeError(f'{scene}: non-uniform size {s}')
            s = s[0]
        if not (np.isfinite(s) and s > 0):
            raise RuntimeError(f'{scene}: size {s} is not a positive finite number')
        sizes.append(float(s))
    vt = np.load(os.path.join(conv_root, scene, 'vis_traj.npy'))[0]          # (T, Nobj, 7)
    if not np.isfinite(vt).all() or not all(np.isfinite(np.asarray(i[k], dtype=np.float64)).all()
                                            for i in insts for k in ('positions', 'quaternions')):
        raise RuntimeError(f'{scene}: non-finite poses in vis_traj.npy or the metadata')
    P = np.stack([np.asarray(i['positions'], dtype=np.float32)[:vt.shape[0]] for i in insts], 1)
    err = float(np.abs(vt[:, :, :3] - P).max())
    if not err <= 1e-5:                                                     # (a NaN fails this test too)
        raise RuntimeError(f'{scene}: vis_traj positions differ from the metadata by {err}')
    # stored frame-0 vertices = R(q_0) v_asset + p_0 with the unscaled asset mesh (and not with the size applied)
    for o, (inst, size) in enumerate(zip(insts, sizes)):
        V = np.load(os.path.join(frame0, f'obj_{o}_pos.npy')).astype(np.float64)
        A = asset_vertices(assets, (inst.get('shape') or inst.get('asset_id')))
        R = trimesh.transformations.quaternion_matrix(np.asarray(inst['quaternions'][0], dtype=np.float64))[:3, :3]
        p = np.asarray(inst['positions'][0], dtype=np.float64)
        err = np.abs(V - (A @ R.T + p)).max() if V.shape == A.shape else np.inf
        if not err <= 1e-5:                                                 # (a NaN fails this test too)
            err_scaled = np.abs(V - (size * A @ R.T + p)).max() if V.shape == A.shape else np.inf
            raise RuntimeError(f'{scene} object {o}: stored frame-0 vertices are not the unscaled asset mesh under the '
                               f'metadata pose (max error {err:.2e} m; with the size applied {err_scaled:.2e} m)')
    # radius0: each object's largest vertex distance from its origin at frame 0, as stored (unscaled): the loader
    # checks it, so vertex data that already carries the size (re-converted) is never scaled a second time
    radius0 = [round(radius_from_origin(os.path.join(frame0, f'obj_{o}_pos.npy'), vt[0, o, :3]), 6) for o in range(n_obj)]
    if not (np.isfinite(radius0).all() and np.isfinite(vt[0, 0, :3]).all()):
        raise RuntimeError(f'{scene}: non-finite radius0 {radius0} or origin {vt[0, 0, :3]}')
    # fingerprint: object 0's origin at frame 0 (MOVi-A and MOVi-Sphere share seeds, hence object counts)
    return scene, (sizes, [round(float(x), 6) for x in vt[0, 0, :3]], radius0)


def radius_from_origin(vertex_path, origin):
    V = np.load(vertex_path).astype(np.float64)
    return float(np.linalg.norm(V - np.asarray(origin, dtype=np.float64), axis=1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='raw MOVi root with <n>/metadata.json')
    ap.add_argument('--converted', required=True, help='converted root with movi_<n>/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--assets', default=ASSETS, help='mesh library with <shape>/collision_geometry.obj')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    scenes = sorted(d for d in os.listdir(args.converted) if d.startswith('movi_'))
    res = dict(ThreadPoolExecutor(args.workers).map(lambda s: scene_sizes(args.raw, args.converted, s, args.assets), scenes))
    allsz = np.concatenate([np.array(v[0]) for v in res.values()])
    json.dump({'source': os.path.basename(os.path.realpath(args.raw)),
               'converted': os.path.basename(os.path.realpath(args.converted)),
               'note': 'metadata size per object, converter order (floor skipped); fingerprint = object 0 origin at frame 0; '
                       'radius0 = largest stored (unscaled) vertex distance from the origin at frame 0, per object',
               'scales': {k: v[0] for k, v in res.items()},
               'fingerprint': {k: v[1] for k, v in res.items()},
               'radius0': {k: v[2] for k, v in res.items()}}, open(args.out, 'w'), separators=(',', ':'), allow_nan=False)
    print(f'{args.out}: {len(res)} scenes, {allsz.size} objects, size min {allsz.min():.3f} max {allsz.max():.3f} '
          f'mean {allsz.mean():.3f}; distinct values {len(np.unique(np.round(allsz, 6)))}')


if __name__ == '__main__':
    main()
