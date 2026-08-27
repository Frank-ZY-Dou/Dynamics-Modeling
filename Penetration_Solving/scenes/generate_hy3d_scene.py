"""Deterministic HY3D-Bench scene generator.

Mirrors generate_kubric_scene's API (returns List[MeshObject]).
Same (N, seed, max_faces, hy3d_dir) -> identical mesh list and
identical initial poses.

HY3D mesh directories share the Kubric pool layout
(data.json + collision_geometry.obj + visual_geometry.obj) but report
num_faces instead of nr_vertices as the decimation budget; some packaged
meshes slipped past the ≤5000-face cap.
We re-filter here by num_faces to keep narrowphase cost bounded and match
HOP-Net/Thingi10K complexity (≤ 5000 faces ≈ ≤ 2500 verts).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's4r'))
from mesh_collision import MeshObject, _load_json_relaxed
from box_collision import euler_to_rotation_matrix


def scan_eligible_hy3d(
    hy3d_dir: str | Path,
    max_faces: int = 5000,
    min_faces: int = 500,
    require_watertight: bool = True,
) -> List[Path]:
    """Return the list of HY3D entries that pass our face-count window and,
    by default, the strict ``is_watertight AND is_volume`` filter.  The
    watertight check uses ``trimesh.load(process=False)`` on each
    ``collision_geometry.obj`` and is cached on disk at
    ``<hy3d_dir>/_good_pool.json`` so subsequent runs are O(1).
    """
    hy3d_dir = Path(hy3d_dir)
    cache_path = hy3d_dir / "_good_pool.json"
    good_set = None
    if require_watertight and cache_path.exists():
        try:
            cached = _load_json_relaxed(cache_path)
            good_set = set(cached.get("good_meshes", []))
        except Exception:
            good_set = None
    out: List[Path] = []
    for d in sorted(hy3d_dir.iterdir()):
        if not d.is_dir():
            continue
        dj = d / "data.json"
        cg = d / "collision_geometry.obj"
        if not (dj.exists() and cg.exists()):
            continue
        try:
            data = _load_json_relaxed(dj)
            nf = int(data.get("num_faces", 0))
            if nf < min_faces or nf > max_faces:
                continue
            if require_watertight:
                if good_set is not None:
                    if d.name not in good_set:
                        continue
                else:
                    # No cache: do the actual trimesh check (slow).
                    import trimesh as _tm
                    m = _tm.load(str(cg), process=False)
                    if not (isinstance(m, _tm.Trimesh)
                            and m.is_watertight and m.is_volume):
                        continue
            out.append(d)
        except Exception:
            continue
    return out


def load_hy3d_object(obj_dir: Path, target_size: float = 0.1) -> dict:
    data = _load_json_relaxed(obj_dir / "data.json")
    bounds = data["kwargs"]["bounds"]
    bbox_min = np.array(bounds[0], dtype=np.float64)
    bbox_max = np.array(bounds[1], dtype=np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    max_extent = float((bbox_max - bbox_min).max())
    nf = target_size / max_extent if max_extent > 1e-12 else 1.0

    coll = trimesh.load(str(obj_dir / "collision_geometry.obj"),
                        process=False, force="mesh")
    coll_v = np.asarray(coll.vertices, dtype=np.float64) - bbox_center
    coll_f = np.asarray(coll.faces, dtype=np.int32)

    vis_path = obj_dir / "visual_geometry.obj"
    if vis_path.exists():
        vis = trimesh.load(str(vis_path), process=False, force="mesh")
        vis_v = np.asarray(vis.vertices, dtype=np.float64) - bbox_center
        vis_f = np.asarray(vis.faces, dtype=np.int32)
    else:
        vis_v, vis_f = coll_v, coll_f

    return dict(
        # Use the directory name (full, with the hy3d_<idx>_<hash> prefix) so
        # downstream baselines that look up per-object data files by
        # ``obj.name`` downstream consumers hit the right path.
        # The bare ``data["id"]`` is shorter but doesn't match HY3D's on-disk
        # layout.
        name=obj_dir.name,
        collision_verts=coll_v,
        collision_faces=coll_f,
        visual_verts=vis_v,
        visual_faces=vis_f,
        normalize_factor=nf,
    )


def generate_hy3d_scene(
    n_objects: int,
    target_size: float,
    spawn_range_xz: float,
    spawn_range_y: float,
    seed: int,
    hy3d_dir: str | Path,
    max_faces: int = 5000,
    allow_repeat: bool = True,
) -> List[MeshObject]:
    eligible = scan_eligible_hy3d(hy3d_dir, max_faces=max_faces)
    if not eligible:
        raise ValueError(f"No eligible HY3D objects under {hy3d_dir}.")

    rng = np.random.RandomState(seed)
    replace = allow_repeat and (n_objects > len(eligible))
    if not replace and n_objects > len(eligible):
        raise ValueError(
            f"Only {len(eligible)} eligible objects (need {n_objects}). "
            f"Set allow_repeat=True or raise max_faces.")
    picked = rng.choice(len(eligible), n_objects, replace=replace)
    picked_dirs = [eligible[idx] for idx in picked]

    objects: List[MeshObject] = []
    for obj_dir in picked_dirs:
        info = load_hy3d_object(obj_dir, target_size)
        px = rng.uniform(-spawn_range_xz, spawn_range_xz)
        py = rng.uniform(-spawn_range_y, spawn_range_y)
        pz = rng.uniform(-spawn_range_xz, spawn_range_xz)
        rot = rng.uniform(0, 2 * np.pi, 3)
        R = euler_to_rotation_matrix(*rot)
        objects.append(MeshObject(
            name=info["name"],
            center=np.array([px, py, pz]),
            rotation=R,
            collision_verts_model=info["collision_verts"],
            collision_faces=info["collision_faces"],
            visual_verts_model=info["visual_verts"],
            visual_faces=info["visual_faces"],
            normalize_factor=info["normalize_factor"],
            inv_mass=1.0,
        ))
    return objects
