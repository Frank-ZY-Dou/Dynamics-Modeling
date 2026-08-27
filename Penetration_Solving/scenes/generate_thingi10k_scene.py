"""Generate scenes from Thingi10K for the S4R-QP cross-dataset benchmark.

Filters for closed, manifold, non-self-intersecting meshes with 500-5000
vertices (matching the Kubric pool's mesh-complexity band). The same generator
API as generate_kubric_scene: returns MeshObject list.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
from typing import List

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's4r'))
from mesh_collision import MeshObject
from box_collision import euler_to_rotation_matrix

_CACHE = {'ds': None, 'ids': None}


_MAX_FACETS_DEFAULT = int(os.environ.get('THINGI10K_MAX_FACETS', '1500'))
_VALID_IDX_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '_thingi10k_valid_idx.json')


def _load_thingi10k_dataset(max_facets: int | None = None):
    """Load (and cache) the filtered Thingi10K dataset.

    Defaults to ≤1500 facets to match the decimated HY3D pool — runtime on
    the S4R sweep was dominated by narrowphase cost against 5000-face meshes,
    so keeping both datasets in the same complexity band gives a fair
    cross-dataset comparison.
    """
    cap = int(max_facets if max_facets is not None else _MAX_FACETS_DEFAULT)
    if _CACHE['ds'] is not None and _CACHE.get('cap') == cap:
        return _CACHE['ds']
    import thingi10k
    thingi10k.init()
    ds = thingi10k.dataset(
        num_vertices=(500, 5000),
        num_facets=(1000, cap),
        closed=True,
        manifold=True,
        self_intersecting=False,
    )
    _CACHE['ds'] = ds
    _CACHE['cap'] = cap
    _CACHE['ids'] = list(range(len(ds)))
    print(f"[Thingi10K] dataset loaded: {len(ds)} eligible meshes "
          f"(≤{cap} facets)", flush=True)
    return ds


def valid_thingi10k_indices(max_facets: int | None = None,
                              rebuild: bool = False) -> list:
    """Return (cached) list of dataset indices whose meshes pass our load +
    geometric sanity checks (trimesh bounds_tree). First call at a given cap
    runs a ~2 min full scan and persists the result; subsequent calls are O(1).
    """
    cap = int(max_facets if max_facets is not None else _MAX_FACETS_DEFAULT)
    key = f'cap{cap}'
    cache = {}
    import json, time
    if os.path.exists(_VALID_IDX_CACHE) and not rebuild:
        try:
            cache = json.load(open(_VALID_IDX_CACHE))
        except Exception:
            cache = {}
    if key in cache:
        return list(cache[key])

    ds = _load_thingi10k_dataset(cap)
    good = []
    t0 = time.time()
    for i in range(len(ds)):
        try:
            _ = load_thingi10k_object(i, target_size=0.1)
            good.append(i)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"[Thingi10K] validating {i+1}/{len(ds)} "
                  f"good={len(good)} ({time.time()-t0:.1f}s)", flush=True)
    cache[key] = good
    with open(_VALID_IDX_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"[Thingi10K] valid pool at cap={cap}: "
          f"{len(good)}/{len(ds)} in {time.time()-t0:.1f}s", flush=True)
    return good


_HF_CACHE = os.environ.get('HF_HOME', '')


def _remap_file_path(p: str) -> str:
    """Row.file_path was baked in at packaging time to absolute paths on the
    packager's machine. Remap the `datasets/downloads/extracted/...` suffix
    onto our local HF_HOME so the cache works anywhere it's unpacked.
    """
    anchor = '/datasets/downloads/extracted/'
    if anchor in p and _HF_CACHE:
        return _HF_CACHE.rstrip('/') + anchor + p.split(anchor, 1)[1]
    return p


def load_thingi10k_object(idx: int, target_size: float = 0.1) -> dict:
    import thingi10k
    ds = _load_thingi10k_dataset()
    row = ds[idx]
    # thingi10k v1.4.x rows contain file_path (to an .npz). Load via helper.
    fpath = _remap_file_path(str(row['file_path']))
    verts, faces = thingi10k.load_file(fpath)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    # A few meshes in Thingi10K come back as flat arrays or with non-triangle
    # facet counts (e.g. thingi_39815 returns faces.shape=(2932,)). Reshape if
    # possible, reject otherwise so generate_thingi10k_scene can skip.
    if faces.ndim == 1:
        if faces.size % 3 == 0:
            faces = faces.reshape(-1, 3)
        else:
            raise ValueError(
                f"thingi10k idx={idx}: faces has {faces.size} elements "
                f"(not divisible by 3), cannot reshape to triangles")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"thingi10k idx={idx}: expected (M,3) faces, got {faces.shape}")
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise ValueError(
            f"thingi10k idx={idx}: expected (N,3) verts, got {verts.shape}")
    if faces.max() >= len(verts):
        raise ValueError(
            f"thingi10k idx={idx}: face index {faces.max()} out of bounds "
            f"(N={len(verts)})")
    # Geometric sanity: some meshes have collinear-triangle issues that make
    # trimesh.proximity (used by our evaluator) crash on bounds_tree. Probe
    # it cheaply before accepting.
    import trimesh  # local import, already imported above
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        _ = tm.triangles_tree  # lazy — forces bounds_tree evaluation
    except Exception as exc:
        raise ValueError(
            f"thingi10k idx={idx}: degenerate geometry — "
            f"triangles_tree failed: {exc}") from exc
    bbox_min = verts.min(axis=0); bbox_max = verts.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    max_extent = float((bbox_max - bbox_min).max())
    nf = target_size / max_extent if max_extent > 1e-12 else 1.0
    verts_centered = verts - center
    return dict(
        name=f"thingi_{row.get('thing_id', idx) if hasattr(row, 'get') else idx}",
        collision_verts=verts_centered,
        collision_faces=faces,
        visual_verts=verts_centered,
        visual_faces=faces,
        normalize_factor=nf,
    )


def generate_thingi10k_scene(
    n_objects: int,
    target_size: float,
    spawn_range_xz: float,
    spawn_range_y: float,
    seed: int,
    max_verts: int = 5000,
    allow_repeat: bool = True,
) -> List[MeshObject]:
    ds = _load_thingi10k_dataset()
    # Sample only from the pre-validated pool (built on first call, cached).
    valid_pool = valid_thingi10k_indices()
    n_eligible = len(valid_pool)
    if n_eligible == 0:
        raise RuntimeError("Thingi10K valid pool is empty.")

    rng = np.random.RandomState(seed)
    replace = allow_repeat and (n_objects > n_eligible)
    if not replace and n_objects > n_eligible:
        raise ValueError(f"Only {n_eligible} eligible (need {n_objects}).")
    # All pool indices already pre-validated, so a small 5% over-sample is
    # enough to absorb rare load-time races (e.g. transient I/O hiccups).
    over = int(np.ceil(n_objects * 1.05))
    local = rng.choice(n_eligible, over, replace=replace or (over > n_eligible))
    picked = np.array([valid_pool[i] for i in local])

    objects: List[MeshObject] = []
    for idx in picked:
        if len(objects) >= n_objects:
            break
        px = rng.uniform(-spawn_range_xz, spawn_range_xz)
        py = rng.uniform(-spawn_range_y, spawn_range_y)
        pz = rng.uniform(-spawn_range_xz, spawn_range_xz)
        ang = rng.uniform(0, 2 * np.pi, 3)
        try:
            info = load_thingi10k_object(int(idx), target_size)
        except Exception as e:
            print(f"  skipping idx={idx}: {e}", flush=True)
            continue
        R = euler_to_rotation_matrix(*ang)
        objects.append(MeshObject(
            name=info['name'],
            center=np.array([px, py, pz]),
            rotation=R,
            collision_verts_model=info['collision_verts'],
            collision_faces=info['collision_faces'],
            visual_verts_model=info['visual_verts'],
            visual_faces=info['visual_faces'],
            normalize_factor=info['normalize_factor'],
            inv_mass=1.0,
        ))
    if len(objects) < n_objects:
        raise RuntimeError(
            f"only built {len(objects)}/{n_objects} objects after 30% over-sample; "
            f"bad-mesh rate in Thingi10K exceeded our slack — raise the oversample factor")
    return objects
