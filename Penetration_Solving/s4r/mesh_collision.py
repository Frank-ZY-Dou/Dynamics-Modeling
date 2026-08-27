"""Triangle-mesh collision oracle for non-convex Kubric objects.

Default pair-signed-distance backend is FCL (both during the solve loop in
MeshOracle and in the end-of-run evaluator evaluate_world_collision_meshes).
Trimesh.proximity is kept as a fallback for environments where python-fcl is
unavailable — set MESH_EVAL_USE_FCL=0 to force the trimesh path.
Provides SignedDistanceOracle and PenetrationDepthOracle interfaces.
"""
from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import trimesh

from common import PairPenetration, PairSignedDistance
from box_collision import euler_to_rotation_matrix


# Default for the end-of-run scene evaluator. FCL is ~5-20× faster than the
# trimesh.proximity path on dense non-convex meshes because it reuses a
# pre-built BVH per body and avoids rebuilding bounds_tree per pair-query.
EVAL_USE_FCL = os.environ.get("MESH_EVAL_USE_FCL", "1") != "0"


@dataclass
class MeshObject:
    """A non-convex mesh object for collision queries."""
    name: str
    center: np.ndarray                 # (3,) world-space centroid (= translation, since mesh is centered)
    rotation: np.ndarray               # (3,3) rotation matrix
    collision_verts_model: np.ndarray  # (Vc, 3) collision geometry vertices, centered at origin
    collision_faces: np.ndarray        # (Fc, 3) collision geometry face indices
    visual_verts_model: np.ndarray     # (Vv, 3) visual geometry vertices, centered at origin
    visual_faces: np.ndarray           # (Fv, 3) visual geometry face indices
    normalize_factor: float            # target_size / max_extent
    inv_mass: float = 1.0


@dataclass
class MeshSceneStats:
    pen_pairs: int
    max_penetration: float
    min_signed_distance: float


def _load_json_relaxed(path: Path) -> dict:
    """Load JSON with trailing comma tolerance."""
    txt = path.read_text()
    txt = re.sub(r',\s*}', '}', txt)
    txt = re.sub(r',\s*]', ']', txt)
    return json.loads(txt)


@functools.lru_cache(maxsize=128)
def _load_Kubric_object_cached(obj_dir_str: str, target_size: float) -> dict:
    """Memoised mesh loader keyed on (path, target_size).

    Kubric pool has ~41 unique meshes; at N=30000 scenes each is sampled
    ~730× on average. Without the cache we'd hit the disk + run trimesh's
    OBJ parser that many times — ≈4 min of pre-roll wall at N=30000.
    Returned dicts are READ-ONLY by convention (the numpy arrays inside
    are shared across all bodies that map to the same template); callers
    must not mutate them in place.
    """
    obj_dir = Path(obj_dir_str)
    data = _load_json_relaxed(obj_dir / "data.json")

    bounds = data["kwargs"]["bounds"]
    bbox_min = np.array(bounds[0], dtype=np.float64)
    bbox_max = np.array(bounds[1], dtype=np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min
    max_extent = float(np.max(extent))
    normalize_factor = target_size / max_extent if max_extent > 1e-12 else 1.0

    # Load collision geometry (watertight)
    coll_mesh = trimesh.load(
        str(obj_dir / "collision_geometry.obj"), process=False, force="mesh",
    )
    coll_verts = np.asarray(coll_mesh.vertices, dtype=np.float64) - bbox_center
    coll_faces = np.asarray(coll_mesh.faces, dtype=np.int32)

    # Load visual geometry
    vis_mesh = trimesh.load(
        str(obj_dir / "visual_geometry.obj"), process=False, force="mesh",
    )
    vis_verts = np.asarray(vis_mesh.vertices, dtype=np.float64) - bbox_center
    vis_faces = np.asarray(vis_mesh.faces, dtype=np.int32)

    return dict(
        name=data.get("id", obj_dir.name),
        collision_verts=coll_verts,
        collision_faces=coll_faces,
        visual_verts=vis_verts,
        visual_faces=vis_faces,
        normalize_factor=normalize_factor,
        bbox_center=bbox_center,
    )


def load_Kubric_object(
    obj_dir: str | Path,
    target_size: float = 0.1,
) -> dict:
    """Thin wrapper around the cached loader (Path → str for lru_cache key)."""
    return _load_Kubric_object_cached(str(Path(obj_dir)), float(target_size))


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _aabb_overlap(
    amin: np.ndarray, amax: np.ndarray,
    bmin: np.ndarray, bmax: np.ndarray,
    margin: float = 0.0,
) -> bool:
    """Check if two AABBs overlap (with optional margin)."""
    return bool(np.all(amin - margin <= bmax) and np.all(bmin - margin <= amax))


def _aabb_gap(
    amin: np.ndarray, amax: np.ndarray,
    bmin: np.ndarray, bmax: np.ndarray,
) -> float:
    """Compute minimum gap between two AABBs (0 if overlapping)."""
    gaps = np.maximum(amin - bmax, 0.0) + np.maximum(bmin - amax, 0.0)
    return float(np.linalg.norm(gaps))


def build_world_collision_mesh(
    obj: MeshObject,
    *,
    process: bool = False,
) -> trimesh.Trimesh:
    """Build a world-space collision mesh for a MeshObject."""
    scaled = obj.normalize_factor * obj.collision_verts_model
    rotated = (obj.rotation @ scaled.T).T
    verts_world = rotated + obj.center[None, :]
    return trimesh.Trimesh(vertices=verts_world, faces=obj.collision_faces, process=process)


def build_world_collision_meshes(
    objects: List[MeshObject],
    *,
    process: bool = False,
) -> List[trimesh.Trimesh]:
    """Build world-space collision meshes for a scene."""
    return [build_world_collision_mesh(obj, process=process) for obj in objects]


def _fallback_inside_test(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Fallback inside/outside test when trimesh.contains is unavailable."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    closest, _, face_idx = trimesh.proximity.closest_point(mesh, points)
    normals = mesh.face_normals[face_idx]
    dot = np.sum((points - closest) * normals, axis=1)
    return dot < 0.0


def _contains_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Safely run trimesh.contains with a geometric fallback."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    try:
        return np.asarray(mesh.contains(points), dtype=bool)
    except Exception:
        return _fallback_inside_test(mesh, points)


def _pair_signed_distance_from_samples(
    mesh_i: trimesh.Trimesh,
    sample_j_in_i: np.ndarray,
    mesh_j: trimesh.Trimesh,
    sample_i_in_j: np.ndarray,
    direction_hint: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute signed distance from per-mesh probe points in matching frames."""
    closest_on_i, dist_j_to_i, _ = trimesh.proximity.closest_point(mesh_i, sample_j_in_i)
    closest_on_j, dist_i_to_j, _ = trimesh.proximity.closest_point(mesh_j, sample_i_in_j)

    inside_j_in_i = _contains_points(mesh_i, sample_j_in_i)
    inside_i_in_j = _contains_points(mesh_j, sample_i_in_j)
    has_pen = bool(np.any(inside_j_in_i) or np.any(inside_i_in_j))

    if has_pen:
        depth_a = float(dist_j_to_i[inside_j_in_i].max()) if np.any(inside_j_in_i) else 0.0
        depth_b = float(dist_i_to_j[inside_i_in_j].max()) if np.any(inside_i_in_j) else 0.0

        if depth_a >= depth_b and np.any(inside_j_in_i):
            mask_dists = np.where(inside_j_in_i, dist_j_to_i, -1.0)
            deepest_idx = int(np.argmax(mask_dists))
            raw_dir = closest_on_i[deepest_idx] - sample_j_in_i[deepest_idx]
        else:
            mask_dists = np.where(inside_i_in_j, dist_i_to_j, -1.0)
            deepest_idx = int(np.argmax(mask_dists))
            raw_dir = -(closest_on_j[deepest_idx] - sample_i_in_j[deepest_idx])

        direction = _safe_normalize(raw_dir if np.linalg.norm(raw_dir) >= 1e-12 else direction_hint)
        return -max(depth_a, depth_b), direction

    min_j = float(dist_j_to_i.min())
    min_i = float(dist_i_to_j.min())
    if min_j <= min_i:
        idx = int(np.argmin(dist_j_to_i))
        raw_dir = sample_j_in_i[idx] - closest_on_i[idx]
        direction = _safe_normalize(raw_dir if np.linalg.norm(raw_dir) >= 1e-12 else direction_hint)
        return min_j, direction

    idx = int(np.argmin(dist_i_to_j))
    raw_dir = closest_on_j[idx] - sample_i_in_j[idx]
    direction = _safe_normalize(raw_dir if np.linalg.norm(raw_dir) >= 1e-12 else direction_hint)
    return min_i, direction


def _fcl_pair_signed_distance(fcl_mod, bvh_i, bvh_j) -> float:
    """FCL signed distance between two BVHs whose verts are already in world
    coords. Mirrors MeshOracle._precise_pair_query: distance() when separated,
    collide() when overlapping (for precise penetration depth).
    """
    ident3 = np.eye(3)
    zero3 = np.zeros(3)
    o_i = fcl_mod.CollisionObject(bvh_i, fcl_mod.Transform(ident3, zero3))
    o_j = fcl_mod.CollisionObject(bvh_j, fcl_mod.Transform(ident3, zero3))
    req = fcl_mod.DistanceRequest(enable_nearest_points=False,
                                   enable_signed_distance=True)
    res = fcl_mod.DistanceResult()
    d = fcl_mod.distance(o_i, o_j, req, res)
    if d > 0.0:
        return float(d)
    # Overlap: use collide() to extract exact penetration depth.
    creq = fcl_mod.CollisionRequest(num_max_contacts=16, enable_contact=True)
    cres = fcl_mod.CollisionResult()
    fcl_mod.collide(o_i, o_j, creq, cres)
    if not cres.is_collision or not cres.contacts:
        return 0.0
    c_best = max(cres.contacts, key=lambda c: c.penetration_depth)
    return -float(c_best.penetration_depth)


def _build_fcl_bvhs(meshes: List[trimesh.Trimesh]):
    """Build one FCL BVHModel per world-space trimesh; returns (fcl_mod, [bvh]).
    Returns (None, None) when python-fcl is unavailable.
    """
    try:
        import fcl as _fcl
    except ImportError:
        return None, None
    bvhs = []
    for mesh in meshes:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        m = _fcl.BVHModel()
        m.beginModel(len(verts), len(faces))
        m.addSubModel(verts, faces)
        m.endModel()
        bvhs.append(m)
    return _fcl, bvhs


def evaluate_world_collision_meshes(
    meshes: List[trimesh.Trimesh],
    use_fcl: bool | None = None,
) -> MeshSceneStats:
    """Evaluate penetrating pairs and min signed distance for world-space meshes.

    Default backend is FCL (EVAL_USE_FCL), matching the solver's narrowphase so
    solver/oracle/evaluator agree. Falls back to trimesh.proximity when FCL is
    unavailable or when ``use_fcl=False`` is requested (e.g. for regression
    testing against the legacy numbers).
    """
    if use_fcl is None:
        use_fcl = EVAL_USE_FCL

    n = len(meshes)
    if n < 2:
        return MeshSceneStats(pen_pairs=0, max_penetration=0.0,
                              min_signed_distance=float("inf"))

    fcl_mod, fcl_bvh = (None, None)
    if use_fcl:
        fcl_mod, fcl_bvh = _build_fcl_bvhs(meshes)
    use_fcl = fcl_bvh is not None

    # ── Vectorized AABB overlap + gap check over all N(N-1)/2 pairs ──
    # Was a Python double-loop with 50M `_aabb_overlap` calls at N=10000;
    # cProfile showed this dominated wall time at large N (>50%).
    # New path: build (N,3) mins/maxs, then triu-index broadcast over all
    # pairs once. FCL only fires on truly overlapping pairs (~O(K), K≪N²).
    bounds_arr = np.stack([mesh.bounds for mesh in meshes], axis=0)  # (N, 2, 3)
    mins = bounds_arr[:, 0, :]  # (N, 3)
    maxs = bounds_arr[:, 1, :]  # (N, 3)
    idx_i, idx_j = np.triu_indices(n, k=1)
    overlap_per_axis = (mins[idx_i] <= maxs[idx_j]) & (mins[idx_j] <= maxs[idx_i])
    overlap_mask = np.all(overlap_per_axis, axis=1)
    # Gap (separating distance, ≥ 0) for non-overlapping pairs.
    gap_per_axis = np.maximum(mins[idx_i] - maxs[idx_j], 0.0) \
                 + np.maximum(mins[idx_j] - maxs[idx_i], 0.0)
    gaps = np.linalg.norm(gap_per_axis, axis=1)

    # Default signed distance = AABB gap (≥0). For overlapping pairs, override.
    signed_distances = gaps.copy()

    overlap_pair_idx = np.where(overlap_mask)[0]
    if len(overlap_pair_idx) > 0:
        if use_fcl:
            for k in overlap_pair_idx:
                i, j = int(idx_i[k]), int(idx_j[k])
                signed_distances[k] = _fcl_pair_signed_distance(
                    fcl_mod, fcl_bvh[i], fcl_bvh[j])
        else:
            centroids = [np.asarray(m.centroid, dtype=np.float64) for m in meshes]
            for k in overlap_pair_idx:
                i, j = int(idx_i[k]), int(idx_j[k])
                verts_i = np.asarray(meshes[i].vertices, dtype=np.float64)
                verts_j = np.asarray(meshes[j].vertices, dtype=np.float64)
                sd, _ = _pair_signed_distance_from_samples(
                    meshes[i], verts_j, meshes[j], verts_i,
                    centroids[j] - centroids[i],
                )
                signed_distances[k] = sd

    pen_mask = signed_distances < 0.0
    pen_pairs = int(pen_mask.sum())
    max_penetration = float(-signed_distances[pen_mask].min()) if pen_pairs else 0.0
    min_signed_distance = float(signed_distances.min())

    return MeshSceneStats(
        pen_pairs=pen_pairs,
        max_penetration=max_penetration,
        min_signed_distance=min_signed_distance,
    )


def evaluate_mesh_object_scene(
    objects: List[MeshObject],
    use_fcl: bool | None = None,
) -> MeshSceneStats:
    """Evaluate penetration stats for a MeshObject scene in its final poses."""
    return evaluate_world_collision_meshes(
        build_world_collision_meshes(objects, process=False),
        use_fcl=use_fcl,
    )


class MeshOracle:
    """Collision oracle for oriented triangle meshes using trimesh proximity.

    Uses collision_geometry (watertight) for signed distance queries.
    Implements both SignedDistanceOracle and PenetrationDepthOracle protocols.
    """

    def __init__(self, objects: List[MeshObject], use_fcl: bool = True):
        self.objects = objects
        self.n = len(objects)
        self.use_fcl = use_fcl

        # Pre-compute: rotated + scaled collision vertices and trimesh objects
        # Both S4R and baselines use collision_geometry (watertight, required by IPC)
        self._rotated_verts: List[np.ndarray] = []
        self._meshes: List[trimesh.Trimesh] = []
        self._local_aabb_min: List[np.ndarray] = []
        self._local_aabb_max: List[np.ndarray] = []

        for obj in objects:
            scaled = obj.normalize_factor * obj.collision_verts_model
            rotated = (obj.rotation @ scaled.T).T
            self._rotated_verts.append(rotated)
            mesh = trimesh.Trimesh(vertices=rotated, faces=obj.collision_faces, process=True)
            self._meshes.append(mesh)
            self._local_aabb_min.append(rotated.min(axis=0))
            self._local_aabb_max.append(rotated.max(axis=0))

        # FCL BVH models: one per body, built once (scale=1, at origin; the
        # pose is applied via fcl.Transform in _precise_pair_query).
        self._fcl_bvh: List = []
        if self.use_fcl:
            try:
                import fcl as _fcl  # noqa: F401
                self._fcl = _fcl
                for obj, verts_rot in zip(objects, self._rotated_verts):
                    m = _fcl.BVHModel()
                    faces = np.asarray(obj.collision_faces, dtype=np.int32)
                    m.beginModel(len(verts_rot), len(faces))
                    m.addSubModel(verts_rot.astype(np.float64), faces)
                    m.endModel()
                    self._fcl_bvh.append(m)
            except ImportError:
                # FCL unavailable: fall back to trimesh + _contains_points.
                self.use_fcl = False
                self._fcl = None
        else:
            self._fcl = None

    def _broadphase(self, centers: np.ndarray, margin: float):
        """Vectorized AABB broadphase over all N(N-1)/2 pairs,
        replacing the per-pair Python double loop (~17 us/pair, ~80% of the
        call). Same overlap/gap semantics as _aabb_overlap / _aabb_gap.
        Returns (idx_i, idx_j, gaps, overlap_mask)."""
        if getattr(self, "_aabb_lo_arr", None) is None:
            self._aabb_lo_arr = np.stack(self._local_aabb_min, axis=0)  # (N,3)
            self._aabb_hi_arr = np.stack(self._local_aabb_max, axis=0)
        c = np.asarray(centers, dtype=np.float64)
        lo = self._aabb_lo_arr + c
        hi = self._aabb_hi_arr + c
        idx_i, idx_j = np.triu_indices(self.n, k=1)
        overlap = np.all((lo[idx_i] - margin <= hi[idx_j]) &
                         (lo[idx_j] - margin <= hi[idx_i]), axis=1)
        gap_axis = np.maximum(lo[idx_i] - hi[idx_j], 0.0) \
                 + np.maximum(lo[idx_j] - hi[idx_i], 0.0)
        gaps = np.linalg.norm(gap_axis, axis=1)
        return idx_i, idx_j, gaps, overlap

    def signed_distances(self, centers: np.ndarray) -> List[PairSignedDistance]:
        margin = 0.05  # generous margin for broad phase
        idx_i, idx_j, gaps, overlap = self._broadphase(centers, margin)
        c = np.asarray(centers, dtype=np.float64)
        dirs = c[idx_j] - c[idx_i]
        nrm = np.linalg.norm(dirs, axis=1)
        pairs: List[PairSignedDistance] = []
        for k in range(len(idx_i)):
            i, j = int(idx_i[k]), int(idx_j[k])
            if overlap[k]:
                sd, direction = self._precise_pair_query(i, j, c[i], c[j])
                pairs.append(PairSignedDistance(i, j, float(sd), direction))
            else:
                d = dirs[k] / nrm[k] if nrm[k] > 1e-12 else np.array([1.0, 0.0, 0.0])
                pairs.append(PairSignedDistance(i, j, float(gaps[k]), d))
        return pairs

    def near_signed_distances(self, centers: np.ndarray,
                              activation: float) -> List[PairSignedDistance]:
        """Signed distances for pairs that can be below `activation` only.

        Note: the margin modes previously
        went through signed_distances(), whose Python loop materializes ALL
        N(N-1)/2 pairs (~0.29 s/call at N=500) — a per-iteration overhead the
        default penetrations() path never pays, i.e. an artificial timing
        handicap on the baselines' margin arms. This fast path enumerates
        only broadphase-overlapping pairs (the 0.05 AABB margin is a lower
        bound on true distance, so no pair with sd < activation <= 0.05 can
        be missed) — same O(K) structure as penetrations().
        """
        assert activation <= 0.05, (
            "near_signed_distances requires activation <= the 0.05 broadphase "
            "margin (AABB gap only lower-bounds the true distance)")
        idx_i, idx_j, gaps, overlap = self._broadphase(centers, 0.05)
        c = np.asarray(centers, dtype=np.float64)
        pairs: List[PairSignedDistance] = []
        for k in np.nonzero(overlap)[0]:
            i, j = int(idx_i[k]), int(idx_j[k])
            sd, direction = self._precise_pair_query(i, j, c[i], c[j])
            if sd < activation:
                pairs.append(PairSignedDistance(i, j, float(sd), direction))
        return pairs

    def penetrations(self, centers: np.ndarray) -> List[PairPenetration]:
                # precise query on those alone (O(K)); identical result to filtering
        # signed_distances()<0, without building the full N^2 pair list.
        margin = 0.05
        idx_i, idx_j, gaps, overlap = self._broadphase(centers, margin)
        c = np.asarray(centers, dtype=np.float64)
        out: List[PairPenetration] = []
        for k in np.nonzero(overlap)[0]:
            i, j = int(idx_i[k]), int(idx_j[k])
            sd, direction = self._precise_pair_query(i, j, c[i], c[j])
            if sd < 0.0:
                out.append(PairPenetration(i=i, j=j, depth=float(-sd),
                                           direction_ij=direction))
        return out

    def _precise_pair_query(
        self, i: int, j: int,
        ci: np.ndarray, cj: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute signed distance between mesh i and mesh j.

        By default uses FCL BVH distance() + collide() (same primitive as
        \\name's fcl contact backend, so solver/oracle/evaluator agree).
        Falls back to trimesh.closest_point + _contains_points (ray-cast
        inside test) if FCL is unavailable.

        Returns (signed_distance, direction_ij) where:
          signed_distance < 0 means penetration
          signed_distance > 0 means separation
          direction_ij points from i toward j
        """
        if self.use_fcl:
            fcl = self._fcl
            ident3 = np.eye(3)
            o_i = fcl.CollisionObject(self._fcl_bvh[i],
                                      fcl.Transform(ident3, np.asarray(ci, dtype=np.float64)))
            o_j = fcl.CollisionObject(self._fcl_bvh[j],
                                      fcl.Transform(ident3, np.asarray(cj, dtype=np.float64)))
            req = fcl.DistanceRequest(enable_nearest_points=True,
                                      enable_signed_distance=True)
            res = fcl.DistanceResult()
            d = fcl.distance(o_i, o_j, req, res)
            if d > 0.0:
                cp_i = np.asarray(res.nearest_points[0], dtype=np.float64)
                cp_j = np.asarray(res.nearest_points[1], dtype=np.float64)
                raw = cp_j - cp_i
                return float(d), _safe_normalize(raw if np.linalg.norm(raw) > 1e-12
                                                 else cj - ci)
            # Overlap: get penetration info.
            creq = fcl.CollisionRequest(num_max_contacts=16, enable_contact=True)
            cres = fcl.CollisionResult()
            fcl.collide(o_i, o_j, creq, cres)
            if not cres.is_collision or not cres.contacts:
                # Distance=0 but not colliding: touching; treat as sd=0.
                return 0.0, _safe_normalize(cj - ci)
            c_best = max(cres.contacts, key=lambda c: c.penetration_depth)
            depth = float(c_best.penetration_depth)
            raw = np.asarray(c_best.normal, dtype=np.float64)
            if np.dot(raw, cj - ci) < 0.0:
                raw = -raw
            return -depth, _safe_normalize(raw)

        # ---------- Fallback: trimesh + ray-cast inside test ----------
        mesh_i = self._meshes[i]
        mesh_j = self._meshes[j]
        verts_i = self._rotated_verts[i]
        verts_j = self._rotated_verts[j]
        offset = cj - ci
        pts_j_in_i = verts_j + offset
        pts_i_in_j = verts_i - offset

        closest_on_i, dist_j_to_i, _ = trimesh.proximity.closest_point(mesh_i, pts_j_in_i)
        closest_on_j, dist_i_to_j, _ = trimesh.proximity.closest_point(mesh_j, pts_i_in_j)

        # Inside test via ray-cast (same helper the evaluator uses).
        inside_j_in_i = _contains_points(mesh_i, pts_j_in_i)
        inside_i_in_j = _contains_points(mesh_j, pts_i_in_j)

        has_pen = bool(np.any(inside_j_in_i) or np.any(inside_i_in_j))
        if has_pen:
            depth_a = float(dist_j_to_i[inside_j_in_i].max()) if np.any(inside_j_in_i) else 0.0
            depth_b = float(dist_i_to_j[inside_i_in_j].max()) if np.any(inside_i_in_j) else 0.0
            if depth_a >= depth_b and np.any(inside_j_in_i):
                mask = np.where(inside_j_in_i, dist_j_to_i, -1.0)
                k = int(np.argmax(mask))
                raw = closest_on_i[k] - pts_j_in_i[k]
            else:
                mask = np.where(inside_i_in_j, dist_i_to_j, -1.0)
                k = int(np.argmax(mask))
                raw = -(closest_on_j[k] - pts_i_in_j[k])
            return -max(depth_a, depth_b), _safe_normalize(raw)

        min_j = float(dist_j_to_i.min())
        min_i = float(dist_i_to_j.min())
        if min_j <= min_i:
            k = int(np.argmin(dist_j_to_i))
            raw = pts_j_in_i[k] - closest_on_i[k]
            return min_j, _safe_normalize(raw)
        k = int(np.argmin(dist_i_to_j))
        raw = closest_on_j[k] - pts_i_in_j[k]
        return min_i, _safe_normalize(raw)


def scan_eligible_objects(
    kubric_dir: str | Path,
    max_verts: int = 5000,
) -> List[Path]:
    """Scan Kubric directory and return sorted list of eligible object dirs."""
    kubric_dir = Path(kubric_dir)
    eligible: List[Path] = []

    for d in sorted(kubric_dir.iterdir()):
        if not d.is_dir():
            continue
        data_path = d / "data.json"
        coll_path = d / "collision_geometry.obj"
        if not (data_path.exists() and coll_path.exists()):
            continue
        try:
            data = _load_json_relaxed(data_path)
            nv = data.get("metadata", {}).get("nr_vertices", 0)
            if nv < 500 or nv > max_verts:
                continue
            eligible.append(d)
        except Exception:
            continue

    return eligible


def generate_kubric_scene(
    n_objects: int,
    target_size: float,
    spawn_range_xz: float,
    spawn_range_y: float,
    seed: int,
    kubric_dir: str | Path,
    max_verts: int = 5000,
    allow_repeat: bool = True,
) -> List[MeshObject]:
    """Generate a scene of randomly placed Kubric objects.

    Returns List[MeshObject] with random positions and rotations.
    Same seed + call order can be replicated in run_s4r_Kubric.py.

    When n_objects > number of eligible objects, sampling falls back to
    replace=True so we can build large benchmark scenes (200, 500, ...).
    """
    eligible = scan_eligible_objects(kubric_dir, max_verts)
    if len(eligible) == 0:
        raise ValueError("No eligible Kubric objects found.")

    rng = np.random.RandomState(seed)
    replace = allow_repeat and (n_objects > len(eligible))
    if not replace and n_objects > len(eligible):
        raise ValueError(
            f"Only {len(eligible)} eligible objects (need {n_objects}). "
            f"Try allow_repeat=True or increase max_verts."
        )
    selected_indices = rng.choice(len(eligible), n_objects, replace=replace)
    selected_dirs = [eligible[idx] for idx in selected_indices]

    objects: List[MeshObject] = []
    for i, obj_dir in enumerate(selected_dirs):
        info = load_Kubric_object(obj_dir, target_size)

        # Random position (same call order as box scene)
        px = rng.uniform(-spawn_range_xz, spawn_range_xz)
        py = rng.uniform(-spawn_range_y, spawn_range_y)
        pz = rng.uniform(-spawn_range_xz, spawn_range_xz)
        pos = np.array([px, py, pz])

        # Random rotation
        rot_angles = rng.uniform(0, 2 * np.pi, 3)
        R = euler_to_rotation_matrix(*rot_angles)

        # Center = pos (mesh is centered at origin)
        objects.append(MeshObject(
            name=info["name"],
            center=pos.copy(),
            rotation=R,
            collision_verts_model=info["collision_verts"],
            collision_faces=info["collision_faces"],
            visual_verts_model=info["visual_verts"],
            visual_faces=info["visual_faces"],
            normalize_factor=info["normalize_factor"],
            inv_mass=1.0,
        ))

    return objects
