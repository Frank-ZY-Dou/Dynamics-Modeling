"""S4R-Warp contact oracle v3 — template-shared wp.Mesh.

Builds on v2 (edge-sampled MTV, no FCL). Key change: each unique mesh
template gets ONE wp.Mesh (with a single BVH built once on the
template's local-frame verts). Per body we keep only (template_id, R,
R_T, center_u). The kernel transforms each query point from u-space
into the target body's template-local frame before traversing the BVH.

Wins at large N:
- Construction: 41 wp.Mesh built once (~0.2s) vs 30k built per scene
  (~150s). Saves ~5 min at N=30000.
- Per-step: no per-body `wp.Mesh.points.assign + refit` loop (was
  ~150s per scene at N=30000). BVHs are static; only the small per-body
  pose arrays update each step.
- GPU memory: O(unique templates) wp.Mesh objects instead of O(N).

Public API identical to V1/V2: ``find_contacts(s, ds, centers, rotations)``
returns the same 8-tuple list.
"""
from __future__ import annotations
import hashlib
from typing import List

import numpy as np
import warp as wp

wp.init()
_DEVICE = "cuda:0"
EDGE_SAMPLES = 5


def _template_key(v_model: np.ndarray, f: np.ndarray, nf: float) -> str:
    """Cheap content hash used to detect duplicate mesh templates across
    bodies. We include nf because two callers could pass the same
    (verts, faces) but different normalize_factor; treat those as
    distinct templates."""
    h = hashlib.blake2b(digest_size=16)
    h.update(v_model.tobytes())
    h.update(f.tobytes())
    h.update(np.float64(nf).tobytes())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Kernel A: vertex SDF on per-template BVH with per-body pose transform.
# Per (pair_k, vert_in_tpl_j) thread:
#   1. Read template_j vert v_local (template-local frame, nf-scaled).
#   2. Transform j's vert to u-space: q_u = R_j @ v_local + center_j_u.
#   3. Transform q_u into body i's template-local frame:
#        p_i_local = R_i_T @ (q_u - center_i_u).
#   4. Query body i's template BVH at p_i_local.
#   5. atomic_min the signed distance; record cp/normal in u-space.
# ─────────────────────────────────────────────────────────────────────
@wp.kernel
def _pair_min_distance_kernel_v3(
    template_mesh_ids: wp.array(dtype=wp.uint64),
    body_template_id: wp.array(dtype=wp.int32),
    body_R: wp.array(dtype=wp.mat33),
    body_R_T: wp.array(dtype=wp.mat33),
    body_center_u: wp.array(dtype=wp.vec3),
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    template_vert_offset: wp.array(dtype=wp.int32),
    template_vert_count: wp.array(dtype=wp.int32),
    template_verts_flat: wp.array(dtype=wp.vec3),
    n_pairs: wp.int32,
    max_verts_per_template: wp.int32,
    max_dist_u: wp.float32,
    # ── deterministic argmin ──
    launch_offset: wp.int32,
    quant_scale: wp.float32,
    max_dist_off: wp.float32,
    tag_shift: wp.int32,
    q_max: wp.int64,
    write_phase: wp.int32,
    pair_key: wp.array(dtype=wp.int64),
    pair_sd: wp.array(dtype=wp.float32),
    pair_nx: wp.array(dtype=wp.float32),
    pair_ny: wp.array(dtype=wp.float32),
    pair_nz: wp.array(dtype=wp.float32),
    pair_cp_ax: wp.array(dtype=wp.float32),
    pair_cp_ay: wp.array(dtype=wp.float32),
    pair_cp_az: wp.array(dtype=wp.float32),
    pair_cp_bx: wp.array(dtype=wp.float32),
    pair_cp_by: wp.array(dtype=wp.float32),
    pair_cp_bz: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    k = tid // max_verts_per_template
    v_idx = tid % max_verts_per_template
    if k >= n_pairs:
        return
    j = pair_j[k]
    j_tpl = body_template_id[j]
    v_off = template_vert_offset[j_tpl]
    v_cnt = template_vert_count[j_tpl]
    if v_idx >= v_cnt:
        return

    v_local_j = template_verts_flat[v_off + v_idx]
    q_u = body_R[j] @ v_local_j + body_center_u[j]

    i_body = pair_i[k]
    p_i_local = body_R_T[i_body] @ (q_u - body_center_u[i_body])

    i_tpl = body_template_id[i_body]
    mesh_i = template_mesh_ids[i_tpl]
    query = wp.mesh_query_point_sign_winding_number(
        mesh_i, p_i_local, max_dist_u, 2.0,
    )
    if not query.result:
        return
    cp_local = wp.mesh_eval_position(mesh_i, query.face, query.u, query.v)
    sd = wp.length(p_i_local - cp_local) * query.sign

    # Packed deterministic key: quantized distance (high bits, order-preserving)
    # | per-feature tag (low bits, unique across the 4 launches). atomic_min on
    # this int64 picks the SAME (min-distance, min-tag) feature regardless of
    # thread order — fixing the racy "last writer wins" normal write (§8.4).
    qd = wp.int64((sd + max_dist_off) * quant_scale)
    if qd < wp.int64(0):
        qd = wp.int64(0)
    if qd > q_max:
        qd = q_max
    tag = wp.int64(launch_offset + v_idx)
    key = (qd << wp.int64(tag_shift)) | tag

    if write_phase == 0:
        # Reduce phase: establish the true min distance and the winning key.
        wp.atomic_min(pair_sd, k, sd)
        wp.atomic_min(pair_key, k, key)
        return

    # Write phase: exactly the unique argmin feature writes companion data.
    if key != pair_key[k]:
        return
    cp_u = body_R[i_body] @ cp_local + body_center_u[i_body]
    n_u = q_u - cp_u
    ln = wp.length(n_u)
    if ln > 1e-12:
        inv = 1.0 / ln
        pair_nx[k] = n_u.x * inv
        pair_ny[k] = n_u.y * inv
        pair_nz[k] = n_u.z * inv
    else:
        pair_nx[k] = 0.0
        pair_ny[k] = 0.0
        pair_nz[k] = 0.0
    pair_cp_ax[k] = cp_u.x
    pair_cp_ay[k] = cp_u.y
    pair_cp_az[k] = cp_u.z
    pair_cp_bx[k] = q_u.x
    pair_cp_by[k] = q_u.y
    pair_cp_bz[k] = q_u.z


# ─────────────────────────────────────────────────────────────────────
# Kernel A': edge-sampled SDF on per-template BVH (same transform pattern).
# ─────────────────────────────────────────────────────────────────────
@wp.kernel
def _pair_min_distance_edge_kernel_v3(
    template_mesh_ids: wp.array(dtype=wp.uint64),
    body_template_id: wp.array(dtype=wp.int32),
    body_R: wp.array(dtype=wp.mat33),
    body_R_T: wp.array(dtype=wp.mat33),
    body_center_u: wp.array(dtype=wp.vec3),
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    template_vert_offset: wp.array(dtype=wp.int32),
    template_edge_offset: wp.array(dtype=wp.int32),
    template_edge_count: wp.array(dtype=wp.int32),
    template_edges_v0_flat: wp.array(dtype=wp.int32),
    template_edges_v1_flat: wp.array(dtype=wp.int32),
    template_verts_flat: wp.array(dtype=wp.vec3),
    n_pairs: wp.int32,
    max_edges_per_template: wp.int32,
    n_samples: wp.int32,
    max_dist_u: wp.float32,
    # ── deterministic argmin ──
    launch_offset: wp.int32,
    quant_scale: wp.float32,
    max_dist_off: wp.float32,
    tag_shift: wp.int32,
    q_max: wp.int64,
    write_phase: wp.int32,
    pair_key: wp.array(dtype=wp.int64),
    pair_sd: wp.array(dtype=wp.float32),
    pair_nx: wp.array(dtype=wp.float32),
    pair_ny: wp.array(dtype=wp.float32),
    pair_nz: wp.array(dtype=wp.float32),
    pair_cp_ax: wp.array(dtype=wp.float32),
    pair_cp_ay: wp.array(dtype=wp.float32),
    pair_cp_az: wp.array(dtype=wp.float32),
    pair_cp_bx: wp.array(dtype=wp.float32),
    pair_cp_by: wp.array(dtype=wp.float32),
    pair_cp_bz: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    samples_per_edge = n_samples
    edges_per_pair = max_edges_per_template
    threads_per_pair = edges_per_pair * samples_per_edge

    k = tid // threads_per_pair
    rem = tid % threads_per_pair
    e_local = rem // samples_per_edge
    s_local = rem % samples_per_edge
    if k >= n_pairs:
        return

    j = pair_j[k]
    j_tpl = body_template_id[j]
    e_off = template_edge_offset[j_tpl]
    e_cnt = template_edge_count[j_tpl]
    if e_local >= e_cnt:
        return

    ev0 = template_edges_v0_flat[e_off + e_local]
    ev1 = template_edges_v1_flat[e_off + e_local]
    v_off = template_vert_offset[j_tpl]
    p0_local = template_verts_flat[v_off + ev0]
    p1_local = template_verts_flat[v_off + ev1]

    t = wp.float32(s_local + 1) / wp.float32(samples_per_edge + 1)
    p_sample_j_local = p0_local * (1.0 - t) + p1_local * t

    # Body j's sample point in u-space.
    q_u = body_R[j] @ p_sample_j_local + body_center_u[j]

    # Transform to body i's template-local frame.
    i_body = pair_i[k]
    p_i_local = body_R_T[i_body] @ (q_u - body_center_u[i_body])

    i_tpl = body_template_id[i_body]
    mesh_i = template_mesh_ids[i_tpl]
    query = wp.mesh_query_point_sign_winding_number(
        mesh_i, p_i_local, max_dist_u, 2.0,
    )
    if not query.result:
        return
    cp_local = wp.mesh_eval_position(mesh_i, query.face, query.u, query.v)
    sd = wp.length(p_i_local - cp_local) * query.sign

    # Same deterministic-argmin scheme as the vertex kernel. The per-pair
    # feature tag here is (edge, sample) flattened, offset by launch_offset so
    # it never collides with the vertex launches sharing pair_key.
    qd = wp.int64((sd + max_dist_off) * quant_scale)
    if qd < wp.int64(0):
        qd = wp.int64(0)
    if qd > q_max:
        qd = q_max
    tag = wp.int64(launch_offset + e_local * samples_per_edge + s_local)
    key = (qd << wp.int64(tag_shift)) | tag

    if write_phase == 0:
        wp.atomic_min(pair_sd, k, sd)
        wp.atomic_min(pair_key, k, key)
        return

    if key != pair_key[k]:
        return
    cp_u = body_R[i_body] @ cp_local + body_center_u[i_body]
    n_u = q_u - cp_u
    ln = wp.length(n_u)
    if ln > 1e-12:
        inv = 1.0 / ln
        pair_nx[k] = n_u.x * inv
        pair_ny[k] = n_u.y * inv
        pair_nz[k] = n_u.z * inv
    else:
        pair_nx[k] = 0.0
        pair_ny[k] = 0.0
        pair_nz[k] = 0.0
    pair_cp_ax[k] = cp_u.x
    pair_cp_ay[k] = cp_u.y
    pair_cp_az[k] = cp_u.z
    pair_cp_bx[k] = q_u.x
    pair_cp_by[k] = q_u.y
    pair_cp_bz[k] = q_u.z


# ─────────────────────────────────────────────────────────────────────
# Extent kernel: body-local frame projection onto contact normal.
# Identical to v2; uses template verts since they're shared per body.
# ─────────────────────────────────────────────────────────────────────
@wp.kernel
def _extent_kernel_v3(
    body_template_id: wp.array(dtype=wp.int32),
    template_vert_offset: wp.array(dtype=wp.int32),
    template_vert_count: wp.array(dtype=wp.int32),
    template_verts_flat: wp.array(dtype=wp.vec3),
    body_rot_T: wp.array(dtype=wp.mat33),
    pair_b: wp.array(dtype=wp.int32),
    pair_nx: wp.array(dtype=wp.float32),
    pair_ny: wp.array(dtype=wp.float32),
    pair_nz: wp.array(dtype=wp.float32),
    pair_sign: wp.float32,
    n_pairs: wp.int32,
    max_verts_per_template: wp.int32,
    pair_min_proj: wp.array(dtype=wp.float32),
    pair_max_proj: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    k = tid // max_verts_per_template
    v_local = tid % max_verts_per_template
    if k >= n_pairs:
        return
    b = pair_b[k]
    tpl = body_template_id[b]
    v_off = template_vert_offset[tpl]
    v_cnt = template_vert_count[tpl]
    if v_local >= v_cnt:
        return
    v_in_local = template_verts_flat[v_off + v_local]
    R_T = body_rot_T[b]
    n_local = R_T @ wp.vec3(pair_nx[k], pair_ny[k], pair_nz[k]) * pair_sign
    proj = wp.dot(v_in_local, n_local)
    wp.atomic_min(pair_min_proj, k, proj)
    wp.atomic_max(pair_max_proj, k, proj)


# ─────────────────────────────────────────────────────────────────────
# Oracle class.
# ─────────────────────────────────────────────────────────────────────
class S4RWarpContactOracleV3:
    """Template-shared variant of the FCL-free Warp contact oracle.

    Drop-in for v1/v2 via the same ``find_contacts`` signature."""

    DEBUG = False

    def __init__(self, mesh_objects, d_hat: float):
        self.N = len(mesh_objects)
        self.d_hat = float(d_hat)

        # ── Detect unique templates ──
        key_to_tpl_id: dict[str, int] = {}
        self.tpl_v_local: List[np.ndarray] = []
        self.tpl_faces: List[np.ndarray] = []
        self.tpl_max_extent: List[float] = []
        body_template_id_host = np.empty(self.N, dtype=np.int32)
        self.nfs: List[float] = []  # per-body nf (== per-template nf)
        self.max_extents: List[float] = []  # per-body max_extent (== template's)

        for i, m in enumerate(mesh_objects):
            nf = float(getattr(m, "normalize_factor", 1.0))
            v_model = np.asarray(m.collision_verts_model, dtype=np.float64)
            f = np.asarray(m.collision_faces, dtype=np.int32)
            self.nfs.append(nf)
            max_ext = float(nf * np.max(np.linalg.norm(v_model, axis=1)))
            self.max_extents.append(max_ext)

            key = _template_key(v_model, f, nf)
            if key not in key_to_tpl_id:
                key_to_tpl_id[key] = len(self.tpl_v_local)
                self.tpl_v_local.append(nf * v_model)  # template local frame
                self.tpl_faces.append(f)
                self.tpl_max_extent.append(max_ext)
            body_template_id_host[i] = key_to_tpl_id[key]

        self.n_templates = len(self.tpl_v_local)
        self.max_extents_arr = np.asarray(self.max_extents, dtype=np.float64)

        # ── Build per-template wp.Mesh (static BVH) + edges ──
        self.template_meshes: List[wp.Mesh] = []
        tpl_vert_offsets = [0]
        tpl_vert_counts = []
        tpl_edge_offsets = [0]
        tpl_edge_counts = []
        all_tpl_verts: List[np.ndarray] = []
        all_tpl_edges_v0: List[np.ndarray] = []
        all_tpl_edges_v1: List[np.ndarray] = []
        local_aabb_lo = np.empty((self.n_templates, 3))
        local_aabb_hi = np.empty((self.n_templates, 3))

        for t, (v_local, f) in enumerate(zip(self.tpl_v_local, self.tpl_faces)):
            points = wp.array(v_local.astype(np.float32),
                              dtype=wp.vec3, device=_DEVICE)
            indices = wp.array(f.flatten(), dtype=int, device=_DEVICE)
            self.template_meshes.append(wp.Mesh(points=points, indices=indices))

            tpl_vert_offsets.append(tpl_vert_offsets[-1] + len(v_local))
            tpl_vert_counts.append(len(v_local))
            local_aabb_lo[t] = v_local.min(axis=0)
            local_aabb_hi[t] = v_local.max(axis=0)
            all_tpl_verts.append(v_local)

            e_v0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
            e_v1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
            all_tpl_edges_v0.append(e_v0.astype(np.int32))
            all_tpl_edges_v1.append(e_v1.astype(np.int32))
            tpl_edge_offsets.append(tpl_edge_offsets[-1] + len(e_v0))
            tpl_edge_counts.append(len(e_v0))

        # ── Host & device template constants ──
        self.body_template_id_host = body_template_id_host
        self.body_template_id_dev = wp.array(body_template_id_host,
                                              dtype=wp.int32, device=_DEVICE)
        self.tpl_vert_offset_host = np.asarray(tpl_vert_offsets[:-1], dtype=np.int32)
        self.tpl_vert_count_host = np.asarray(tpl_vert_counts, dtype=np.int32)
        self.max_verts_per_template = int(self.tpl_vert_count_host.max())
        self.tpl_edge_offset_host = np.asarray(tpl_edge_offsets[:-1], dtype=np.int32)
        self.tpl_edge_count_host = np.asarray(tpl_edge_counts, dtype=np.int32)
        self.max_edges_per_template = int(self.tpl_edge_count_host.max())
        self.local_aabb_lo = local_aabb_lo
        self.local_aabb_hi = local_aabb_hi
        # Per-body local AABBs (template AABB indexed by template_id).
        self.body_aabb_lo_local = local_aabb_lo[body_template_id_host]
        self.body_aabb_hi_local = local_aabb_hi[body_template_id_host]

        self.tpl_vert_offset_dev = wp.array(self.tpl_vert_offset_host,
                                             dtype=wp.int32, device=_DEVICE)
        self.tpl_vert_count_dev = wp.array(self.tpl_vert_count_host,
                                            dtype=wp.int32, device=_DEVICE)
        self.tpl_edge_offset_dev = wp.array(self.tpl_edge_offset_host,
                                             dtype=wp.int32, device=_DEVICE)
        self.tpl_edge_count_dev = wp.array(self.tpl_edge_count_host,
                                            dtype=wp.int32, device=_DEVICE)
        tpl_verts_flat = np.concatenate(all_tpl_verts, axis=0).astype(np.float32)
        self.tpl_verts_flat_dev = wp.array(tpl_verts_flat, dtype=wp.vec3,
                                            device=_DEVICE)
        edges_v0_flat = np.concatenate(all_tpl_edges_v0).astype(np.int32)
        edges_v1_flat = np.concatenate(all_tpl_edges_v1).astype(np.int32)
        self.tpl_edges_v0_dev = wp.array(edges_v0_flat, dtype=wp.int32,
                                          device=_DEVICE)
        self.tpl_edges_v1_dev = wp.array(edges_v1_flat, dtype=wp.int32,
                                          device=_DEVICE)

        tpl_ids = np.asarray([m.id for m in self.template_meshes],
                              dtype=np.uint64)
        self.template_mesh_ids_dev = wp.array(tpl_ids, dtype=wp.uint64,
                                                device=_DEVICE)

        # Per-body pose buffers (updated each find_contacts call).
        self.body_R_dev = wp.zeros(self.N, dtype=wp.mat33, device=_DEVICE)
        self.body_R_T_dev = wp.zeros(self.N, dtype=wp.mat33, device=_DEVICE)
        self.body_center_u_dev = wp.zeros(self.N, dtype=wp.vec3, device=_DEVICE)

    def find_contacts(self, s: float, ds: float,
                      centers: np.ndarray, rotations: List[np.ndarray]):
        inv_s = 1.0 / s

        # ── Update per-body pose (small N-sized buffers; no per-body BVH refit!) ──
        R_host = np.empty((self.N, 3, 3), dtype=np.float32)
        R_T_host = np.empty((self.N, 3, 3), dtype=np.float32)
        center_u_host = np.empty((self.N, 3), dtype=np.float32)
        world_aabb_lo = np.empty((self.N, 3))
        world_aabb_hi = np.empty((self.N, 3))
        for i in range(self.N):
            R = np.asarray(rotations[i], dtype=np.float64)
            R_host[i] = R.astype(np.float32)
            R_T_host[i] = R.T.astype(np.float32)
            center_u = centers[i] * inv_s
            center_u_host[i] = center_u.astype(np.float32)
            # AABB in u-space: template AABB rotated by R, translated by center_u.
            lo_local = self.body_aabb_lo_local[i]
            hi_local = self.body_aabb_hi_local[i]
            # Conservative world AABB: enumerate 8 rotated corners.
            corners = np.array([
                [lo_local[0], lo_local[1], lo_local[2]],
                [hi_local[0], lo_local[1], lo_local[2]],
                [lo_local[0], hi_local[1], lo_local[2]],
                [hi_local[0], hi_local[1], lo_local[2]],
                [lo_local[0], lo_local[1], hi_local[2]],
                [hi_local[0], lo_local[1], hi_local[2]],
                [lo_local[0], hi_local[1], hi_local[2]],
                [hi_local[0], hi_local[1], hi_local[2]],
            ])
            rot_corners = (R @ corners.T).T + center_u
            world_aabb_lo[i] = rot_corners.min(axis=0)
            world_aabb_hi[i] = rot_corners.max(axis=0)
        self.body_R_dev.assign(R_host.reshape(-1))
        self.body_R_T_dev.assign(R_T_host.reshape(-1))
        self.body_center_u_dev.assign(center_u_host.reshape(-1))

        # ── Broad-phase (CPU, vectorized) — identical to v2 ──
        N = self.N
        idx_i, idx_j = np.triu_indices(N, k=1)
        margin_world = ds * max(self.nfs) * 0.2
        margin_u = margin_world * inv_s
        ovlp = np.all(
            (world_aabb_lo[idx_i] - margin_u <= world_aabb_hi[idx_j])
            & (world_aabb_lo[idx_j] - margin_u <= world_aabb_hi[idx_i]),
            axis=1,
        )
        c_i = centers[idx_i]
        c_j = centers[idx_j]
        d_ij_now = np.linalg.norm(c_i - c_j, axis=1)
        denom = self.max_extents_arr[idx_i] + self.max_extents_arr[idx_j] + 1e-12
        s_contact_now = np.maximum(0.0, (d_ij_now - self.d_hat) / denom)
        ovlp &= (s + ds >= s_contact_now * 0.9)
        idx_i = idx_i[ovlp]; idx_j = idx_j[ovlp]
        K = len(idx_i)
        if K == 0:
            return []

        # ── Narrow phase: vertex + edge kernels (both directions) ──
        p_i = np.ascontiguousarray(idx_i, dtype=np.int32)
        p_j = np.ascontiguousarray(idx_j, dtype=np.int32)
        p_i_dev = wp.array(p_i, dtype=wp.int32, device=_DEVICE)
        p_j_dev = wp.array(p_j, dtype=wp.int32, device=_DEVICE)
        max_ext_world = float(self.max_extents_arr.max())
        max_dist_world = max(2.0 * max_ext_world,
                              ds * 2.0 * max_ext_world + self.d_hat)
        max_dist_u = float(max_dist_world * inv_s)

        sd = wp.array(np.full(K, max_dist_u, dtype=np.float32),
                      dtype=wp.float32, device=_DEVICE)
        nx = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        ny = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        nz = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpax = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpay = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpaz = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpbx = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpby = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpbz = wp.zeros(K, dtype=wp.float32, device=_DEVICE)

        # ── Deterministic-argmin bookkeeping (§8.5) ──
        # All four narrow-phase launches reduce into a single packed int64 key
        # per pair; a second (write) pass lets only the unique winning feature
        # write the contact normal/closest-points. This makes find_contacts
        # bit-for-bit deterministic and identical across V3/V4 oracles, killing
        # the racy "last writer wins" normal write.
        pair_key = wp.array(
            np.full(K, np.iinfo(np.int64).max, dtype=np.int64),
            dtype=wp.int64, device=_DEVICE)
        edge_threads = K * self.max_edges_per_template * EDGE_SAMPLES
        feat_stride = max(self.max_verts_per_template,
                          self.max_edges_per_template * EDGE_SAMPLES) + 1
        TAG_SHIFT = 28
        assert 4 * feat_stride < (1 << TAG_SHIFT), (
            f"feature tag space overflow: 4*{feat_stride} >= 2^{TAG_SHIFT}")
        Q_MAX = (1 << 34) - 1   # (Q_MAX << TAG_SHIFT) stays < int64 max
        max_dist_off = float(max_dist_u)
        quant_scale = float(Q_MAX) / (2.0 * max_dist_u)
        off_v0, off_v1 = 0, feat_stride
        off_e0, off_e1 = 2 * feat_stride, 3 * feat_stride

        def _launch_vert(p_a, p_b, cax, cay, caz, cbx, cby, cbz, offset, phase):
            wp.launch(
                kernel=_pair_min_distance_kernel_v3,
                dim=K * self.max_verts_per_template,
                inputs=[
                    self.template_mesh_ids_dev, self.body_template_id_dev,
                    self.body_R_dev, self.body_R_T_dev, self.body_center_u_dev,
                    p_a, p_b,
                    self.tpl_vert_offset_dev, self.tpl_vert_count_dev,
                    self.tpl_verts_flat_dev,
                    K, self.max_verts_per_template, max_dist_u,
                    offset, quant_scale, max_dist_off, TAG_SHIFT, Q_MAX, phase,
                    pair_key,
                    sd, nx, ny, nz, cax, cay, caz, cbx, cby, cbz,
                ],
                device=_DEVICE,
            )

        def _launch_edge(p_a, p_b, cax, cay, caz, cbx, cby, cbz, offset, phase):
            wp.launch(
                kernel=_pair_min_distance_edge_kernel_v3,
                dim=edge_threads,
                inputs=[
                    self.template_mesh_ids_dev, self.body_template_id_dev,
                    self.body_R_dev, self.body_R_T_dev, self.body_center_u_dev,
                    p_a, p_b,
                    self.tpl_vert_offset_dev,
                    self.tpl_edge_offset_dev, self.tpl_edge_count_dev,
                    self.tpl_edges_v0_dev, self.tpl_edges_v1_dev,
                    self.tpl_verts_flat_dev,
                    K, self.max_edges_per_template, EDGE_SAMPLES, max_dist_u,
                    offset, quant_scale, max_dist_off, TAG_SHIFT, Q_MAX, phase,
                    pair_key,
                    sd, nx, ny, nz, cax, cay, caz, cbx, cby, cbz,
                ],
                device=_DEVICE,
            )

        # Phase 0 (all four) reduces sd + key; phase 1 (all four) writes the
        # winner's companion data. Same-stream launches serialize, so every
        # phase-0 reduce completes before any phase-1 read of pair_key.
        for phase in (0, 1):
            # Vertex passes (j-verts vs i-mesh, then i-verts vs j-mesh).
            _launch_vert(p_i_dev, p_j_dev,
                         cpax, cpay, cpaz, cpbx, cpby, cpbz, off_v0, phase)
            _launch_vert(p_j_dev, p_i_dev,
                         cpbx, cpby, cpbz, cpax, cpay, cpaz, off_v1, phase)
            # Edge passes (j-edges vs i-mesh, then i-edges vs j-mesh).
            _launch_edge(p_i_dev, p_j_dev,
                         cpax, cpay, cpaz, cpbx, cpby, cpbz, off_e0, phase)
            _launch_edge(p_j_dev, p_i_dev,
                         cpbx, cpby, cpbz, cpax, cpay, cpaz, off_e1, phase)
        wp.synchronize()

        sd_h = sd.numpy()
        nx_h = nx.numpy(); ny_h = ny.numpy(); nz_h = nz.numpy()
        cpa_h = np.stack([cpax.numpy(), cpay.numpy(), cpaz.numpy()], axis=1)
        cpb_h = np.stack([cpbx.numpy(), cpby.numpy(), cpbz.numpy()], axis=1)

        hit = sd_h < max_dist_u - 1e-7
        if not np.any(hit):
            return []
        idx_i = idx_i[hit]; idx_j = idx_j[hit]
        sd_h = sd_h[hit] * s          # u → world
        nx_h = nx_h[hit]; ny_h = ny_h[hit]; nz_h = nz_h[hit]
        cpa_h = cpa_h[hit] * s
        cpb_h = cpb_h[hit] * s
        K = len(idx_i)

        n_h = np.stack([nx_h, ny_h, nz_h], axis=1)
        cdir = centers[idx_j] - centers[idx_i]
        flip = (n_h * cdir).sum(axis=1) < 0
        n_h[flip] = -n_h[flip]
        ln = np.linalg.norm(n_h, axis=1, keepdims=True)
        ln = np.where(ln < 1e-12, 1.0, ln)
        n_h = n_h / ln

        # Extents via template-local kernel.
        def _extents_for(which_pair: np.ndarray, sign: float) -> np.ndarray:
            b_dev = wp.array(np.ascontiguousarray(which_pair, dtype=np.int32),
                             dtype=wp.int32, device=_DEVICE)
            nx_d = wp.array(np.ascontiguousarray(n_h[:, 0], dtype=np.float32),
                            dtype=wp.float32, device=_DEVICE)
            ny_d = wp.array(np.ascontiguousarray(n_h[:, 1], dtype=np.float32),
                            dtype=wp.float32, device=_DEVICE)
            nz_d = wp.array(np.ascontiguousarray(n_h[:, 2], dtype=np.float32),
                            dtype=wp.float32, device=_DEVICE)
            minp = wp.array(np.full(K, 1e30, dtype=np.float32),
                            dtype=wp.float32, device=_DEVICE)
            maxp = wp.array(np.full(K, -1e30, dtype=np.float32),
                            dtype=wp.float32, device=_DEVICE)
            wp.launch(
                kernel=_extent_kernel_v3,
                dim=K * self.max_verts_per_template,
                inputs=[
                    self.body_template_id_dev,
                    self.tpl_vert_offset_dev, self.tpl_vert_count_dev,
                    self.tpl_verts_flat_dev,
                    self.body_R_T_dev,
                    b_dev, nx_d, ny_d, nz_d, float(sign),
                    K, self.max_verts_per_template, minp, maxp,
                ],
                device=_DEVICE,
            )
            wp.synchronize()
            # UNIFY-SUPPORT: one-sided support max_v n^T R vbar (clamped at 0),
            # not full width — matches Prop. assumption (v); certified since
            # the closure coefficient only needs E >= c.
            return np.maximum(maxp.numpy(), 0.0)

        ext_i = _extents_for(idx_i, +1.0)
        ext_j = _extents_for(idx_j, -1.0)

        gap_needed = ds * (ext_i + ext_j)
        keep = sd_h < gap_needed + self.d_hat
        if not np.any(keep):
            return []
        idx_i = idx_i[keep]; idx_j = idx_j[keep]
        sd_h = sd_h[keep]; n_h = n_h[keep]
        cpa_h = cpa_h[keep]; cpb_h = cpb_h[keep]
        ext_i = ext_i[keep]; ext_j = ext_j[keep]

        out = []
        for k in range(len(idx_i)):
            out.append((
                int(idx_i[k]), int(idx_j[k]),
                float(sd_h[k]), n_h[k].astype(np.float64),
                float(ext_i[k]), float(ext_j[k]),
                cpa_h[k].astype(np.float64), cpb_h[k].astype(np.float64),
            ))
        return out
