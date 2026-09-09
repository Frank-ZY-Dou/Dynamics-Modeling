"""S4R-Warp contact oracle v3: template-shared wp.Mesh.

Each unique mesh template gets ONE wp.Mesh (a single BVH built once on the
template's local-frame vertices). Per body only (template_id, R, Rᵀ,
center_u) is kept; the kernels transform every query point from u-space
into the target body's template frame before traversing its BVH.

Narrow phase per candidate pair (i, j), both directions:

* vertex samples of j against the winding-number SDF of i;
* five interior samples per triangle edge of j against the SDF of i;
* every edge of j cast as a ray against the surface of i. A hit is a
  surface crossing and certifies the pair as penetrating even when no
  vertex or edge sample lies inside i; interior points of the crossing
  edge are then sampled to measure a depth.

The reported normal is derived from the winning witness: the sample point
q, its closest surface point cp on the other body, and the SDF sign, so
that moving body j along +n (and i along −n) opens that witness. It is
not aligned to the centroid line, which points the wrong way inside the
cavity of a non-convex body.

Public API: ``find_contacts(s, ds, centers, rotations)`` returns a list of
(i, j, d_signed, n, e_i, e_j, cp_on_i, cp_on_j) in world units.
"""
from __future__ import annotations
import hashlib
import os
import sys
from typing import List

import numpy as np
import warp as wp

_S4R_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s4r")
if _S4R_DIR not in sys.path:
    sys.path.insert(0, _S4R_DIR)
from mesh_collision import outward_faces  # noqa: E402

wp.init()
_DEVICE = "cuda:0"
EDGE_SAMPLES = 5      # interior samples per edge for the SDF pass
CROSS_SAMPLES = 3     # interior samples per crossing edge for the depth


def _template_key(v_model: np.ndarray, f: np.ndarray, nf: float) -> str:
    """Content hash used to detect duplicate mesh templates across bodies;
    nf is included because the same (verts, faces) with a different
    normalize_factor is a different template."""
    h = hashlib.blake2b(digest_size=16)
    h.update(v_model.tobytes())
    h.update(f.tobytes())
    h.update(np.float64(nf).tobytes())
    return h.hexdigest()


@wp.func
def _witness_normal(dir_sign: wp.float32, sd_sign: wp.float32,
                    q_u: wp.vec3, cp_u: wp.vec3) -> wp.vec3:
    """Separation direction in the i->j convention for a sample q with
    closest surface point cp on the other body. dir_sign is +1 when q
    belongs to body j (queried against i) and -1 when q belongs to body i
    (queried against j); sd_sign is the SDF sign of q (+1 outside)."""
    n_u = q_u - cp_u
    ln = wp.length(n_u)
    if ln > 1e-12:
        return n_u * (dir_sign * sd_sign / ln)
    return wp.vec3(0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────
# Kernel A: vertex SDF on the per-template BVH with per-body pose.
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
    dir_sign: wp.float32,
    # deterministic argmin
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
    query = wp.mesh_query_point_sign_winding_number(mesh_i, p_i_local, max_dist_u, 2.0)
    if not query.result:
        return
    cp_local = wp.mesh_eval_position(mesh_i, query.face, query.u, query.v)
    sd = wp.length(p_i_local - cp_local) * query.sign

    # Packed deterministic key: quantised distance (high bits, order
    # preserving) | per-feature tag (low bits, unique across launches).
    # atomic_min on this int64 picks the same (min-distance, min-tag)
    # feature regardless of thread order.
    qd = wp.int64((sd + max_dist_off) * quant_scale)
    if qd < wp.int64(0):
        qd = wp.int64(0)
    if qd > q_max:
        qd = q_max
    tag = wp.int64(launch_offset + v_idx)
    key = (qd << wp.int64(tag_shift)) | tag

    if write_phase == 0:
        wp.atomic_min(pair_sd, k, sd)
        wp.atomic_min(pair_key, k, key)
        return

    if key != pair_key[k]:
        return
    cp_u = body_R[i_body] @ cp_local + body_center_u[i_body]
    n = _witness_normal(dir_sign, query.sign, q_u, cp_u)
    pair_nx[k] = n[0]
    pair_ny[k] = n[1]
    pair_nz[k] = n[2]
    pair_cp_ax[k] = cp_u[0]
    pair_cp_ay[k] = cp_u[1]
    pair_cp_az[k] = cp_u[2]
    pair_cp_bx[k] = q_u[0]
    pair_cp_by[k] = q_u[1]
    pair_cp_bz[k] = q_u[2]


# ─────────────────────────────────────────────────────────────────────
# Kernel A': edge-sampled SDF (same transform pattern).
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
    dir_sign: wp.float32,
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
    threads_per_pair = max_edges_per_template * n_samples
    k = tid // threads_per_pair
    rem = tid % threads_per_pair
    e_local = rem // n_samples
    s_local = rem % n_samples
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

    t = wp.float32(s_local + 1) / wp.float32(n_samples + 1)
    p_sample_j_local = p0_local * (1.0 - t) + p1_local * t
    q_u = body_R[j] @ p_sample_j_local + body_center_u[j]

    i_body = pair_i[k]
    p_i_local = body_R_T[i_body] @ (q_u - body_center_u[i_body])
    i_tpl = body_template_id[i_body]
    mesh_i = template_mesh_ids[i_tpl]
    query = wp.mesh_query_point_sign_winding_number(mesh_i, p_i_local, max_dist_u, 2.0)
    if not query.result:
        return
    cp_local = wp.mesh_eval_position(mesh_i, query.face, query.u, query.v)
    sd = wp.length(p_i_local - cp_local) * query.sign

    qd = wp.int64((sd + max_dist_off) * quant_scale)
    if qd < wp.int64(0):
        qd = wp.int64(0)
    if qd > q_max:
        qd = q_max
    tag = wp.int64(launch_offset + e_local * n_samples + s_local)
    key = (qd << wp.int64(tag_shift)) | tag

    if write_phase == 0:
        wp.atomic_min(pair_sd, k, sd)
        wp.atomic_min(pair_key, k, key)
        return

    if key != pair_key[k]:
        return
    cp_u = body_R[i_body] @ cp_local + body_center_u[i_body]
    n = _witness_normal(dir_sign, query.sign, q_u, cp_u)
    pair_nx[k] = n[0]
    pair_ny[k] = n[1]
    pair_nz[k] = n[2]
    pair_cp_ax[k] = cp_u[0]
    pair_cp_ay[k] = cp_u[1]
    pair_cp_az[k] = cp_u[2]
    pair_cp_bx[k] = q_u[0]
    pair_cp_by[k] = q_u[1]
    pair_cp_bz[k] = q_u[2]


# ─────────────────────────────────────────────────────────────────────
# Kernel A'': edge crossing. Every edge of j is cast as a ray against the
# surface of i. A hit certifies a surface intersection; interior points
# of the crossing segment are then sampled through the SDF so the pair
# gets a measured depth and witness like every other sample.
# ─────────────────────────────────────────────────────────────────────
@wp.kernel
def _pair_edge_crossing_kernel_v3(
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
    dir_sign: wp.float32,
    launch_offset: wp.int32,
    quant_scale: wp.float32,
    max_dist_off: wp.float32,
    tag_shift: wp.int32,
    q_max: wp.int64,
    write_phase: wp.int32,
    pair_key: wp.array(dtype=wp.int64),
    pair_cross: wp.array(dtype=wp.int32),
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
    k = tid // max_edges_per_template
    e_local = tid % max_edges_per_template
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
    p0_u = body_R[j] @ template_verts_flat[v_off + ev0] + body_center_u[j]
    p1_u = body_R[j] @ template_verts_flat[v_off + ev1] + body_center_u[j]

    i_body = pair_i[k]
    c_i = body_center_u[i_body]
    a = body_R_T[i_body] @ (p0_u - c_i)
    b = body_R_T[i_body] @ (p1_u - c_i)
    seg = b - a
    L = wp.length(seg)
    if L < 1e-12:
        return
    d = seg / L
    i_tpl = body_template_id[i_body]
    mesh_i = template_mesh_ids[i_tpl]
    r = wp.mesh_query_ray(mesh_i, a, d, L)
    if not r.result:
        return
    # A hit at the very start or end of the edge is a touching face, not a
    # crossing (the point samples cover it as a zero gap).
    if r.t <= 1.0e-6 * L or r.t >= (1.0 - 1.0e-6) * L:
        return
    if write_phase == 0:
        wp.atomic_max(pair_cross, k, 1)

    # Interior segment of the edge inside body i.
    t0 = wp.float32(0.0)
    t1 = r.t
    if r.sign > 0.0:
        # Entering: interior runs from the hit to the next exit (or the
        # end of the edge).
        t0 = r.t
        t1 = L
        skip = r.t + 1.0e-5 * L
        if skip < L:
            r2 = wp.mesh_query_ray(mesh_i, a + d * skip, d, L - skip)
            if r2.result:
                t1 = skip + r2.t

    for s_local in range(n_samples):
        f = wp.float32(s_local + 1) / wp.float32(n_samples + 1)
        q_loc = a + d * (t0 + f * (t1 - t0))
        query = wp.mesh_query_point_sign_winding_number(mesh_i, q_loc, max_dist_u, 2.0)
        if query.result:
            cp_local = wp.mesh_eval_position(mesh_i, query.face, query.u, query.v)
            sd = wp.length(q_loc - cp_local) * query.sign
            qd = wp.int64((sd + max_dist_off) * quant_scale)
            if qd < wp.int64(0):
                qd = wp.int64(0)
            if qd > q_max:
                qd = q_max
            tag = wp.int64(launch_offset + e_local * n_samples + s_local)
            key = (qd << wp.int64(tag_shift)) | tag
            if write_phase == 0:
                wp.atomic_min(pair_sd, k, sd)
                wp.atomic_min(pair_key, k, key)
            elif key == pair_key[k]:
                q_u = body_R[i_body] @ q_loc + c_i
                cp_u = body_R[i_body] @ cp_local + c_i
                n = _witness_normal(dir_sign, query.sign, q_u, cp_u)
                pair_nx[k] = n[0]
                pair_ny[k] = n[1]
                pair_nz[k] = n[2]
                pair_cp_ax[k] = cp_u[0]
                pair_cp_ay[k] = cp_u[1]
                pair_cp_az[k] = cp_u[2]
                pair_cp_bx[k] = q_u[0]
                pair_cp_by[k] = q_u[1]
                pair_cp_bz[k] = q_u[2]


# ─────────────────────────────────────────────────────────────────────
# Extent kernel: one-sided support of a body along the contact normal.
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
    """Template-shared Warp contact oracle (no FCL)."""

    DEBUG = False

    def __init__(self, mesh_objects, d_hat: float):
        self.N = len(mesh_objects)
        self.d_hat = float(d_hat)

        key_to_tpl_id: dict[str, int] = {}
        self.tpl_v_local: List[np.ndarray] = []
        self.tpl_faces: List[np.ndarray] = []
        self.tpl_max_extent: List[float] = []
        body_template_id_host = np.empty(self.N, dtype=np.int32)
        self.nfs: List[float] = []
        self.max_extents: List[float] = []

        for i, m in enumerate(mesh_objects):
            nf = float(getattr(m, "normalize_factor", 1.0))
            v_model = np.asarray(m.collision_verts_model, dtype=np.float64)
            # Outward winding: the winding-number sign, the ray sign and the
            # witness direction all assume it.
            f = outward_faces(v_model, m.collision_faces)
            self.nfs.append(nf)
            max_ext = float(nf * np.max(np.linalg.norm(v_model, axis=1)))
            self.max_extents.append(max_ext)
            key = _template_key(v_model, f, nf)
            if key not in key_to_tpl_id:
                key_to_tpl_id[key] = len(self.tpl_v_local)
                self.tpl_v_local.append(nf * v_model)
                self.tpl_faces.append(f)
                self.tpl_max_extent.append(max_ext)
            body_template_id_host[i] = key_to_tpl_id[key]

        self.n_templates = len(self.tpl_v_local)
        self.max_extents_arr = np.asarray(self.max_extents, dtype=np.float64)

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
            points = wp.array(v_local.astype(np.float32), dtype=wp.vec3, device=_DEVICE)
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

        self.body_template_id_host = body_template_id_host
        self.body_template_id_dev = wp.array(body_template_id_host, dtype=wp.int32, device=_DEVICE)
        self.tpl_vert_offset_host = np.asarray(tpl_vert_offsets[:-1], dtype=np.int32)
        self.tpl_vert_count_host = np.asarray(tpl_vert_counts, dtype=np.int32)
        self.max_verts_per_template = int(self.tpl_vert_count_host.max())
        self.tpl_edge_offset_host = np.asarray(tpl_edge_offsets[:-1], dtype=np.int32)
        self.tpl_edge_count_host = np.asarray(tpl_edge_counts, dtype=np.int32)
        self.max_edges_per_template = int(self.tpl_edge_count_host.max())
        self.local_aabb_lo = local_aabb_lo
        self.local_aabb_hi = local_aabb_hi
        self.body_aabb_lo_local = local_aabb_lo[body_template_id_host]
        self.body_aabb_hi_local = local_aabb_hi[body_template_id_host]

        self.tpl_vert_offset_dev = wp.array(self.tpl_vert_offset_host, dtype=wp.int32, device=_DEVICE)
        self.tpl_vert_count_dev = wp.array(self.tpl_vert_count_host, dtype=wp.int32, device=_DEVICE)
        self.tpl_edge_offset_dev = wp.array(self.tpl_edge_offset_host, dtype=wp.int32, device=_DEVICE)
        self.tpl_edge_count_dev = wp.array(self.tpl_edge_count_host, dtype=wp.int32, device=_DEVICE)
        tpl_verts_flat = np.concatenate(all_tpl_verts, axis=0).astype(np.float32)
        self.tpl_verts_flat_dev = wp.array(tpl_verts_flat, dtype=wp.vec3, device=_DEVICE)
        self.tpl_edges_v0_dev = wp.array(np.concatenate(all_tpl_edges_v0).astype(np.int32), dtype=wp.int32, device=_DEVICE)
        self.tpl_edges_v1_dev = wp.array(np.concatenate(all_tpl_edges_v1).astype(np.int32), dtype=wp.int32, device=_DEVICE)
        tpl_ids = np.asarray([m.id for m in self.template_meshes], dtype=np.uint64)
        self.template_mesh_ids_dev = wp.array(tpl_ids, dtype=wp.uint64, device=_DEVICE)

        # Per-body pose buffers (updated on every find_contacts call).
        self.body_R_dev = wp.zeros(self.N, dtype=wp.mat33, device=_DEVICE)
        self.body_R_T_dev = wp.zeros(self.N, dtype=wp.mat33, device=_DEVICE)
        self.body_center_u_dev = wp.zeros(self.N, dtype=wp.vec3, device=_DEVICE)

    # ── broadphase margin shared by V3 (host) and V4 (device) ──
    def broadphase_margin_world(self, ds: float, idx_i, idx_j) -> np.ndarray:
        """Per-pair AABB enlargement (world units): the inflation by ds can
        close at most ds (E_i + E_j) of gap (E = bounding-sphere radius
        about the scaling centre) and the narrow filter keeps pairs up to
        d_hat beyond that, so any pair with a smaller AABB gap must be a
        candidate. The bound depends only on the world geometry."""
        E = self.max_extents_arr
        return ds * (E[idx_i] + E[idx_j]) + self.d_hat

    def find_contacts(self, s: float, ds: float,
                      centers: np.ndarray, rotations: List[np.ndarray]):
        inv_s = 1.0 / s
        N = self.N
        R_host = np.empty((N, 3, 3), dtype=np.float32)
        R_T_host = np.empty((N, 3, 3), dtype=np.float32)
        center_u_host = np.empty((N, 3), dtype=np.float32)
        world_aabb_lo = np.empty((N, 3))
        world_aabb_hi = np.empty((N, 3))
        for i in range(N):
            R = np.asarray(rotations[i], dtype=np.float64)
            R_host[i] = R.astype(np.float32)
            R_T_host[i] = R.T.astype(np.float32)
            center_u = centers[i] * inv_s
            center_u_host[i] = center_u.astype(np.float32)
            lo_local = self.body_aabb_lo_local[i]
            hi_local = self.body_aabb_hi_local[i]
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

        # ── Broad phase (CPU, vectorised) ──
        idx_i, idx_j = np.triu_indices(N, k=1)
        margin_u = self.broadphase_margin_world(ds, idx_i, idx_j) * inv_s
        # Euclidean AABB gap (a lower bound on the body distance) against
        # the conservative margin.
        gap_axis = np.maximum(np.maximum(world_aabb_lo[idx_i] - world_aabb_hi[idx_j],
                                         world_aabb_lo[idx_j] - world_aabb_hi[idx_i]), 0.0)
        ovlp = np.einsum('ij,ij->i', gap_axis, gap_axis) <= margin_u * margin_u
        d_ij_now = np.linalg.norm(centers[idx_i] - centers[idx_j], axis=1)
        denom = self.max_extents_arr[idx_i] + self.max_extents_arr[idx_j] + 1e-12
        s_contact_now = np.maximum(0.0, (d_ij_now - self.d_hat) / denom)
        ovlp &= (s + ds >= s_contact_now * 0.9)
        idx_i = idx_i[ovlp]; idx_j = idx_j[ovlp]
        if len(idx_i) == 0:
            return []
        return self._narrow_phase(s, ds, idx_i, idx_j, centers)

    def _narrow_phase(self, s: float, ds: float, idx_i: np.ndarray,
                      idx_j: np.ndarray, centers: np.ndarray):
        """Vertex, edge-sample and edge-crossing passes in both directions
        over the candidate pairs; expects the pose buffers to be current."""
        inv_s = 1.0 / s
        K = len(idx_i)
        p_i_dev = wp.array(np.ascontiguousarray(idx_i, dtype=np.int32), dtype=wp.int32, device=_DEVICE)
        p_j_dev = wp.array(np.ascontiguousarray(idx_j, dtype=np.int32), dtype=wp.int32, device=_DEVICE)
        max_ext_world = float(self.max_extents_arr.max())
        max_dist_world = max(2.0 * max_ext_world, ds * 2.0 * max_ext_world + self.d_hat)
        max_dist_u = float(max_dist_world * inv_s)

        sd = wp.array(np.full(K, max_dist_u, dtype=np.float32), dtype=wp.float32, device=_DEVICE)
        nx = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        ny = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        nz = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpax = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpay = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpaz = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpbx = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpby = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        cpbz = wp.zeros(K, dtype=wp.float32, device=_DEVICE)
        pair_cross = wp.zeros(K, dtype=wp.int32, device=_DEVICE)

        # Deterministic argmin: all six launches reduce into one packed
        # int64 key per pair; a second pass lets only the winning feature
        # write its normal and witness points.
        pair_key = wp.array(np.full(K, np.iinfo(np.int64).max, dtype=np.int64),
                            dtype=wp.int64, device=_DEVICE)
        feat_stride = max(self.max_verts_per_template,
                          self.max_edges_per_template * EDGE_SAMPLES,
                          self.max_edges_per_template * CROSS_SAMPLES) + 1
        TAG_SHIFT = 28
        assert 6 * feat_stride < (1 << TAG_SHIFT), (
            f"feature tag space overflow: 6*{feat_stride} >= 2^{TAG_SHIFT}")
        Q_MAX = (1 << 34) - 1
        max_dist_off = float(max_dist_u)
        quant_scale = float(Q_MAX) / (2.0 * max_dist_u)
        offsets = [t * feat_stride for t in range(6)]
        edge_threads = K * self.max_edges_per_template * EDGE_SAMPLES
        cross_threads = K * self.max_edges_per_template

        def _launch_vert(p_a, p_b, dir_sign, cax, cay, caz, cbx, cby, cbz, offset, phase):
            wp.launch(
                kernel=_pair_min_distance_kernel_v3,
                dim=K * self.max_verts_per_template,
                inputs=[
                    self.template_mesh_ids_dev, self.body_template_id_dev,
                    self.body_R_dev, self.body_R_T_dev, self.body_center_u_dev,
                    p_a, p_b,
                    self.tpl_vert_offset_dev, self.tpl_vert_count_dev,
                    self.tpl_verts_flat_dev,
                    K, self.max_verts_per_template, max_dist_u, float(dir_sign),
                    offset, quant_scale, max_dist_off, TAG_SHIFT, Q_MAX, phase,
                    pair_key,
                    sd, nx, ny, nz, cax, cay, caz, cbx, cby, cbz,
                ],
                device=_DEVICE,
            )

        def _launch_edge(p_a, p_b, dir_sign, cax, cay, caz, cbx, cby, cbz, offset, phase):
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
                    K, self.max_edges_per_template, EDGE_SAMPLES, max_dist_u, float(dir_sign),
                    offset, quant_scale, max_dist_off, TAG_SHIFT, Q_MAX, phase,
                    pair_key,
                    sd, nx, ny, nz, cax, cay, caz, cbx, cby, cbz,
                ],
                device=_DEVICE,
            )

        def _launch_cross(p_a, p_b, dir_sign, cax, cay, caz, cbx, cby, cbz, offset, phase):
            wp.launch(
                kernel=_pair_edge_crossing_kernel_v3,
                dim=cross_threads,
                inputs=[
                    self.template_mesh_ids_dev, self.body_template_id_dev,
                    self.body_R_dev, self.body_R_T_dev, self.body_center_u_dev,
                    p_a, p_b,
                    self.tpl_vert_offset_dev,
                    self.tpl_edge_offset_dev, self.tpl_edge_count_dev,
                    self.tpl_edges_v0_dev, self.tpl_edges_v1_dev,
                    self.tpl_verts_flat_dev,
                    K, self.max_edges_per_template, CROSS_SAMPLES, max_dist_u, float(dir_sign),
                    offset, quant_scale, max_dist_off, TAG_SHIFT, Q_MAX, phase,
                    pair_key, pair_cross,
                    sd, nx, ny, nz, cax, cay, caz, cbx, cby, cbz,
                ],
                device=_DEVICE,
            )

        # Phase 0 reduces sd + key (and the crossing flag); phase 1 writes
        # the winner's companion data. The "a" buffers always hold the
        # witness on body i, the "b" buffers the sample on body j, so the
        # swapped launches swap the buffer roles.
        for phase in (0, 1):
            _launch_vert(p_i_dev, p_j_dev, +1.0, cpax, cpay, cpaz, cpbx, cpby, cpbz, offsets[0], phase)
            _launch_vert(p_j_dev, p_i_dev, -1.0, cpbx, cpby, cpbz, cpax, cpay, cpaz, offsets[1], phase)
            _launch_edge(p_i_dev, p_j_dev, +1.0, cpax, cpay, cpaz, cpbx, cpby, cpbz, offsets[2], phase)
            _launch_edge(p_j_dev, p_i_dev, -1.0, cpbx, cpby, cpbz, cpax, cpay, cpaz, offsets[3], phase)
            _launch_cross(p_i_dev, p_j_dev, +1.0, cpax, cpay, cpaz, cpbx, cpby, cpbz, offsets[4], phase)
            _launch_cross(p_j_dev, p_i_dev, -1.0, cpbx, cpby, cpbz, cpax, cpay, cpaz, offsets[5], phase)
        wp.synchronize()

        sd_h = sd.numpy()
        cross_h = pair_cross.numpy() > 0
        nx_h = nx.numpy(); ny_h = ny.numpy(); nz_h = nz.numpy()
        cpa_h = np.stack([cpax.numpy(), cpay.numpy(), cpaz.numpy()], axis=1)
        cpb_h = np.stack([cpbx.numpy(), cpby.numpy(), cpbz.numpy()], axis=1)

        hit = (sd_h < max_dist_u - 1e-7) | cross_h
        if not np.any(hit):
            return []
        idx_i = idx_i[hit]; idx_j = idx_j[hit]
        sd_h = sd_h[hit]; cross_h = cross_h[hit]
        # A crossing edge is a surface intersection whatever the samples
        # say: report at least a vanishing penetration so the pair is
        # constrained and re-detected.
        sd_h = np.where(cross_h & (sd_h >= 0.0), -1e-9, sd_h)
        sd_h = sd_h * s
        n_h = np.stack([nx_h[hit], ny_h[hit], nz_h[hit]], axis=1).astype(np.float64)
        cpa_h = cpa_h[hit] * s
        cpb_h = cpb_h[hit] * s
        K = len(idx_i)

        # Degenerate witness (coincident sample and surface point): fall
        # back to the centre line, which is well defined for such a pair.
        ln = np.linalg.norm(n_h, axis=1)
        bad = ln < 1e-6
        if np.any(bad):
            cdir = centers[idx_j[bad]] - centers[idx_i[bad]]
            cl = np.linalg.norm(cdir, axis=1, keepdims=True)
            n_h[bad] = cdir / np.where(cl < 1e-12, 1.0, cl)
            ln = np.linalg.norm(n_h, axis=1)
        n_h = n_h / np.where(ln < 1e-12, 1.0, ln)[:, None]

        def _extents_for(which_pair: np.ndarray, sign: float) -> np.ndarray:
            b_dev = wp.array(np.ascontiguousarray(which_pair, dtype=np.int32), dtype=wp.int32, device=_DEVICE)
            nx_d = wp.array(np.ascontiguousarray(n_h[:, 0], dtype=np.float32), dtype=wp.float32, device=_DEVICE)
            ny_d = wp.array(np.ascontiguousarray(n_h[:, 1], dtype=np.float32), dtype=wp.float32, device=_DEVICE)
            nz_d = wp.array(np.ascontiguousarray(n_h[:, 2], dtype=np.float32), dtype=wp.float32, device=_DEVICE)
            minp = wp.array(np.full(K, 1e30, dtype=np.float32), dtype=wp.float32, device=_DEVICE)
            maxp = wp.array(np.full(K, -1e30, dtype=np.float32), dtype=wp.float32, device=_DEVICE)
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
            # One-sided support max_v n^T R v (clamped at 0).
            return np.maximum(maxp.numpy(), 0.0)

        ext_i = _extents_for(idx_i, +1.0)
        ext_j = _extents_for(idx_j, -1.0)

        keep = sd_h < ds * (ext_i + ext_j) + self.d_hat
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
