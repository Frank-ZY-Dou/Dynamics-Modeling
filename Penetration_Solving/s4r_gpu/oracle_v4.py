"""GPU broad-phase + GPU pose update on top of V3 narrow-phase.

V3 had two host-side hot spots in ``find_contacts``:

  • a Python loop over N to compute per-body R / Rᵀ / center_u and a
    rotated world AABB — gets slow at N ≳ 2000
  • a numpy O(N²) broad-phase via ``np.triu_indices(N)`` plus AABB
    overlap filtering, which materialises a 2M-row pair array at
    N=2000 (~150 ms / call)

V4 replaces both with Warp kernels:

  • ``_pose_update_v4`` — one thread per body, computes R / Rᵀ /
    center_u and the world AABB by enumerating 8 rotated corners of
    the body's local AABB.
  • ``_broadphase_v4`` — N² threads, each tests one (i, j) ordered
    pair; survivors atomically appended to (pair_i, pair_j) device
    arrays. Only the surviving pair count is read back to host
    (1 int).

The narrow-phase (vertex + edge + extent kernels) is reused verbatim
from V3 — same 8-tuple output, same sign convention, no behavioural
change downstream.
"""
from __future__ import annotations
import os
import sys
from typing import List

import numpy as np
import warp as wp

_REPO_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from warp_pair_contact_v3 import (  # noqa: E402
    S4RWarpContactOracleV3,
    _pair_min_distance_kernel_v3,
    _pair_min_distance_edge_kernel_v3,
    _extent_kernel_v3,
    EDGE_SAMPLES,
)

_DEVICE = "cuda:0"


# ───────────────────────────────────────────────────────────────────────
# Warp kernels for V4's GPU broad-phase
# ───────────────────────────────────────────────────────────────────────


@wp.kernel
def _pose_update_v4(
    center_u_in: wp.array(dtype=wp.vec3),
    rotations_dev: wp.array(dtype=wp.mat33),
    aabb_lo_local: wp.array(dtype=wp.vec3),
    aabb_hi_local: wp.array(dtype=wp.vec3),
    # outputs:
    R_dev: wp.array(dtype=wp.mat33),
    R_T_dev: wp.array(dtype=wp.mat33),
    center_u_dev: wp.array(dtype=wp.vec3),
    world_aabb_lo_dev: wp.array(dtype=wp.vec3),
    world_aabb_hi_dev: wp.array(dtype=wp.vec3),
):
    """One thread per body.

    ``center_u_in`` is the scaled center ``center / s`` computed HOST-SIDE in
    float64 then cast to float32 — bit-identical to V3's host pose loop. Doing
    the ``center * (1/s)`` multiply in native float32 here instead (the original
    V4) rounds ~1e-7 differently whenever 1/s is not a power of two, which flips
    the contact argmin on a few near-tied features per step and compounds into
    a measurable RMSD drift at very large N. The N-body scaling is trivial on
    the host, so this stays effectively all-GPU."""
    i = wp.tid()
    R = rotations_dev[i]
    c_u = center_u_in[i]

    R_dev[i] = R
    R_T_dev[i] = wp.transpose(R)
    center_u_dev[i] = c_u

    lo = aabb_lo_local[i]
    hi = aabb_hi_local[i]

    # 8 rotated corners of the local AABB → world u-space AABB.
    p0 = R * wp.vec3(lo[0], lo[1], lo[2]) + c_u
    p1 = R * wp.vec3(hi[0], lo[1], lo[2]) + c_u
    p2 = R * wp.vec3(lo[0], hi[1], lo[2]) + c_u
    p3 = R * wp.vec3(hi[0], hi[1], lo[2]) + c_u
    p4 = R * wp.vec3(lo[0], lo[1], hi[2]) + c_u
    p5 = R * wp.vec3(hi[0], lo[1], hi[2]) + c_u
    p6 = R * wp.vec3(lo[0], hi[1], hi[2]) + c_u
    p7 = R * wp.vec3(hi[0], hi[1], hi[2]) + c_u

    mn = wp.min(wp.min(wp.min(p0, p1), wp.min(p2, p3)),
                wp.min(wp.min(p4, p5), wp.min(p6, p7)))
    mx = wp.max(wp.max(wp.max(p0, p1), wp.max(p2, p3)),
                wp.max(wp.max(p4, p5), wp.max(p6, p7)))
    world_aabb_lo_dev[i] = mn
    world_aabb_hi_dev[i] = mx


@wp.kernel
def _zero_int1(buf: wp.array(dtype=wp.int32)):
    buf[0] = 0


@wp.kernel
def _broadphase_v4(
    N: wp.int32,
    margin_u: wp.float32,
    d_hat: wp.float32,
    s_plus_ds: wp.float32,
    world_aabb_lo: wp.array(dtype=wp.vec3),
    world_aabb_hi: wp.array(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    max_extents: wp.array(dtype=wp.float32),
    pair_count: wp.array(dtype=wp.int32),
    pair_i_out: wp.array(dtype=wp.int32),
    pair_j_out: wp.array(dtype=wp.int32),
    max_pairs: wp.int32,
):
    """N² threads; each tests one (i, j) ordered pair (upper triangle).
    AABB overlap + first-contact prune; survivors atomically appended
    to (pair_i_out, pair_j_out)."""
    tid = wp.tid()
    i = tid / N
    j = tid % N
    if i >= N:
        return
    if j <= i:
        return

    # AABB overlap test (with margin).
    lo_i = world_aabb_lo[i]
    hi_i = world_aabb_hi[i]
    lo_j = world_aabb_lo[j]
    hi_j = world_aabb_hi[j]
    if lo_i[0] - margin_u > hi_j[0]:
        return
    if lo_j[0] - margin_u > hi_i[0]:
        return
    if lo_i[1] - margin_u > hi_j[1]:
        return
    if lo_j[1] - margin_u > hi_i[1]:
        return
    if lo_i[2] - margin_u > hi_j[2]:
        return
    if lo_j[2] - margin_u > hi_i[2]:
        return

    # First-contact prune.
    c_i = centers[i]
    c_j = centers[j]
    dx = c_j[0] - c_i[0]
    dy = c_j[1] - c_i[1]
    dz = c_j[2] - c_i[2]
    d_ij = wp.sqrt(dx * dx + dy * dy + dz * dz)
    denom = max_extents[i] + max_extents[j] + 1e-12
    s_contact = (d_ij - d_hat) / denom
    if s_contact < 0.0:
        s_contact = 0.0
    if s_plus_ds < s_contact * 0.9:
        return

    # Pair survives → atomic append to output list.
    idx = wp.atomic_add(pair_count, 0, 1)
    if idx < max_pairs:
        pair_i_out[idx] = i
        pair_j_out[idx] = j


# ───────────────────────────────────────────────────────────────────────
# V4 oracle = V3 init + GPU pose / broad-phase override
# ───────────────────────────────────────────────────────────────────────


class S4RWarpContactOracleV4(S4RWarpContactOracleV3):
    """V3 oracle with broad-phase + pose update moved to GPU."""

    def __init__(self, mesh_objects, d_hat: float):
        super().__init__(mesh_objects, d_hat)
        self._init_v4()

    def _init_v4(self):
        N = self.N
        # Per-body local AABB on device (template-driven, constant).
        self.body_aabb_lo_local_dev = wp.array(
            self.body_aabb_lo_local.astype(np.float32).reshape(-1),
            dtype=wp.vec3, device=_DEVICE)
        self.body_aabb_hi_local_dev = wp.array(
            self.body_aabb_hi_local.astype(np.float32).reshape(-1),
            dtype=wp.vec3, device=_DEVICE)
        # Per-body world AABB on device (updated per call).
        self.world_aabb_lo_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        self.world_aabb_hi_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        # Centers and rotations on device (uploaded per call).
        self.centers_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        # Scaled center (center/s) computed host-side in float64→float32 to
        # match V3 bit-for-bit (§8.8); fed to the pose kernel.
        self.center_u_in_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        self.rotations_dev = wp.zeros(N, dtype=wp.mat33, device=_DEVICE)
        # max_extents on device (constant after init).
        self.max_extents_dev = wp.array(
            self.max_extents_arr.astype(np.float32),
            dtype=wp.float32, device=_DEVICE)
        # Pair output buffers; conservative capacity = N(N-1)/2.
        self.max_pairs_v4 = max(N * (N - 1) // 2, 1)
        self.pair_count_dev = wp.zeros(1, dtype=wp.int32, device=_DEVICE)
        self.broad_pair_i_dev = wp.zeros(self.max_pairs_v4,
                                         dtype=wp.int32, device=_DEVICE)
        self.broad_pair_j_dev = wp.zeros(self.max_pairs_v4,
                                         dtype=wp.int32, device=_DEVICE)
        self.nfs_max = float(max(self.nfs))

    def find_contacts(self, s: float, ds: float,
                      centers: np.ndarray, rotations: List[np.ndarray]):
        inv_s = 1.0 / s
        N = self.N

        # ── [A] Upload pose, run GPU pose update ──
        # Raw centers (float32) drive the broad-phase d_ij prune; the SCALED
        # center center/s is computed host-side in float64 then cast to float32
        # — bit-identical to V3 (§8.8) — and feeds the pose / narrow-phase.
        centers_h = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1)
        center_u_h = (np.ascontiguousarray(centers, dtype=np.float64)
                      * inv_s).astype(np.float32).reshape(-1)
        R_host = np.stack(
            [np.asarray(r, dtype=np.float32) for r in rotations]
        ).reshape(-1)
        self.centers_dev.assign(centers_h)
        self.center_u_in_dev.assign(center_u_h)
        self.rotations_dev.assign(R_host)

        wp.launch(_pose_update_v4, dim=N,
                  inputs=[self.center_u_in_dev, self.rotations_dev,
                          self.body_aabb_lo_local_dev,
                          self.body_aabb_hi_local_dev,
                          self.body_R_dev, self.body_R_T_dev,
                          self.body_center_u_dev,
                          self.world_aabb_lo_dev, self.world_aabb_hi_dev],
                  device=_DEVICE)

        # ── [B] GPU broad-phase ──
        margin_world = ds * self.nfs_max * 0.2
        margin_u = margin_world * inv_s

        wp.launch(_zero_int1, dim=1,
                  inputs=[self.pair_count_dev], device=_DEVICE)
        wp.launch(_broadphase_v4, dim=N * N,
                  inputs=[N, float(margin_u), float(self.d_hat),
                          float(s + ds),
                          self.world_aabb_lo_dev, self.world_aabb_hi_dev,
                          self.centers_dev, self.max_extents_dev,
                          self.pair_count_dev,
                          self.broad_pair_i_dev, self.broad_pair_j_dev,
                          int(self.max_pairs_v4)],
                  device=_DEVICE)
        wp.synchronize()

        K = int(self.pair_count_dev.numpy()[0])
        if K == 0:
            return []
        if K > self.max_pairs_v4:
            # Should never happen with N(N-1)/2 capacity, but defensive.
            K = self.max_pairs_v4

        # Read idx_i, idx_j back to host for final tuple construction
        # (small K, ~1 ms at K=10k).
        idx_i = self.broad_pair_i_dev.numpy()[:K].copy()
        idx_j = self.broad_pair_j_dev.numpy()[:K].copy()

        # Deterministic lexicographic (i, j) order. The GPU broad-phase appends
        # survivors via atomic_add, so their order is non-deterministic
        # run-to-run; the pipeline consumes contacts in list order to build the
        # QP, so that scrambled order makes float32 APGD accumulate differently
        # each run (the ~0.3% run-to-run RMSD spread, §8.8). Sorting to V3's
        # np.triu order makes V4 reproducible AND bit-identical to V3 end-to-end.
        order = np.lexsort((idx_j, idx_i))
        idx_i = np.ascontiguousarray(idx_i[order])
        idx_j = np.ascontiguousarray(idx_j[order])
        p_i_dev = wp.array(idx_i, dtype=wp.int32, device=_DEVICE)
        p_j_dev = wp.array(idx_j, dtype=wp.int32, device=_DEVICE)

        # ── [C] Narrow phase — verbatim from V3 ──
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

        # ── Deterministic-argmin bookkeeping (shared kernels, §8.5) ──
        # Identical two-phase scheme to V3.find_contacts: phase 0 reduces sd +
        # packed key, phase 1 lets only the winning feature write its normal/cp.
        # Because the narrow-phase kernels are shared with V3, V4 now returns
        # bit-for-bit the same contact normals as V3 on shared pairs.
        pair_key = wp.array(
            np.full(K, np.iinfo(np.int64).max, dtype=np.int64),
            dtype=wp.int64, device=_DEVICE)
        edge_threads = K * self.max_edges_per_template * EDGE_SAMPLES
        feat_stride = max(self.max_verts_per_template,
                          self.max_edges_per_template * EDGE_SAMPLES) + 1
        TAG_SHIFT = 28
        assert 4 * feat_stride < (1 << TAG_SHIFT), (
            f"feature tag space overflow: 4*{feat_stride} >= 2^{TAG_SHIFT}")
        Q_MAX = (1 << 34) - 1
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

        for phase in (0, 1):
            _launch_vert(p_i_dev, p_j_dev,
                         cpax, cpay, cpaz, cpbx, cpby, cpbz, off_v0, phase)
            _launch_vert(p_j_dev, p_i_dev,
                         cpbx, cpby, cpbz, cpax, cpay, cpaz, off_v1, phase)
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
        sd_h = sd_h[hit] * s
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
