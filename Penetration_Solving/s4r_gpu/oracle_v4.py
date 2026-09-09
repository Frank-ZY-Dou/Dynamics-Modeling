"""GPU broad phase and pose update on top of the V3 narrow phase.

V3 keeps two host-side hot spots in ``find_contacts``: a Python loop over
N for the per-body pose and world AABB, and an O(N²) numpy broad phase
that materialises every pair. V4 replaces both with Warp kernels:

* ``_pose_update_v4``: one thread per body computes R / Rᵀ / center_u
  and the world AABB from the 8 rotated corners of the local AABB.
* ``_broadphase_v4``: N² threads, each tests one ordered pair with the
  same conservative test as V3 (Euclidean AABB gap against
  ds (E_i + E_j) + d_hat, in u-space); survivors are appended atomically
  and only the count is read back.

The narrow phase is V3's, called through ``_narrow_phase``: the same
kernels, the same witness-oriented normals and the same crossing
certificate, so V3 and V4 return identical contacts for identical pairs.
"""
from __future__ import annotations
import os
import sys
from typing import List

import numpy as np
import warp as wp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from warp_pair_contact_v3 import S4RWarpContactOracleV3  # noqa: E402

_DEVICE = "cuda:0"


@wp.kernel
def _pose_update_v4(
    center_u_in: wp.array(dtype=wp.vec3),
    rotations_dev: wp.array(dtype=wp.mat33),
    aabb_lo_local: wp.array(dtype=wp.vec3),
    aabb_hi_local: wp.array(dtype=wp.vec3),
    R_dev: wp.array(dtype=wp.mat33),
    R_T_dev: wp.array(dtype=wp.mat33),
    center_u_dev: wp.array(dtype=wp.vec3),
    world_aabb_lo_dev: wp.array(dtype=wp.vec3),
    world_aabb_hi_dev: wp.array(dtype=wp.vec3),
):
    """One thread per body. ``center_u_in`` is center / s computed on the
    host in float64 then cast, bit-identical to V3's host pose loop."""
    i = wp.tid()
    R = rotations_dev[i]
    c_u = center_u_in[i]
    R_dev[i] = R
    R_T_dev[i] = wp.transpose(R)
    center_u_dev[i] = c_u
    lo = aabb_lo_local[i]
    hi = aabb_hi_local[i]
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
    ds: wp.float32,
    d_hat: wp.float32,
    inv_s: wp.float32,
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
    """N² threads; each tests one (i, j) upper-triangle pair: AABB overlap
    with the conservative per-pair margin, then the first-contact prune."""
    tid = wp.tid()
    i = tid / N
    j = tid % N
    if i >= N:
        return
    if j <= i:
        return

    margin_u = (ds * (max_extents[i] + max_extents[j]) + d_hat) * inv_s
    lo_i = world_aabb_lo[i]
    hi_i = world_aabb_hi[i]
    lo_j = world_aabb_lo[j]
    hi_j = world_aabb_hi[j]
    # Euclidean AABB gap (lower bound on the body distance) vs the margin.
    gx = wp.max(wp.max(lo_i[0] - hi_j[0], lo_j[0] - hi_i[0]), 0.0)
    gy = wp.max(wp.max(lo_i[1] - hi_j[1], lo_j[1] - hi_i[1]), 0.0)
    gz = wp.max(wp.max(lo_i[2] - hi_j[2], lo_j[2] - hi_i[2]), 0.0)
    if gx * gx + gy * gy + gz * gz > margin_u * margin_u:
        return

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

    idx = wp.atomic_add(pair_count, 0, 1)
    if idx < max_pairs:
        pair_i_out[idx] = i
        pair_j_out[idx] = j


class S4RWarpContactOracleV4(S4RWarpContactOracleV3):
    """V3 oracle with the broad phase and pose update on the GPU."""

    def __init__(self, mesh_objects, d_hat: float):
        super().__init__(mesh_objects, d_hat)
        self._init_v4()

    def _init_v4(self):
        N = self.N
        self.body_aabb_lo_local_dev = wp.array(
            self.body_aabb_lo_local.astype(np.float32).reshape(-1), dtype=wp.vec3, device=_DEVICE)
        self.body_aabb_hi_local_dev = wp.array(
            self.body_aabb_hi_local.astype(np.float32).reshape(-1), dtype=wp.vec3, device=_DEVICE)
        self.world_aabb_lo_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        self.world_aabb_hi_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        self.centers_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        self.center_u_in_dev = wp.zeros(N, dtype=wp.vec3, device=_DEVICE)
        self.rotations_dev = wp.zeros(N, dtype=wp.mat33, device=_DEVICE)
        self.max_extents_dev = wp.array(self.max_extents_arr.astype(np.float32), dtype=wp.float32, device=_DEVICE)
        # Pair output buffers; capacity N(N-1)/2 (every pair can survive).
        self.max_pairs_v4 = max(N * (N - 1) // 2, 1)
        self.pair_count_dev = wp.zeros(1, dtype=wp.int32, device=_DEVICE)
        self.broad_pair_i_dev = wp.zeros(self.max_pairs_v4, dtype=wp.int32, device=_DEVICE)
        self.broad_pair_j_dev = wp.zeros(self.max_pairs_v4, dtype=wp.int32, device=_DEVICE)

    def find_contacts(self, s: float, ds: float,
                      centers: np.ndarray, rotations: List[np.ndarray]):
        inv_s = 1.0 / s
        N = self.N

        centers_h = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1)
        center_u_h = (np.ascontiguousarray(centers, dtype=np.float64) * inv_s).astype(np.float32).reshape(-1)
        R_host = np.stack([np.asarray(r, dtype=np.float32) for r in rotations]).reshape(-1)
        self.centers_dev.assign(centers_h)
        self.center_u_in_dev.assign(center_u_h)
        self.rotations_dev.assign(R_host)

        wp.launch(_pose_update_v4, dim=N,
                  inputs=[self.center_u_in_dev, self.rotations_dev,
                          self.body_aabb_lo_local_dev, self.body_aabb_hi_local_dev,
                          self.body_R_dev, self.body_R_T_dev, self.body_center_u_dev,
                          self.world_aabb_lo_dev, self.world_aabb_hi_dev],
                  device=_DEVICE)

        wp.launch(_zero_int1, dim=1, inputs=[self.pair_count_dev], device=_DEVICE)
        wp.launch(_broadphase_v4, dim=N * N,
                  inputs=[N, float(ds), float(self.d_hat), float(inv_s), float(s + ds),
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
        K = min(K, self.max_pairs_v4)
        idx_i = self.broad_pair_i_dev.numpy()[:K].copy()
        idx_j = self.broad_pair_j_dev.numpy()[:K].copy()
        # The atomic append order is not deterministic; sort to V3's
        # lexicographic order so the QP sees the same contact order run
        # after run and V4 matches V3 bit for bit.
        order = np.lexsort((idx_j, idx_i))
        idx_i = np.ascontiguousarray(idx_i[order])
        idx_j = np.ascontiguousarray(idx_j[order])
        return self._narrow_phase(s, ds, idx_i, idx_j, centers)
