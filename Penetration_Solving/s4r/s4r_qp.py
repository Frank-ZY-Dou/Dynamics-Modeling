"""S4R-QP: Progressive scaling + QP displacement correction (6-DOF).

Each step: compute exact push (translation + rotation) via QP, then inflate.
No CCD, no Newton, no iteration — one QP solve per scale step.
"""
import os
import numpy as np
import trimesh
import time
import scipy.sparse as sp
import osqp
from scipy.spatial.transform import Rotation as RotLib

import sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mesh_collision import _contains_points as _mesh_contains_points  # noqa: E402


def solve_s4r_qp(objects, d_hat=0.02, ds_max=0.05, max_steps=200, verbose=True,
                  enable_rotation=False, rotation_weight=None, max_omega=0.1,
                  adaptive_ds=False, contact_sparsity=False, use_dual=False,
                  revalidate_interval=3, audit=False,
                  contact_backend='fcl',
                  trajectory_dumper=None, dump_every=1,
                  target_centers=None, attraction_alpha=None,
                  perturb_rot_deg=0.0, perturb_seed=None,
                  box_bounds=None, joints=None,
                  profile=False):
    """
    S4R with QP-based displacement correction (6-DOF: translation + rotation).

    At each step:
    1. Find active contacts (pairs that would collide after inflation by ds)
    2. Solve QP: min ||Δp||² + β||ω||²  s.t. linearized contact constraints
    3. Apply displacement + rotation + inflate
    """
    def _sync_backend():
        if contact_backend == 'warp':
            import warp as wp
            wp.synchronize()

    # Count all per-scene method setup; module imports are excluded.
    _sync_backend()
    method_t0 = time.perf_counter()
    N = len(objects)
    centers = np.array([o.center for o in objects], dtype=np.float64)
    centers0 = centers.copy()
    rots = [o.rotation.copy() for o in objects]
    nfs = [o.normalize_factor for o in objects]
    mverts = [o.collision_verts_model.copy() for o in objects]
    mfaces = [o.collision_faces.copy() for o in objects]

    # Phase-time accumulators (only populated when profile=True).
    _phase_times = {'contact': 0.0, 'qp': 0.0, 'tail': 0.0}
    # Initial scale s_min: must satisfy eq:smin_safe (§3.1.1):
    #     s_min < min_{i≠j} (||c_i - c_j|| - d_hat) / (R_i + R_j)
    # The fixed default 0.01 is verified against this bound at the top of
    # the solve and a warning is printed if it is violated.
    scale = 0.01  # global scale (s_min)

    # ── Optional one-time stochastic SO(3) re-orientation at s_min ───────
    # "Shake the rice bag": at the shrunk, collision-free state, randomly
    # perturb each body's orientation, then let the 6-DOF QP re-optimize
    # rotation as bodies re-inflate. On packing-limited scenes this can
    # escape a loose local packing and settle into a denser one (lower
    # centroid RMSD). Rotation-only: writes rots[i] about the body's own
    # pivot centers[i], so the RMSD reference centers0 (line above) is
    # untouched. Uses a SEPARATE rng so it never desyncs the scene seed.
    if enable_rotation and perturb_rot_deg and perturb_rot_deg > 0.0:
        _prng = np.random.default_rng(perturb_seed)
        _ang = np.deg2rad(float(perturb_rot_deg))
        for i in range(N):
            axis = _prng.normal(size=3)
            nrm = np.linalg.norm(axis)
            if nrm < 1e-12:
                continue
            rvec = (axis / nrm) * _ang
            rots[i] = RotLib.from_rotvec(rvec).as_matrix() @ rots[i]

    # Pre-compute max extent per body (for rotation weight normalization)
    max_extents = np.array([nfs[i] * np.max(np.linalg.norm(mverts[i], axis=1))
                            for i in range(N)])
    max_extent_all = max_extents.max()

    # ── Optional box-confinement (container) constraints ──────────────────
    # When box_bounds=((lo_x,lo_y,lo_z),(hi_x,hi_y,hi_z)) is given, every body
    # must stay inside the axis-aligned container box at each scale step. This
    # is a demonstration of adding hard scene-structure constraints (walls of a
    # container) to the per-step QP, beyond the single support plane of
    # Sec. upright-on-plane. box_bounds=None (default) leaves the solver's
    # behaviour bit-identical to the unconstrained version.
    if box_bounds is not None:
        box_lo = np.asarray(box_bounds[0], dtype=np.float64)
        box_hi = np.asarray(box_bounds[1], dtype=np.float64)
        # Per-body, per-axis support half-extents at full scale (s=1), under the
        # body's fixed rotation: offset of the extreme vertex from the centroid
        # along +axis (sup_plus) and -axis (sup_minus).
        # NOTE: these support half-extents are computed once at the INITIAL
        # rotation. box_bounds is therefore exact for the translation-only
        # (default) path used in the container experiment. If enable_rotation
        # is combined with box_bounds in future use, recompute these per step
        # from the current rots[i]; otherwise a rotating body could protrude.
        box_sup_plus = np.zeros((N, 3))
        box_sup_minus = np.zeros((N, 3))
        for i in range(N):
            wv = nfs[i] * (mverts[i] @ rots[i].T)   # world-frame vertex offsets at s=1
            box_sup_plus[i] = wv.max(axis=0)
            box_sup_minus[i] = -wv.min(axis=0)
        if box_bounds is not None and enable_rotation:
            import warnings as _w
            _w.warn("box_bounds support extents are fixed at the initial rotation; "
                    "exact only for translation-only. Recompute per-step for 6-DOF.")

    if rotation_weight is None:
        # β = R_max^2 so that β||ω||^2 matches ||Δp||^2 in length^2 units
        # (surface arc displacement from rotation ω is R||ω||).  Previous
        # versions of this file used the dimensionally inverted form
        # 1/R_max^2; the small-angle box ||ω||_∞ <= ω_max keeps both forms
        # in the same operating regime, but only β = R_max^2 makes the
        # objective penalty units consistent (cf. §3 of the paper).
        rotation_weight = (max_extent_all ** 2)

    # Pre-compute first-contact scale for broadphase acceleration
    # and simultaneously verify the eq:smin_safe bound on the chosen s_min.
    # this is S4R-specific algorithmic work (SOI
    # schedule), so it is (a) vectorized — the old Python double loop was
    # O(N^2) interpreter work — and (b) charged to S4R's reported solver
    # time via _soi_pre_elapsed below (baselines are not billed for it, so
    # S4R must not get it for free either).
    _soi_pre_t0 = time.time()
    _ii, _jj = np.triu_indices(N, k=1)
    _C = np.asarray(centers, dtype=np.float64)
    _E = np.asarray(max_extents, dtype=np.float64)
    _d = np.linalg.norm(_C[_ii] - _C[_jj], axis=1)
    _sc = np.minimum(np.maximum(0.0, (_d - d_hat) / (_E[_ii] + _E[_jj] + 1e-12)), 2.0)
    pair_first_contact = dict(zip(zip(_ii.tolist(), _jj.tolist()), _sc.tolist()))
    s_min_safe = float(_sc.min()) if len(_sc) else float('inf')
    _soi_pre_elapsed = time.time() - _soi_pre_t0
    if verbose:
        print(f"  [s_min check] eq:smin_safe bound = {s_min_safe:.4f}, "
              f"using s_min = {scale:.4f}  "
              f"({'OK' if scale < s_min_safe else 'TOO LARGE; will jitter'})")
    if scale >= s_min_safe:
        # Some pair violates the s_min bound (coincident or near-coincident
        # centroids). Deterministically perturb the pair apart along their
        # center axis until  ||c_i - c_j|| >= d_hat + s_min*(r_i+r_j) + eps,
        # which is exactly what Eq. (smin_safe) needs for the fixed s_min to
        # be admissible. (Jittering by d_hat alone is NOT enough: for
        # exactly-coincident centroids it leaves the Eq. numerator at zero,
        # so no positive s_min exists.) For an exactly-coincident pair the
        # separation axis is chosen deterministically (+x).
        for (i, j), sc in pair_first_contact.items():
            if sc < scale:
                axis = centers[j] - centers[i]
                n = float(np.linalg.norm(axis))
                if n < 1e-12:
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis = axis / n
                ext_sum = max_extents[i] + max_extents[j]
                target = d_hat + scale * ext_sum + 1e-6
                gap = max(0.0, target - n)
                shift = gap * axis
                centers[j] = centers[j] + 0.5 * shift
                centers[i] = centers[i] - 0.5 * shift
                if verbose:
                    print(f"    jitter pair ({i},{j}) by ±{0.5 * gap:.5f} "
                          f"(to ||c_i-c_j||={target:.5f})")

    def world_verts(i, s=None):
        if s is None: s = scale
        return s * nfs[i] * (rots[i] @ mverts[i].T).T + centers[i]

    def build_mesh(i, s=None):
        return trimesh.Trimesh(vertices=world_verts(i, s), faces=mfaces[i], process=False)

    def compute_extent(i, direction):
        """One-sided support of body i along direction (world frame),
        about the scaling center: max_v n^T R_i vbar (Prop. assumption (v)).
        Clamped below at zero — a further conservative bound; the closure
        coefficient only needs E >= c. (UNIFY-SUPPORT: was full projected
        width max-min, which under-estimates the support when the scaling
        center lies outside the hull slab along n.)"""
        local_dir = rots[i].T @ direction
        projs = nfs[i] * (mverts[i] @ local_dir)
        return max(projs.max(), 0.0)

    # Cache for the FCL backend: rebuilt per find_contacts call at
    # the current scale (we can't express uniform scaling via FCL
    # Transform alone because FCL rotation must be orthonormal).
    fcl_objs_cache = [None]

    def _build_fcl_objs(s_now):
        import fcl
        fcl_list = []
        for i in range(N):
            v_world = world_verts(i, s_now)
            m = fcl.BVHModel()
            m.beginModel(len(v_world), len(mfaces[i]))
            m.addSubModel(v_world.astype(np.float64),
                          mfaces[i].astype(np.int32))
            m.endModel()
            fcl_list.append(fcl.CollisionObject(m, fcl.Transform()))
        return fcl_list

    # ------------------------------------------------------------------
    # fcl_prebuilt backend: build BVHs ONCE at scale=1 and reuse.
    # Trick: change of variable u = x / s. The scale-s body has world
    # vertices s * M[i] + centers[i] where M[i] = nfs[i] * rots[i] @ mverts[i].
    # Equivalently, in u-space: M[i] + centers[i]/s. So we build BVH on
    # M[i] once and query with translation centers[i]/s. FCL distances
    # in u-space are (d_world / s); multiply by s to recover world units.
    # Contact normals are scale-invariant (uniform scaling preserves
    # directions). Contact points in u-space → multiply by s.
    # ------------------------------------------------------------------
    # Rotation-free, scale-1 BVH cache shared by translation oracle
    # (_find_contacts_fcl_prebuilt) and SO(3) refinement loop
    # (optimize_rotations_on_manifold). Built on nfs[i]*mverts[i]; rotation
    # is applied via fcl.Transform(rots[i], centers[i]/s) per query so the
    # cache stays valid as 6-DOF refinement mutates rots[i].
    _prebuilt_bvh: list = []       # fcl.BVHModel per body
    _prebuilt_M: list = []         # nfs[i] * mverts[i]  (model frame)

    def _init_prebuilt_fcl():
        """Build unit-scale, rotation-FREE BVHs and model-space cache.

        We build BVHs on ``nfs[i] * mverts[i]`` (model frame, scale-1, no
        rotation baked in) and let ``fcl.Transform(rots[i], centers[i]/s)``
        apply rotation per query. This keeps the BVH valid even when
        ``rots[i]`` mutates mid-solve (6-DOF path); for 3-DOF it is
        semantically identical to baking rotation into the BVH.

        World AABBs are pose-dependent and recomputed per call in
        ``_find_contacts_fcl_prebuilt`` (cheap: ~N·V flops in numpy).
        """
        import fcl
        _prebuilt_bvh.clear()
        _prebuilt_M.clear()
        for i in range(N):
            Mi = (nfs[i] * mverts[i]).astype(np.float64)  # model-space, scale-1
            _prebuilt_M.append(Mi)
            m = fcl.BVHModel()
            faces_i = mfaces[i].astype(np.int32)
            m.beginModel(len(Mi), len(faces_i))
            m.addSubModel(Mi, faces_i)
            m.endModel()
            _prebuilt_bvh.append(m)

    # Warp GPU oracle (lazily initialised; we don't pay the wp.init cost
    # unless the user actually requests it).
    _warp_oracle = [None]

    def _init_warp_oracle():
        if _warp_oracle[0] is None:
            # The released Warp GPU oracle lives in ``s4r_gpu/``. Add that
            # directory to the path and import the template-shared V3 oracle
            # (the only Warp oracle shipped; requires an NVIDIA GPU +
            # ``warp-lang``). The import is deferred to here so the default
            # CPU backends never pay the ``warp`` import cost.
            _gpu_dir = _os.path.join(_os.path.dirname(_HERE), "s4r_gpu")
            if _gpu_dir not in sys.path:
                sys.path.insert(0, _gpu_dir)
            from warp_pair_contact_v3 import (
                S4RWarpContactOracleV3 as S4RWarpContactOracle,
            )
            # Build a thin shim so the oracle has the .normalize_factor,
            # .collision_verts_model and .collision_faces it expects.
            class _ObjShim:
                __slots__ = ("normalize_factor", "collision_verts_model",
                             "collision_faces")
                def __init__(self, nf, v, f):
                    self.normalize_factor = nf
                    self.collision_verts_model = v
                    self.collision_faces = f
            shims = [_ObjShim(nfs[i], mverts[i], mfaces[i])
                     for i in range(N)]
            _warp_oracle[0] = S4RWarpContactOracle(shims, d_hat=d_hat)
        return _warp_oracle[0]

    def _find_contacts_warp(s, ds):
        oracle = _init_warp_oracle()
        return oracle.find_contacts(s, ds, centers, rots)

    def find_contacts(s, ds, bidirectional=True):
        """Find pairs needing attention. Returns list of:
        (i, j, d_signed, normal, extent_i, extent_j, cp_on_i, cp_on_j)

        Dispatches to `contact_backend` ∈
        {"trimesh", "fcl", "fcl_prebuilt", "warp"}. All paths return the
        same tuple shape and sign convention:
          - d_signed < 0 means penetration (depth = -d_signed)
          - d_signed > 0 means a gap
          - `normal` points roughly from i toward j (positive component
            along centers[j] - centers[i]), matching the QP constraint
            n · (Δp_j - Δp_i) ≥ b.

        `bidirectional` is kept as a kwarg for backwards compatibility
        (ignored by fcl; trimesh path is always bidirectional now).
        """
        if contact_backend == 'warp':
            return _find_contacts_warp(s, ds)
        if contact_backend == 'fcl_prebuilt':
            return _find_contacts_fcl_prebuilt(s, ds)
        if contact_backend == 'fcl':
            return _find_contacts_fcl(s, ds)
        return _find_contacts_trimesh(s, ds)

    def find_contacts_margin(s, ds, extra_margin):
        """find_contacts with an explicit broadphase margin (world units).

        Needed by the E3 safe-margin tail: with ds=0 the prebuilt broadphase
        margin is 0, so SEPARATED pairs with gap < target are invisible to
        the default query (fine for the pen<0 exit, blind for a margin
        target). extra_margin=0.0 is bit-identical to find_contacts.
        Only the paper's 'fcl_prebuilt' backend supports it.
        """
        if extra_margin <= 0.0:
            return find_contacts(s, ds)
        if contact_backend != 'fcl_prebuilt':
            raise NotImplementedError(
                "S4R_TAIL_TARGET_MARGIN requires contact_backend='fcl_prebuilt'")
        return _find_contacts_fcl_prebuilt(s, ds, extra_margin)

    def _find_contacts_fcl_prebuilt(s, ds, extra_margin=0.0):
        """Variant of _find_contacts_fcl that reuses prebuilt scale-1 BVHs.

        Uses the change-of-variable u = x/s: in u-space the body is
        unit-scale at position centers[i]/s. FCL distance in u-space
        times s recovers world distance; contact points in u-space times
        s recover world points; normals are scale-invariant.

        BVHs are built on rotation-FREE model verts and rotation is
        applied via ``fcl.Transform(rots[i], centers[i]/s)`` per query,
        so the cache stays valid when 6-DOF rotation refinement mutates
        ``rots[i]`` between calls. The world AABB used for the broadphase
        is therefore pose-dependent and recomputed per call.
        """
        import fcl
        if not _prebuilt_bvh:
            _init_prebuilt_fcl()
        contacts = []
        inv_s = 1.0 / s

        # Per-call world AABB in u-space: rotate model verts and translate
        # by centers[i]/s. Cheap: ~N·V flops, negligible vs FCL calls.
        u_aabb_min = []; u_aabb_max = []
        for i in range(N):
            Vw = (rots[i] @ _prebuilt_M[i].T).T + centers[i] * inv_s
            u_aabb_min.append(Vw.min(axis=0))
            u_aabb_max.append(Vw.max(axis=0))

        # Pre-wrap CollisionObjects with current rotation (cheap: shares BVH).
        fcl_objs = [fcl.CollisionObject(
            _prebuilt_bvh[i],
            fcl.Transform(rots[i].astype(np.float64),
                          (centers[i] * inv_s).astype(np.float64))
        ) for i in range(N)]

        # Vectorized broadphase over all N(N-1)/2 pairs,
        # replacing the O(N^2) Python per-pair loop (~8M np.linalg.norm/dot calls
        # at N=1000, the top hot spot). Same candidate set as the loop (SOI-cull
        # + u-space AABB overlap), so contacts are IDENTICAL — only faster. This
        # is the S4R analogue of the C1 fix applied to the baseline MeshOracle,
        # so both sides get the same detection optimization (fair comparison).
        C = np.asarray(centers, dtype=np.float64)
        ext_a = np.asarray(max_extents, dtype=np.float64)
        nf_a = np.asarray(nfs, dtype=np.float64)
        lo = np.asarray(u_aabb_min, dtype=np.float64)   # (N,3)
        hi = np.asarray(u_aabb_max, dtype=np.float64)   # (N,3)
        ii, jj = np.triu_indices(N, k=1)
        d_ij = np.linalg.norm(C[ii] - C[jj], axis=1)
        s_contact = np.maximum(0.0,
                               (d_ij - d_hat) / (ext_a[ii] + ext_a[jj] + 1e-12))
        soi_keep = (s + ds) >= (s_contact * 0.9)
        margin_u = ((ds * np.maximum(nf_a[ii], nf_a[jj]) * 0.2
                     + extra_margin) * inv_s)[:, None]
        overlap = np.all((lo[ii] - margin_u <= hi[jj]) &
                         (lo[jj] - margin_u <= hi[ii]), axis=1)
        cand = np.nonzero(soi_keep & overlap)[0]

        for _k in cand:
            i = int(ii[_k]); j = int(jj[_k])
            # Distance query in u-space.
            req = fcl.DistanceRequest(enable_nearest_points=True,
                                      enable_signed_distance=True)
            res = fcl.DistanceResult()
            d_u = fcl.distance(fcl_objs[i], fcl_objs[j], req, res)

            if d_u > 0.0:
                cp_i_u = np.asarray(res.nearest_points[0], dtype=np.float64)
                cp_j_u = np.asarray(res.nearest_points[1], dtype=np.float64)
                n_raw = cp_j_u - cp_i_u        # direction is scale-invariant
                d_signed = d_u * s              # u → world distance
                cp_on_a = cp_i_u * s
                cp_on_b = cp_j_u * s
            else:
                creq = fcl.CollisionRequest(num_max_contacts=16,
                                            enable_contact=True)
                cres = fcl.CollisionResult()
                fcl.collide(fcl_objs[i], fcl_objs[j], creq, cres)
                if not cres.is_collision or not cres.contacts:
                    continue
                # Diagnostic (multi-contact ablation): S4R_ALL_CONTACTS=1 keeps
                # EVERY returned contact as its own constraint row instead of
                # the single deepest witness. Default (0) = deployed behavior.
                if os.environ.get('S4R_ALL_CONTACTS', '0') == '1':
                    for c_k in cres.contacts:
                        depth_u = float(c_k.penetration_depth)
                        depth_world = depth_u * s
                        n_raw = np.asarray(c_k.normal, dtype=np.float64)
                        if np.dot(n_raw, centers[j] - centers[i]) < 0.0:
                            n_raw = -n_raw
                        nn = np.linalg.norm(n_raw)
                        if nn < 1e-12:
                            continue
                        n_k = n_raw / nn
                        pos_world = np.asarray(c_k.pos, dtype=np.float64) * s
                        ext_i_k = compute_extent(i, n_k)
                        ext_j_k = compute_extent(j, -n_k)
                        contacts.append((i, j, -depth_world, n_k, ext_i_k,
                                         ext_j_k, pos_world.copy(),
                                         pos_world + n_k * depth_world))
                    continue
                c_best = max(cres.contacts,
                             key=lambda c: c.penetration_depth)
                depth_u = float(c_best.penetration_depth)
                depth_world = depth_u * s       # u → world depth
                n_raw = np.asarray(c_best.normal, dtype=np.float64)
                if np.dot(n_raw, centers[j] - centers[i]) < 0.0:
                    n_raw = -n_raw
                pos_u = np.asarray(c_best.pos, dtype=np.float64)
                pos_world = pos_u * s
                cp_on_a = pos_world.copy()
                cp_on_b = pos_world + n_raw * depth_world
                d_signed = -depth_world

            nn = np.linalg.norm(n_raw)
            if nn < 1e-12:
                n_raw = centers[j] - centers[i]
                nn = np.linalg.norm(n_raw)
            if nn < 1e-12:
                continue
            n = n_raw / nn

            ext_i = compute_extent(i, n)
            ext_j = compute_extent(j, -n)
            gap_needed = ds * (ext_i + ext_j)
            if d_signed < gap_needed + d_hat:
                contacts.append((i, j, d_signed, n, ext_i, ext_j,
                                 cp_on_a, cp_on_b))
        return contacts

    def _find_contacts_fcl(s, ds):
        import fcl
        contacts = []
        fcl_objs = _build_fcl_objs(s)
        meshes_trim = [build_mesh(i, s) for i in range(N)]  # for broadphase AABBs
        for i in range(N):
            ai0, ai1 = meshes_trim[i].bounds
            for j in range(i + 1, N):
                d_ij_now = float(np.linalg.norm(centers[i] - centers[j]))
                s_contact_now = max(
                    0.0,
                    (d_ij_now - d_hat) / (max_extents[i] + max_extents[j] + 1e-12),
                )
                if s + ds < s_contact_now * 0.9:
                    continue
                aj0, aj1 = meshes_trim[j].bounds
                margin = ds * max(nfs[i], nfs[j]) * 0.2
                if not (np.all(ai0 - margin <= aj1) and np.all(aj0 - margin <= ai1)):
                    continue

                # Distance query first.
                req = fcl.DistanceRequest(enable_nearest_points=True,
                                          enable_signed_distance=True)
                res = fcl.DistanceResult()
                d = fcl.distance(fcl_objs[i], fcl_objs[j], req, res)

                if d > 0.0:
                    cp_i = np.asarray(res.nearest_points[0], dtype=np.float64)
                    cp_j = np.asarray(res.nearest_points[1], dtype=np.float64)
                    n_raw = cp_j - cp_i
                    d_signed = d
                    cp_on_a, cp_on_b = cp_i, cp_j
                else:
                    # Overlap — collision detection for penetration info.
                    creq = fcl.CollisionRequest(num_max_contacts=16,
                                                enable_contact=True)
                    cres = fcl.CollisionResult()
                    fcl.collide(fcl_objs[i], fcl_objs[j], creq, cres)
                    if not cres.is_collision or not cres.contacts:
                        continue
                    c_best = max(cres.contacts,
                                 key=lambda c: c.penetration_depth)
                    depth = float(c_best.penetration_depth)
                    n_raw = np.asarray(c_best.normal, dtype=np.float64)
                    # Align normal so its component points from i toward j.
                    if np.dot(n_raw, centers[j] - centers[i]) < 0.0:
                        n_raw = -n_raw
                    pos = np.asarray(c_best.pos, dtype=np.float64)
                    cp_on_a = pos.copy()
                    cp_on_b = pos + n_raw * depth
                    d_signed = -depth

                nn = np.linalg.norm(n_raw)
                if nn < 1e-12:
                    n_raw = centers[j] - centers[i]
                    nn = np.linalg.norm(n_raw)
                if nn < 1e-12:
                    continue
                n = n_raw / nn

                ext_i = compute_extent(i, n)
                ext_j = compute_extent(j, -n)
                gap_needed = ds * (ext_i + ext_j)
                if d_signed < gap_needed + d_hat:
                    contacts.append((i, j, d_signed, n, ext_i, ext_j,
                                     cp_on_a, cp_on_b))
        return contacts

    def _find_contacts_trimesh(s, ds):
        contacts = []
        meshes = [build_mesh(i, s) for i in range(N)]
        for i in range(N):
            vi = np.asarray(meshes[i].vertices)
            ai0, ai1 = vi.min(0), vi.max(0)
            fn_i = meshes[i].face_normals
            for j in range(i + 1, N):
                # Recompute first-contact scale from CURRENT centroids.
                # (Pre-computed value assumed initial centers; drift invalidates it.)
                d_ij_now = float(np.linalg.norm(centers[i] - centers[j]))
                s_contact_now = max(
                    0.0,
                    (d_ij_now - d_hat) / (max_extents[i] + max_extents[j] + 1e-12),
                )
                if s + ds < s_contact_now * 0.9:
                    continue
                vj = np.asarray(meshes[j].vertices)
                aj0, aj1 = vj.min(0), vj.max(0)
                margin = ds * max(nfs[i], nfs[j]) * 0.2
                if not (np.all(ai0 - margin <= aj1) and np.all(aj0 - margin <= ai1)):
                    continue

                # Bidirectional closest-point queries.
                cl_ji, d_ji, fidx_i = trimesh.proximity.closest_point(meshes[i], vj)
                cl_ij, d_ij, fidx_j = trimesh.proximity.closest_point(meshes[j], vi)

                # Inside flags via the SAME routine the evaluator uses
                # (trimesh.contains with face-normal fallback). Optimization:
                # only the subset of vj that lies within mesh_i's AABB (plus
                # a d_hat margin) can possibly be inside; the rest are
                # guaranteed outside, so we skip the expensive ray-cast.
                inside_ji = np.zeros(len(vj), dtype=bool)
                mask_j_in_ai = np.all(
                    (vj >= ai0 - d_hat) & (vj <= ai1 + d_hat), axis=1
                )
                if mask_j_in_ai.any():
                    inside_ji[mask_j_in_ai] = _mesh_contains_points(
                        meshes[i], vj[mask_j_in_ai]
                    )

                inside_ij = np.zeros(len(vi), dtype=bool)
                mask_i_in_aj = np.all(
                    (vi >= aj0 - d_hat) & (vi <= aj1 + d_hat), axis=1
                )
                if mask_i_in_aj.any():
                    inside_ij[mask_i_in_aj] = _mesh_contains_points(
                        meshes[j], vi[mask_i_in_aj]
                    )

                # Signed distances: + gap, - depth
                sd_ji = np.where(inside_ji, -d_ji, d_ji)
                sd_ij = np.where(inside_ij, -d_ij, d_ij)

                # Pick the most "critical" sample (smallest signed distance,
                # i.e. deepest pen or smallest gap) across both directions.
                min_ji = sd_ji.min()
                min_ij = sd_ij.min()
                if min_ji <= min_ij:
                    k = int(np.argmin(sd_ji))
                    d_signed = float(sd_ji[k])
                    if inside_ji[k]:
                        # vj inside i: push vj outward (cl is on i's surface, exit direction)
                        n = cl_ji[k] - vj[k]
                        cp_on_a, cp_on_b = vj[k].copy(), cl_ji[k].copy()
                    else:
                        # vj outside i: gap; push j in +(vj - cl) direction
                        n = vj[k] - cl_ji[k]
                        cp_on_a, cp_on_b = cl_ji[k].copy(), vj[k].copy()
                else:
                    k = int(np.argmin(sd_ij))
                    d_signed = float(sd_ij[k])
                    if inside_ij[k]:
                        # vi inside j: push vi outward (cl on j's surface)
                        # We flip to keep `n` pointing i→j in the QP convention.
                        n = -(cl_ij[k] - vi[k])
                        cp_on_a, cp_on_b = cl_ij[k].copy(), vi[k].copy()
                    else:
                        # vi outside j: gap; push i in -(vi - cl) direction,
                        # equivalently j in +(vi - cl)
                        n = -(vi[k] - cl_ij[k])
                        cp_on_a, cp_on_b = vi[k].copy(), cl_ij[k].copy()

                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    n = centers[j] - centers[i]
                    nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n = n / nn

                ext_i = compute_extent(i, n)
                ext_j = compute_extent(j, -n)
                gap_needed = ds * (ext_i + ext_j)

                # Trigger if signed distance is below the safety band.
                # d_signed < 0 (penetration) is ALWAYS included.
                if d_signed < gap_needed + d_hat:
                    contacts.append((i, j, d_signed, n, ext_i, ext_j,
                                     cp_on_a, cp_on_b))

        return contacts

    # ── Pre-compute contact event schedule ──────────────────────────
    _soi_pre_t1 = time.time()
    event_scales = sorted(set(
        min(s, 1.0) for s in pair_first_contact.values() if s <= 1.0 + 0.1
    ))
    if not event_scales or event_scales[-1] < 1.0:
        event_scales.append(1.0)
    _soi_pre_elapsed += time.time() - _soi_pre_t1

    # Unified timing policy v1: everything from function entry to here ---
    # including the (vectorized) SOI-schedule precompute above and the BVH /
    # backend construction --- is setup_time; the continuation loop below is
    # solve_time; the reported `time` = setup + solve. This subsumes the
    # earlier back-dating fix that charged only the SOI precompute.
    _sync_backend()
    setup_time = time.perf_counter() - method_t0
    solve_t0 = time.perf_counter()
    diagnostics = []
    warm_cache = {}
    prev_sensitivity = None
    prev_active_bodies = None
    prev_dof_per_body = None

    # ── Audit: evaluator-equivalent check at every iteration ─────────
    # Uses trimesh.contains-based signed distance (same path as the
    # final evaluator) to detect pairs the solver's unsigned
    # `find_contacts` may have missed or mis-oriented.
    audit_log = []
    if audit:
        from mesh_collision import evaluate_world_collision_meshes

        def _audit_pairs_at(s_now):
            """Per-pair signed-distance scan. Returns (pen_pairs,
            max_pen, min_sd, pen_pair_set)."""
            ms = [build_mesh(i, s_now) for i in range(N)]
            stats = evaluate_world_collision_meshes(ms)
            # Identify *which* pairs are penetrating, for overlap analysis.
            pen_set = set()
            bounds = [m.bounds for m in ms]
            for i in range(N):
                amin, amax = bounds[i]
                for j in range(i + 1, N):
                    bmin, bmax = bounds[j]
                    if not (np.all(amin <= bmax) and np.all(bmin <= amax)):
                        continue
                    vj = np.asarray(ms[j].vertices, dtype=np.float64)
                    vi = np.asarray(ms[i].vertices, dtype=np.float64)
                    cl_i, d_ji, fi = trimesh.proximity.closest_point(ms[i], vj)
                    cl_j, d_ij, fj = trimesh.proximity.closest_point(ms[j], vi)
                    n_i = ms[i].face_normals[fi]
                    n_j = ms[j].face_normals[fj]
                    inside_ji = np.sum((vj - cl_i) * n_i, axis=1) < 0
                    inside_ij = np.sum((vi - cl_j) * n_j, axis=1) < 0
                    if np.any(inside_ji) or np.any(inside_ij):
                        pen_set.add((i, j))
            return stats.pen_pairs, stats.max_penetration, stats.min_signed_distance, pen_set

    # Analytical update state
    cached_contacts = None  # full detection result
    ds_retry_cap = float('inf')  # shrunk by the QP-failure retry; reset on accept
    steps_since_detection = 999
    last_dp = None

    def _snapshot_verts():
        """Gather world-space verts for every body at current (centers, rots, scale)."""
        return [world_verts(i, scale).astype(np.float32) for i in range(N)]

    def _snapshot_verts_at(s_now):
        return [world_verts(i, s_now).astype(np.float32) for i in range(N)]

    if trajectory_dumper is not None:
        # set_bodies wants a reference verts list; using the *full-scale*
        # sample so bounds captured at registration cover the real scene.
        trajectory_dumper.set_bodies(
            faces_list=[f.astype(np.int32) for f in mfaces],
            verts_list=_snapshot_verts_at(1.0),
        )
        # Pre-solve: the unresolved deep-penetration configuration at full
        # scale. This is what the user sees before S4R starts shrinking.
        trajectory_dumper.add_frame(
            step=-2, sub=0, scale=1.0, verts_list=_snapshot_verts_at(1.0),
        )
        # Start of progressive scaling (bodies just shrunk to scale_min).
        trajectory_dumper.add_frame(
            step=-1, sub=0, scale=float(scale), verts_list=_snapshot_verts(),
        )

    def _dump_frame(step_idx: int):
        if trajectory_dumper is None:
            return
        if step_idx != 0 and step_idx % dump_every != 0 and step_idx != max_steps - 1:
            return
        trajectory_dumper.add_frame(
            step=step_idx, sub=0, scale=float(scale), verts_list=_snapshot_verts(),
        )

    # Set True if the container walls (box_bounds) become mutually infeasible at
    # some scale: no translation keeps every body inside the box at that scale.
    # The container walls are HARD constraints, so we report infeasibility rather
    # than silently relaxing them (which would return a wall-violating "success").
    container_infeasible = False

    for step in range(max_steps):
        if scale >= 1.0 - 1e-6:
            break

                # applied in the previous iteration; the analytic cache update must
        # advance d by THAT inflation (paper Eq. cache_update), not by the
        # upcoming step's ds, which differs on truncated/extended strides.
        prev_ds = ds if step > 0 else 0.0
        # the QP-failure retry below halves ds; without this cap
        # the halving was overwritten here and never took effect.
        ds = min(ds_max, 1.0 - scale, ds_retry_cap)

        # ── Event-driven scheduling ──────────────────────────────────
        if adaptive_ds:
            next_events = [s for s in event_scales if s > scale + 1e-6]
            if next_events:
                ds_to_event = next_events[0] - scale
                if ds_to_event > ds_max:
                    ds = min(ds_to_event, 3.0 * ds_max)
            if step > 0 and not diagnostics[-1].get('had_contacts', True):
                ds = min(2.0 * ds_max, 1.0 - scale)
            # The event/skip extensions above reassign ds, so re-apply the
            # retry cap here or a halved step would be undone right away.
            ds = min(ds, ds_retry_cap)

        # ── Full detection or analytical update? ─────────────────────
        # with `>= revalidate_interval` the step right after a
        # detection (steps_since_detection=0) already fell through to the cache,
        # so interval=1 detected only every 2nd step. Use `- 1` so interval=M
        # detects every M steps (interval=1 => re-detect every step, as stated).
        if cached_contacts is None or steps_since_detection >= revalidate_interval - 1:
            # Full detection: bidirectional only at final step for precision
            is_final = (scale + ds >= 1.0 - 1e-6)
            _t_contact = time.time() if profile else 0.0
            contacts = find_contacts(scale, ds, bidirectional=is_final)
            if profile: _phase_times['contact'] += time.time() - _t_contact
            # Audit: evaluator's ground-truth pen at current scale
            if audit:
                truth_pen, truth_maxp, truth_minsd, truth_set = _audit_pairs_at(scale)
                solver_set = set((min(i, j), max(i, j)) for (i, j, *_) in contacts)
                missed = truth_set - solver_set
                print(f"[audit] step={step:3d} scale={scale:.4f} "
                      f"solver_contacts={len(contacts):4d}  "
                      f"evaluator_pen={truth_pen:4d}  "
                      f"missed_by_solver={len(missed):3d}  "
                      f"max_pen={truth_maxp:.4f}  "
                      f"min_sd={truth_minsd:.4f}")
                if missed:
                    print(f"        missed pairs: {sorted(missed)[:5]}"
                          f"{'...' if len(missed)>5 else ''}")
            cached_contacts = contacts
            steps_since_detection = 0
        else:
            # Analytical distance update: advance by the
            # PREVIOUS step's inflation prev_ds — the stride over which
            # last_dp was applied — matching d~^{k} = d~^{k-1} + n·Δp^{k-1}
            # - ds_{k-1}(e_i+e_j).
            if last_dp is not None and cached_contacts:
                cached_contacts = [
                    (i, j, d + n.dot(last_dp[j] - last_dp[i]) - prev_ds * (ei + ej),
                     n, ei, ej, cpi, cpj)
                    for i, j, d, n, ei, ej, cpi, cpj in cached_contacts
                ]
            contacts = cached_contacts
            steps_since_detection += 1

        # Optional attraction-only predictor: when no active contacts are
        # pushing, still pull bodies toward target_centers so the shape-
        # matching viz keeps moving. Closed-form minimiser of
        # ½‖Δ‖² + ½α‖c+Δ−t‖² is Δ = −α/(1+α)·(c−t); we cap by a fraction of
        # the current scale step so velocity stays bounded.
        def _apply_attraction_only(max_frac=1.0):
            if target_centers is None or attraction_alpha is None:
                return False
            raw = attraction_alpha(scale) if callable(attraction_alpha) else attraction_alpha
            a = float(np.asarray(raw).max())
            if a <= 1e-8:
                return False
            err = centers - target_centers
            delta = -a / (1.0 + a) * err
            # cap the per-step motion at max_frac * ds in body-extent units
            cap = max_frac * ds * float(max_extent_all + 1e-9)
            norms = np.linalg.norm(delta, axis=1, keepdims=True)
            scale_dn = np.minimum(1.0, cap / np.clip(norms, 1e-9, None))
            delta = delta * scale_dn
            centers[:] += delta
            return True

        if not contacts:
            _apply_attraction_only()
            scale += ds
            ds_retry_cap = float('inf')  # step accepted
            diagnostics.append({'step': step, 'scale': scale, 'ds': ds,
                               'n_active': 0, 'n_pen': 0, 'had_contacts': False})
            _dump_frame(step)
            continue

        # Filter to contacts that actually need pushing
        active = [(i, j, d, n, ei, ej, cpi, cpj)
                  for i, j, d, n, ei, ej, cpi, cpj in contacts
                  if ds * (ei + ej) + d_hat - d > 0]
        if not active:
            n_pen_noactive = sum(1 for c in contacts if c[2] < -1e-6)
            _apply_attraction_only()
            scale += ds
            ds_retry_cap = float('inf')  # step accepted
            diagnostics.append({'step': step, 'scale': scale, 'ds': ds,
                               'n_active': 0, 'n_pen': n_pen_noactive,
                               'had_contacts': True})
            _dump_frame(step)
            continue

        n_active = len(active)
        # Penetrating pairs at the current scale (d_signed < 0) — written
        # to diagnostics so the caller can plot pen-vs-step curves.
        n_pen = sum(1 for c in contacts if c[2] < -1e-6)

        # ── Identify active bodies (contact graph sparsity) ──────────
        # With box_bounds, EVERY body must carry a wall constraint every step
        # (a body with no contact still inflates with the scale and would poke
        # through a wall if left out of the QP), so sparsity is disabled here.
        if contact_sparsity and box_bounds is None:
            active_set = set()
            for (i, j, *_) in active:
                active_set.add(i)
                active_set.add(j)
            active_bodies = sorted(active_set)
            body_map = {b: idx for idx, b in enumerate(active_bodies)}
            n_bodies_qp = len(active_bodies)
        else:
            active_bodies = list(range(N))
            body_map = {b: b for b in range(N)}
            n_bodies_qp = N

        dof_per_body = 6 if enable_rotation else 3
        n_vars = dof_per_body * n_bodies_qp

        # ── Build QP ─────────────────────────────────────────────────
        # Objective: min (1/2) x^T P x
        # Optional attraction to target_centers: adds α/2 ‖c + Δp − t‖² to the
        # objective, which expands to α on P's translation diagonal and
        # α(c−t) on q (translation slice). α can be a scalar, (N,) array,
        # or a callable α(scale)→scalar|(N,) array. Only used when
        # target_centers is supplied.
        alpha_tr = None
        if target_centers is not None and attraction_alpha is not None:
            raw = attraction_alpha(scale) if callable(attraction_alpha) else attraction_alpha
            raw = np.asarray(raw, dtype=np.float64)
            if raw.ndim == 0:
                alpha_tr = np.full(N, float(raw))
            else:
                alpha_tr = raw.reshape(-1)

        if enable_rotation:
            diag = []
            for b in active_bodies:
                a_b = float(alpha_tr[b]) if alpha_tr is not None else 0.0
                diag.extend([1.0 + a_b, 1.0 + a_b, 1.0 + a_b])  # translation
                rw = rotation_weight
                diag.extend([rw, rw, rw])  # rotation weight
            P = sp.diags(diag, format='csc')
        elif alpha_tr is not None:
            diag = []
            for b in active_bodies:
                a_b = float(alpha_tr[b])
                diag.extend([1.0 + a_b, 1.0 + a_b, 1.0 + a_b])
            P = sp.diags(diag, format='csc')
        else:
            P = sp.eye(n_vars, format='csc')

        q = np.zeros(n_vars)
        if alpha_tr is not None:
            for idx, b in enumerate(active_bodies):
                a_b = float(alpha_tr[b])
                if a_b > 0.0:
                    q[dof_per_body * idx: dof_per_body * idx + 3] = a_b * (
                        centers[b] - target_centers[b])

        # Constraints
        A = np.zeros((n_active, n_vars))
        l = np.zeros(n_active)

        for ci, (i, j, d_curr, n_ij, ext_i, ext_j, cp_i, cp_j) in enumerate(active):
            b = ds * (ext_i + ext_j) + d_hat - d_curr
            ii = body_map[i]
            jj = body_map[j]

            # Translation part: n · (Δp_j - Δp_i) ≥ b
            A[ci, dof_per_body * ii: dof_per_body * ii + 3] = -n_ij
            A[ci, dof_per_body * jj: dof_per_body * jj + 3] = n_ij

            if enable_rotation:
                # Moment arms: y_i = cp_on_i - center_i, y_j = cp_on_j - center_j
                y_i = cp_i - centers[i]
                y_j = cp_j - centers[j]
                # Cross products: (y × n) gives the rotation-to-displacement coupling
                cross_i = np.cross(y_i, n_ij)
                cross_j = np.cross(y_j, n_ij)
                # Rotation part: (y_j × n)·ω_j - (y_i × n)·ω_i
                A[ci, dof_per_body * ii + 3: dof_per_body * ii + 6] = -cross_i
                A[ci, dof_per_body * jj + 3: dof_per_body * jj + 6] = cross_j

            l[ci] = b

        # Add rotation clamp as box constraints
        if enable_rotation:
            n_box = 3 * n_bodies_qp  # one bound per rotation component
            A_box = np.zeros((n_box, n_vars))
            l_box = np.full(n_box, -max_omega)
            u_box = np.full(n_box, max_omega)
            for idx in range(n_bodies_qp):
                for d in range(3):
                    row = idx * 3 + d
                    A_box[row, dof_per_body * idx + 3 + d] = 1.0
            # Stack: original constraints + box constraints
            A_full = np.vstack([A, A_box])
            l_full = np.concatenate([l, l_box])
            u_full = np.concatenate([np.full(n_active, np.inf), u_box])
        else:
            A_full = A
            l_full = l
            u_full = np.full(n_active, np.inf)

        # ── Box-confinement (container wall) constraints ──────────────────
        # For every body in this QP, on each world axis a, keep the body inside
        # the container at the end of this step (scale s_next = scale + ds):
        #   lo[a] + s_next*sup_minus[i,a] <= center_i[a] + Δp_i[a] <= hi[a] - s_next*sup_plus[i,a]
        # i.e. a two-sided bound on the single translational variable Δp_i[a].
        if box_bounds is not None:
            s_next = scale + ds
            n_wall = 3 * n_bodies_qp
            A_wall = np.zeros((n_wall, n_vars))
            l_wall = np.empty(n_wall)
            u_wall = np.empty(n_wall)
            for orig_b in active_bodies:
                idx = body_map[orig_b]
                for a in range(3):
                    row = idx * 3 + a
                    A_wall[row, dof_per_body * idx + a] = 1.0
                    ci_a = centers[orig_b][a]
                    u_wall[row] = box_hi[a] - s_next * box_sup_plus[orig_b][a] - ci_a
                    l_wall[row] = box_lo[a] + s_next * box_sup_minus[orig_b][a] - ci_a
            # If the box is too tight at this scale the two-sided bound inverts
            # (l > u): there is NO body position that stays inside the container,
            # so the hard-constrained QP is genuinely infeasible. We report this
            # (no wall-violating "success") and stop the continuation — the box
            # constraint cannot be satisfied at a scale below the target, so no
            # continuous scale path reaches a collision-free in-box full-scale
            # state. (Tolerance guards against float round-off at equality.)
            if np.any(l_wall > u_wall + 1e-9):
                container_infeasible = True
                if verbose:
                    print(f"  Step {step + 1}: container infeasible at "
                          f"scale={scale + ds:.3f} (walls cannot bound all "
                          f"bodies inside the box). Stopping.")
                break
            A_full = np.vstack([A_full, A_wall])
            l_full = np.concatenate([l_full, l_wall])
            u_full = np.concatenate([u_full, u_wall])

        # ── Articulated-joint (hinge) equality constraints ────────────────
        # Each joint (i, j, anchor_i, anchor_j) pins a point on body i to a
        # point on body j (a hinge/ball anchor). anchor_* are full-scale world
        # offsets from each centroid at the current rotation; they scale with s
        # and rotate with the per-step ω. Keeping the two anchors coincident at
        # s_next is a 3-row linear equality (l==u) in (Δp, ω) of both bodies:
        #   Δp_i - Δp_j - s*[anchor_i]_× ω_i + s*[anchor_j]_× ω_j
        #        = (p_j - p_i) + s*(anchor_j - anchor_i).
        # Requires the 6-DOF solve so links can swing about the hinge to
        # separate; demonstrates resolving interpenetration on articulated
        # bodies without breaking their joints.
        if joints and enable_rotation:
            def _skew(v):
                return np.array([[0.0, -v[2], v[1]],
                                 [v[2], 0.0, -v[0]],
                                 [-v[1], v[0], 0.0]])
            # The hinge anchors are pinned at their FULL-SCALE offsets (coeff 1,
            # NOT scaled by s): a link shrinks toward its own centroid for
            # collision avoidance, but its joint point stays fixed relative to
            # the centroid so the articulation topology is preserved throughout
            # the continuation and exactly satisfied at s=1.
            j_rows = []
            j_rhs = []
            for (bi, bj, a_i, a_j) in joints:
                if bi not in body_map or bj not in body_map:
                    continue
                ii = body_map[bi]; jj = body_map[bj]
                a_iw = rots[bi] @ np.asarray(a_i, dtype=np.float64)
                a_jw = rots[bj] @ np.asarray(a_j, dtype=np.float64)
                rhs = (centers[bj] - centers[bi]) + (a_jw - a_iw)
                block = np.zeros((3, n_vars))
                block[:, dof_per_body * ii: dof_per_body * ii + 3] = np.eye(3)
                block[:, dof_per_body * jj: dof_per_body * jj + 3] = -np.eye(3)
                block[:, dof_per_body * ii + 3: dof_per_body * ii + 6] = -_skew(a_iw)
                block[:, dof_per_body * jj + 3: dof_per_body * jj + 6] = _skew(a_jw)
                j_rows.append(block); j_rhs.append(rhs)
            if j_rows:
                A_j = np.vstack(j_rows); b_j = np.concatenate(j_rhs)
                A_full = np.vstack([A_full, A_j])
                l_full = np.concatenate([l_full, b_j])
                u_full = np.concatenate([u_full, b_j])

        A_sparse = sp.csc_matrix(A_full)

        # ── Predictor: use ODE sensitivity from previous step ─────────
        # dq*/ds = -A_active^+ (∂d/∂s) where ∂d/∂s = -(ext_i + ext_j)
        # If active set matches previous step, predict directly
        x0 = np.zeros(n_vars)
        used_predictor = False

        if (prev_sensitivity is not None and prev_active_bodies is not None
                and dof_per_body == prev_dof_per_body
                and len(prev_sensitivity) == n_vars
                and prev_active_bodies == set(active_bodies)):
            # Exact same active set → full ODE prediction
            x0 = ds * prev_sensitivity
            used_predictor = True
        elif (prev_sensitivity is not None and prev_active_bodies is not None
                and dof_per_body == prev_dof_per_body
                and len(prev_sensitivity) == n_vars):
            # Active set changed but same dimension → use sensitivity as hint
            # Scale by overlap ratio
            overlap = prev_active_bodies & set(active_bodies)
            if len(overlap) > len(active_bodies) * 0.5:
                x0 = ds * prev_sensitivity * 0.5  # dampened prediction
                used_predictor = True
            else:
                for idx, b in enumerate(active_bodies):
                    if b in warm_cache:
                        cached = warm_cache[b]
                        copy_len = min(len(cached), dof_per_body)
                        x0[dof_per_body * idx: dof_per_body * idx + copy_len] = cached[:copy_len]
        else:
            # Fall back to warm cache
            for idx, b in enumerate(active_bodies):
                if b in warm_cache:
                    cached = warm_cache[b]
                    copy_len = min(len(cached), dof_per_body)
                    x0[dof_per_body * idx: dof_per_body * idx + copy_len] = cached[:copy_len]

        # ── Solve QP ──────────────────────────────────────────────────
        t_qp = time.time()
        qp_iters = 0

        if use_dual and not enable_rotation:
            # Dual QP: solve in contact force space (dimension = n_active)
            P_diag = np.ones(n_vars)  # P = I for translation-only
            dual_result = solve_dual_qp(A, l, P_diag, n_vars)
            if dual_result is not None:
                result_x = dual_result[0]
                qp_iters = dual_result[2]
                qp_solved = True
            else:
                qp_solved = False
        elif use_dual and enable_rotation:
            # Dual QP with weighted P
            P_diag = np.array([
                val for b in active_bodies
                for val in [1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight]
            ])
            dual_result = solve_dual_qp(A, l, P_diag, n_vars)
            if dual_result is not None:
                result_x = dual_result[0]
                qp_iters = dual_result[2]
                qp_solved = True
            else:
                qp_solved = False
        else:
            # Primal QP (original)
            _t_qp = time.time() if profile else 0.0
            solver = osqp.OSQP()
            solver.setup(P, q, A_sparse, l_full, u_full,
                         verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                         max_iter=4000, polish=True, warm_starting=True)
            solver.warm_start(x=x0)
            result = solver.solve()
            if profile: _phase_times['qp'] += time.time() - _t_qp
            qp_solved = result.info.status in ('solved', 'solved_inaccurate')
            if qp_solved:
                result_x = result.x
                qp_iters = result.info.iter

        qp_time = time.time() - t_qp

        if qp_solved:
            x = result_x.reshape(n_bodies_qp, dof_per_body)
            dp = x[:, :3]

            # Cache solution for warm-starting next step
            for idx, b in enumerate(active_bodies):
                warm_cache[b] = x[idx].copy()

            # ── Compute sensitivity dq*/ds for predictor ─────────────
            # Active set: constraints where QP solution is binding
            # Use the contact constraints (first n_active rows of A_full)
            # ∂d/∂s for each active contact = -(ext_i + ext_j)
            dd_ds = np.array([-(ei + ej) for _, _, _, _, ei, ej, _, _ in active])
            # A_active = first n_active rows of A (contact part only)
            A_contact = A[:n_active]
            # Identify binding constraints: where residual ≈ 0
            residuals = A_contact @ result_x - l[:n_active]
            binding = residuals < 1e-4  # approximately active
            if binding.any():
                A_bind = A_contact[binding]
                dd_bind = dd_ds[binding]
                # dq*/ds = -A_bind^+ @ dd_bind = -A_bind^T (A_bind A_bind^T)^{-1} dd_bind
                try:
                    G = A_bind @ A_bind.T  # |binding| × |binding|
                    dλ = np.linalg.solve(G + 1e-8 * np.eye(G.shape[0]), dd_bind)
                    prev_sensitivity = -A_bind.T @ dλ
                except np.linalg.LinAlgError:
                    prev_sensitivity = None
            else:
                prev_sensitivity = None
            prev_active_bodies = set(active_bodies)
            prev_dof_per_body = dof_per_body

            # Apply translation
            dp_full = np.zeros((N, 3))
            for idx, b in enumerate(active_bodies):
                centers[b] += dp[idx]
                dp_full[b] = dp[idx]
            last_dp = dp_full

            # do NOT apply Delta-p to the cache here. The analytic
            # branch already applies `last_dp` (this same Delta-p) plus the
            # inflation term at the start of the next step, so updating here too
            # double-counted the QP displacement. The cache equation
            #   d~^{k+1} = d~^k + n.Delta-p - ds(e_i+e_j)
            # is now applied exactly once per step (in the analytic branch).

            # Apply rotation
            max_rot_applied = 0.0
            if enable_rotation:
                omega = x[:, 3:6]
                for idx, b in enumerate(active_bodies):
                    w = omega[idx]
                    nw = np.linalg.norm(w)
                    max_rot_applied = max(max_rot_applied, nw)
                    if nw > 1e-12:
                        rots[b] = RotLib.from_rotvec(w).as_matrix() @ rots[b]

            # ── Step B: Rotation manifold optimization ────────────────
            # After translation QP, optimize rotations on SO(3) for each body.
            # Reuses the same rotation-free, scale-1 BVH cache as the
            # translation oracle (built once on first contact query).
            rot_total = 0.0
            if enable_rotation and n_active >= 3:
                if not _prebuilt_bvh:
                    _init_prebuilt_fcl()
                rot_total = optimize_rotations_on_manifold(
                    centers, rots, nfs, mverts, mfaces, N, d_hat, scale + ds,
                    contact_backend=contact_backend,
                    bvh_cache=_prebuilt_bvh,
                    model_verts_cache=_prebuilt_M)

            scale += ds
            ds_retry_cap = float('inf')  # step accepted
            _dump_frame(step)

            diagnostics.append({
                'step': step, 'scale': scale, 'ds': ds,
                'n_active': n_active, 'n_pen': n_pen,
                'had_contacts': True,
                'qp_time_ms': qp_time * 1000,
                'qp_iters': qp_iters,
                'max_disp': float(np.max(np.linalg.norm(dp, axis=1))),
                'max_rot': max_rot_applied,
                'n_vars': n_vars,
                'used_predictor': used_predictor,
            })

            # Audit: ground-truth pen after applying QP displacement + inflation
            if audit:
                post_pen, post_maxp, post_minsd, post_set = _audit_pairs_at(scale)
                new_pen = post_set - (truth_set if 'truth_set' in dir() else set())
                print(f"        post-step: scale={scale:.4f}  "
                      f"pen_after={post_pen:3d}  max_pen={post_maxp:.4f}  "
                      f"new_pen_this_step={len(new_pen)}")
                audit_log.append({
                    'step': step, 'scale_before': scale - ds, 'scale_after': scale,
                    'solver_contacts': n_active,
                    'evaluator_pen_before': truth_pen,
                    'evaluator_pen_after': post_pen,
                    'max_pen_before': truth_maxp,
                    'max_pen_after': post_maxp,
                    'missed_by_solver_before': len(missed) if 'missed' in dir() else 0,
                    'new_pen_this_step': len(new_pen),
                })

            if verbose and (step + 1) % 5 == 0:
                rmsd = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1))))
                rot_str = f" max_rot={max_rot_applied:.4f}" if enable_rotation else ""
                print(f"  Step {step + 1}: scale={scale:.3f} contacts={n_active} "
                      f"pen={n_pen} "
                      f"max_push={np.max(np.linalg.norm(dp, axis=1)):.4f} "
                      f"RMSD={rmsd:.4f}{rot_str}")
        else:
            if enable_rotation:
                # Fallback: try translation-only
                result_fb = solve_s4r_qp_step_translation_only(
                    N, active, ds, d_hat, centers, body_map, active_bodies,
                    contact_sparsity)
                if result_fb is not None:
                    centers += result_fb
                    scale += ds
                    ds_retry_cap = float('inf')  # step accepted
                    _dump_frame(step)
                    diagnostics.append({
                        'step': step, 'scale': scale, 'ds': ds,
                        'n_active': n_active, 'n_pen': n_pen,
                        'had_contacts': True,
                        'fallback': True,
                    })
                    continue

            # Reduce ds and retry
            ds *= 0.5
            ds_retry_cap = ds
            # Force a full detection on the retry: the cache was already
            # propagated for this failed attempt, and propagating it again
            # would advance it by an inflation that never happened.
            steps_since_detection = 999
            if ds < 1e-6:
                if verbose:
                    print(f"  Step {step + 1}: QP infeasible, ds too small. Stopping.")
                break
            continue

    # ── Tail refinement: correction QPs at scale=1 with ds=0 ─────────
    # Residual linearization error can leave a handful of pairs with
    # d_signed < 0 at scale=1. Run correction QPs with ds=0 using the
    # same active-contact set as the main loop (penetrating AND near-
    # contact pairs), so pushing one pair does not flip a neighbor.
    # Stop as soon as no penetrating pair remains, or when max_pen
    # stagnates / rebounds.
    max_tail_iters = int(os.environ.get('S4R_MAX_TAIL_ITERS', 20))
    # Stagnation check: stop when max_pen hasn't improved for this many
    # consecutive iterations. Default 3 (matches tab:main). Bump higher
    # for dense sphere-packed scenes where max_pen oscillates between
    # competing contacts even while pen_pair_count steadily decreases.
    stagnation_cap = int(os.environ.get('S4R_TAIL_STAGNATION_CAP', 3))
    # E3 safe-margin variant (default 0.0 = published behavior, bit-identical:
    # selection sd<0 and deficit=-sd). When >0, the tail keeps iterating until
    # every near pair clears the target margin (its correction QP already
    # pushes toward the full d_hat clearance; only this exit test stopped at
    # evaluator-level pen=0). 'deficit' below = target - sd.
    tail_margin = float(os.environ.get('S4R_TAIL_TARGET_MARGIN', '0.0'))
    prev_max_pen = float('inf')
    prev_pen_pairs = None
    stagnation = 0
    _t_tail_start = time.time() if profile else 0.0
    tail_stop_reason = 'iter_cap'
    for tail in range(max_tail_iters):
        contacts_tail = find_contacts_margin(scale, 0.0, tail_margin)
        pen_pairs_tail = [c for c in contacts_tail if c[2] < tail_margin]
        if not pen_pairs_tail:
            tail_stop_reason = 'feasible'
            if audit:
                print(f"[TAIL] iter={tail} pen=0 (margin target {tail_margin}), converged")
            break

        max_pen_now = max(tail_margin - c[2] for c in pen_pairs_tail)
        n_pen_now = len(pen_pairs_tail)
        # Stagnation = neither max_pen nor pen_pair_count has dropped.
        # This protects against the dense-scene failure mode where
        # max_pen oscillates between contacts but pair count is still
        # falling.
        pair_dropped = prev_pen_pairs is None or n_pen_now < prev_pen_pairs
        if max_pen_now >= prev_max_pen - 1e-5 and not pair_dropped:
            stagnation += 1
            if stagnation >= stagnation_cap:
                tail_stop_reason = 'tail_stagnation'
                if audit:
                    print(f"[TAIL] iter={tail} stagnated at max_pen={max_pen_now:.4f} pairs={n_pen_now}, stopping")
                break
        else:
            stagnation = 0
        prev_max_pen = max_pen_now
        prev_pen_pairs = n_pen_now

        if audit:
            print(f"[TAIL] iter={tail} scale={scale:.4f} "
                  f"pen_pairs={len(pen_pairs_tail)} "
                  f"active_total={len(contacts_tail)} "
                  f"max_pen={max_pen_now:.4f}")

        # Build correction QP using ALL active contacts (pen + near-contact)
        # so pushing a pen pair doesn't flip a near-contact neighbor.
        if contact_sparsity and box_bounds is None:
            active_set_t = set()
            for (i, j, *_) in contacts_tail:
                active_set_t.add(i); active_set_t.add(j)
            active_bodies_t = sorted(active_set_t)
            body_map_t = {b: idx for idx, b in enumerate(active_bodies_t)}
        else:
            active_bodies_t = list(range(N))
            body_map_t = {b: b for b in range(N)}
        n_b_t = len(active_bodies_t)
        n_vars_t = 3 * n_b_t
        n_c_t = len(contacts_tail)

        P_t = sp.eye(n_vars_t, format='csc')
        q_t = np.zeros(n_vars_t)
        A_t = np.zeros((n_c_t, n_vars_t))
        l_t = np.zeros(n_c_t)
        for ci, (i, j, d_s, n_ij, ext_i, ext_j, _, _) in enumerate(contacts_tail):
            b_t = d_hat - d_s  # ds = 0; negative for far gaps (constraint inactive)
            ii = body_map_t[i]; jj = body_map_t[j]
            A_t[ci, 3*ii:3*ii+3] = -n_ij
            A_t[ci, 3*jj:3*jj+3] =  n_ij
            l_t[ci] = b_t

        A_t_full = A_t; l_t_full = l_t; u_t_full = np.full(n_c_t, np.inf)
        # Box-confinement walls also apply during tail refinement (at s=1), so
        # the residual-cleanup push cannot eject a body from the container.
        if box_bounds is not None:
            n_wall_t = 3 * n_b_t
            A_wall_t = np.zeros((n_wall_t, n_vars_t))
            l_wall_t = np.empty(n_wall_t); u_wall_t = np.empty(n_wall_t)
            for orig_b in active_bodies_t:
                idx = body_map_t[orig_b]
                for a in range(3):
                    row = idx * 3 + a
                    A_wall_t[row, 3 * idx + a] = 1.0
                    ci_a = centers[orig_b][a]
                    u_wall_t[row] = box_hi[a] - box_sup_plus[orig_b][a] - ci_a
                    l_wall_t[row] = box_lo[a] + box_sup_minus[orig_b][a] - ci_a
            # Hard walls: an inverted bound at s=1 means the container cannot
            # hold every body — report infeasible instead of relaxing.
            if np.any(l_wall_t > u_wall_t + 1e-9):
                container_infeasible = True
                if verbose:
                    print(f"  Tail: container infeasible at s=1 "
                          f"(walls cannot bound all bodies inside the box).")
                break
            A_t_full = np.vstack([A_t, A_wall_t])
            l_t_full = np.concatenate([l_t, l_wall_t])
            u_t_full = np.concatenate([u_t_full, u_wall_t])

        solver_t = osqp.OSQP()
        solver_t.setup(P_t, q_t, sp.csc_matrix(A_t_full), l_t_full,
                       u_t_full,
                       verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                       max_iter=8000, polish=True)
        res_t = solver_t.solve()
        if res_t.info.status not in ('solved', 'solved_inaccurate'):
            if audit:
                print(f"[TAIL] iter={tail} QP status={res_t.info.status}, stopping")
            break

        # Damp the step to stay within the linearization radius.
        # Use alpha = min(1, d_hat / max_disp_predicted) so no body moves
        # more than d_hat per iteration (the scale at which the linear
        # constraint is accurate).
        dp_t = res_t.x.reshape(n_b_t, 3)
        max_disp_pred = float(np.max(np.linalg.norm(dp_t, axis=1)))
        alpha = 1.0 if max_disp_pred <= d_hat else (d_hat / max_disp_pred)

        for idx, b in enumerate(active_bodies_t):
            centers[b] += alpha * dp_t[idx]

        diagnostics.append({
            'step': 'tail', 'tail_iter': tail,
            'pen_pairs': len(pen_pairs_tail),
            'active_total': len(contacts_tail),
            'max_pen_before': max_pen_now,
            'max_disp_pred': max_disp_pred,
            'alpha': alpha,
        })

    if profile:
        _phase_times['tail'] = time.time() - _t_tail_start

    # ── Penalty cleanup pass ────────────────────────────────────────────
    # The QP-based tail can stagnate on extreme-density scenes where
    # pushing one pair micro-collides another. For each residual pen
    # pair, apply a direct mass-balanced Jacobi push of size
    # (|depth| + epsilon) along the contact normal — AVBD-style overshoot,
    # applied ONLY to the stuck pairs so the global RMSD barely moves.
    # Iterate K times with re-detection. Disabled with
    # S4R_DISABLE_PENALTY_CLEANUP=1.
    #
    # Note: this fallback is NOT the QP-based tail-refinement above; it is
    # a simple pairwise penalty push retained only for extreme-density
    # corner cases where the tail QP stagnates. It is disabled in every
    # result reported in the paper. `cleanup_iters_used` (returned below)
    # records the number of cleanup iterations that actually ran (0 == the pass
    # was entered but the scene was already pen-free, i.e. no push applied;
    # None == the pass was disabled via S4R_DISABLE_PENALTY_CLEANUP=1), so that
    # zero-trigger claim is self-verifiable from each run's JSON.
    cleanup_iters_used = None
    if not int(os.environ.get('S4R_DISABLE_PENALTY_CLEANUP', 0)):
        cleanup_iters_used = 0
        max_cleanup_iters = int(os.environ.get('S4R_MAX_CLEANUP_ITERS', 200))
        cleanup_eps = float(os.environ.get('S4R_CLEANUP_EPS', d_hat * 0.1))
        cleanup_eps_init = cleanup_eps
        prev_pen_n = None
        stagnant = 0
        gauss_seidel = False  # switch to Gauss-Seidel after Jacobi stalls
        for ci in range(max_cleanup_iters):
            cleanup_contacts = find_contacts(scale, 0.0, bidirectional=True)
            cleanup_pen = [c for c in cleanup_contacts if c[2] < 0.0]
            if not cleanup_pen:
                if audit:
                    print(f"[CLEANUP] iter={ci} pen=0, converged")
                break
            # A residual pen pair exists AND we are about to apply a push:
            # count this as one cleanup iteration that actually did work.
            cleanup_iters_used += 1
            n_pen_now = len(cleanup_pen)
            # Stagnation heuristic: bump eps and switch Jacobi→Gauss-Seidel
            # so we don't undershoot. Cycle-detection (pen grew) shrinks eps.
            if prev_pen_n is not None:
                if n_pen_now > prev_pen_n:
                    cleanup_eps *= 0.5
                    stagnant = 0
                elif n_pen_now == prev_pen_n:
                    stagnant += 1
                    if stagnant >= 2:
                        cleanup_eps = min(cleanup_eps * 1.5, d_hat * 2.0)
                        gauss_seidel = True
                else:
                    stagnant = 0
            prev_pen_n = n_pen_now
            if audit:
                print(f"[CLEANUP] iter={ci} pen_pairs={n_pen_now} "
                      f"eps={cleanup_eps:.5f} mode={'GS' if gauss_seidel else 'J'}")
            if gauss_seidel:
                # Apply pair-by-pair with re-detection between iterations
                # (not strictly GS — true GS would re-detect every pair,
                # which is expensive — but we re-query after each cleanup
                # iter so consecutive iters see updated state).
                # Within one iter, deepest-first order to prioritise the
                # worst pen pairs.
                cleanup_pen.sort(key=lambda c: c[2])  # most-negative first
            push = np.zeros_like(centers)
            for (i, j, d_signed, n_ij, _ei, _ej, _cpa, _cpb) in cleanup_pen:
                magnitude = (-d_signed) + cleanup_eps
                if gauss_seidel:
                    # Apply each push immediately to centers (sequential).
                    centers[i] -= 0.5 * magnitude * n_ij
                    centers[j] += 0.5 * magnitude * n_ij
                else:
                    push[i] -= 0.5 * magnitude * n_ij
                    push[j] += 0.5 * magnitude * n_ij
            if not gauss_seidel:
                centers += push

    _sync_backend()
    solve_time = time.perf_counter() - solve_t0
    method_total_time = setup_time + solve_time

    # ── Final evaluation ──────────────────────────────────────────────
    eval_t0 = time.perf_counter()
    meshes = [build_mesh(i) for i in range(N)]
    from mesh_collision import evaluate_world_collision_meshes
    stats = evaluate_world_collision_meshes(meshes)
    rmsd = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1))))
    evaluation_time = time.perf_counter() - eval_t0

    if profile:
        _phase_times['other'] = max(0.0, solve_time - sum(_phase_times.values()))

    if trajectory_dumper is not None:
        # Final frame at whatever scale we stopped at.
        trajectory_dumper.add_frame(
            step=(step if 'step' in dir() else 0) + 1,
            sub=0,
            scale=float(scale),
            verts_list=_snapshot_verts(),
        )
        out_path = trajectory_dumper.write()
        print(f"  Trajectory dumped to {out_path}  "
              f"({len(trajectory_dumper._frames)} frames)")

    return {
        "pen": stats.pen_pairs,
        "max_pen": stats.max_penetration,
        "rmsd": rmsd,
        "timing_policy": "per_scene_setup_plus_solve_v1",
        "setup_time": setup_time,
        "solve_time": solve_time,
        "method_total_time": method_total_time,
        "evaluation_time": evaluation_time,
        "solver_internal_time": solve_time,
        "time": method_total_time,
        "scale": scale,
        "steps": step if 'step' in dir() else 0,
        # Number of penalty-cleanup iterations that actually applied a push.
        # 0  => fallback enabled but never needed (the paper's main-table claim);
        # None => fallback disabled via S4R_DISABLE_PENALTY_CLEANUP=1.
        "cleanup_iters_used": cleanup_iters_used,
        # Auditability (same fields the baseline wrappers export): why the
        # tail loop ended, and whether it ended on its own feasibility test.
        "stop_reason": tail_stop_reason,
        "native_converged": tail_stop_reason == 'feasible',
        "final_centers": centers.copy(),
        "final_rotations": np.stack(rots, axis=0),
        "container_infeasible": container_infeasible,
        "diagnostics": diagnostics,
        "audit_log": audit_log,
        "phase_times": _phase_times if profile else None,
    }


def optimize_rotations_on_manifold(centers, rots, nfs, mverts, mfaces, N,
                                     d_hat, scale, max_rot_iters=3,
                                     contact_backend='fcl',
                                     bvh_cache=None,
                                     model_verts_cache=None):
    """Step B of alternating minimization: optimize rotation for each body on SO(3).

    For each body i with neighbors, find rotation that maximizes minimum distance
    to neighbors while minimizing deviation from current rotation.

    Uses geodesic gradient descent on SO(3): R_new = exp(α ω) @ R_old
    where ω = Σ_j (torque from contact j).

    Two backends:
      - 'fcl' / 'fcl_prebuilt' : build BVHs ONCE on rotation-FREE model verts
        ``nfs[i] * mverts[i]`` and reuse via ``fcl.Transform(R_i, c_i/s)``.
        Rotation updates are then a Transform swap (free), with no BVH rebuild.
        Uses change-of-variable u = x/s so the BVHs stay unit-scale across the
        S4R schedule.
      - 'trimesh' : legacy Python BVH + ``trimesh.proximity.closest_point``.
        Kept as fallback for environments without python-fcl.

    `bvh_cache` / `model_verts_cache` may be passed by the outer solver
    (``solve_s4r_qp``) to skip the per-call BVH build. They are mutable lists
    that this function populates on first use and reads thereafter.
    """
    use_fcl = contact_backend in ('fcl', 'fcl_prebuilt')
    if use_fcl:
        try:
            import fcl  # noqa: F401
        except ImportError:
            use_fcl = False

    if use_fcl:
        return _optimize_rotations_fcl(centers, rots, nfs, mverts, mfaces, N,
                                        d_hat, scale, max_rot_iters,
                                        bvh_cache=bvh_cache,
                                        model_verts_cache=model_verts_cache)
    return _optimize_rotations_trimesh(centers, rots, nfs, mverts, mfaces, N,
                                        d_hat, scale, max_rot_iters)


def _optimize_rotations_fcl(centers, rots, nfs, mverts, mfaces, N,
                             d_hat, scale, max_rot_iters,
                             bvh_cache=None, model_verts_cache=None):
    """FCL-backed rotation optimization.

    Change-of-variable trick: in u-space (u = x/s) a body at world scale `s`
    is unit-scale at translation ``centers[i]/s``. The body's rotation is
    applied identically in both spaces (R commutes with uniform scale), so a
    single unit-scale BVH built on ``nfs[i] * mverts[i]`` works for every
    scale step — only the Transform changes when rotation/translation update.

    Distances/contact points come back in u-space; multiply by `s` to
    recover world units. Normals are scale-invariant.
    """
    import fcl

    inv_s = 1.0 / scale

    # BVH cache: build once on rotation-free model verts ``nfs[i]*mverts[i]``
    # and reuse across every call to this function for the rest of the solve.
    # The caller passes mutable lists so the BVHs survive across scale steps.
    if bvh_cache is not None and len(bvh_cache) == N:
        bvh_models = bvh_cache
        model_verts_scaled = model_verts_cache
    else:
        bvh_models = [] if bvh_cache is None else bvh_cache
        model_verts_scaled = [] if model_verts_cache is None else model_verts_cache
        bvh_models.clear() if hasattr(bvh_models, 'clear') else None
        model_verts_scaled.clear() if hasattr(model_verts_scaled, 'clear') else None
        for i in range(N):
            Vi = (nfs[i] * mverts[i]).astype(np.float64)
            model_verts_scaled.append(Vi)
            faces_i = mfaces[i].astype(np.int32)
            m = fcl.BVHModel()
            m.beginModel(len(Vi), len(faces_i))
            m.addSubModel(Vi, faces_i)
            m.endModel()
            bvh_models.append(m)

    def _world_aabb_u(i):
        """AABB of body i in u-space at the current rots/centers."""
        Vw = (rots[i] @ model_verts_scaled[i].T).T + centers[i] * inv_s
        v_min = Vw.min(axis=0); v_max = Vw.max(axis=0)
        return (float(v_min[0]), float(v_min[1]), float(v_min[2]),
                float(v_max[0]), float(v_max[1]), float(v_max[2]))

    def _make_fcl_obj(i):
        return fcl.CollisionObject(
            bvh_models[i],
            fcl.Transform(rots[i].astype(np.float64),
                          (centers[i] * inv_s).astype(np.float64)),
        )

    fcl_objs = [_make_fcl_obj(i) for i in range(N)]
    aabbs = [_world_aabb_u(i) for i in range(N)]  # 6-tuple per body
    d_hat_u = d_hat * inv_s
    near_thresh_u = d_hat * 1.5 * inv_s
    margin = 2.0 * d_hat_u

    total_rot = 0.0
    for rot_iter in range(max_rot_iters):
        any_updated = False
        for i in range(N):
            ax0, ay0, az0, ax1, ay1, az1 = aabbs[i]
            torque = np.zeros(3)
            n_contacts = 0

            for j in range(N):
                if j == i:
                    continue
                bx0, by0, bz0, bx1, by1, bz1 = aabbs[j]
                # Scalar AABB filter (~10x faster than np.all on 3-element arrays)
                if (ax0 - margin > bx1 or ay0 - margin > by1 or az0 - margin > bz1 or
                    bx0 - margin > ax1 or by0 - margin > ay1 or bz0 - margin > az1):
                    continue

                req = fcl.DistanceRequest(enable_nearest_points=True,
                                          enable_signed_distance=True)
                res = fcl.DistanceResult()
                d_u = fcl.distance(fcl_objs[i], fcl_objs[j], req, res)

                if d_u > near_thresh_u:
                    continue

                if d_u > 0.0:
                    cp_i_u = np.asarray(res.nearest_points[0], dtype=np.float64)
                    cp_j_u = np.asarray(res.nearest_points[1], dtype=np.float64)
                    # Push direction from i's surface toward the closest point on j.
                    n_raw = cp_j_u - cp_i_u
                    cl_world = cp_i_u * scale       # contact on i's surface, world
                    p_other = cp_j_u * scale        # nearest point on j, world
                    d_min_world = d_u * scale
                else:
                    # Overlap: collision query for contact info.
                    creq = fcl.CollisionRequest(num_max_contacts=8,
                                                enable_contact=True)
                    cres = fcl.CollisionResult()
                    fcl.collide(fcl_objs[i], fcl_objs[j], creq, cres)
                    if not cres.is_collision or not cres.contacts:
                        continue
                    c_best = max(cres.contacts,
                                 key=lambda c: c.penetration_depth)
                    depth_u = float(c_best.penetration_depth)
                    n_raw = np.asarray(c_best.normal, dtype=np.float64)
                    if np.dot(n_raw, centers[j] - centers[i]) < 0.0:
                        n_raw = -n_raw
                    pos_u = np.asarray(c_best.pos, dtype=np.float64)
                    cl_world = pos_u * scale
                    p_other = (pos_u + n_raw * depth_u) * scale
                    d_min_world = -depth_u * scale

                nn = np.linalg.norm(n_raw)
                if nn < 1e-12:
                    continue
                n_ij = n_raw / nn

                # Lever arm from center of i to contact point on i's surface
                r_i = cl_world - centers[i]
                # Torque: r × F, where F magnitude grows with penetration depth
                force_mag = max(0.0, d_hat - d_min_world)
                if force_mag <= 0.0:
                    continue
                tau = np.cross(r_i, force_mag * n_ij)
                torque += tau
                n_contacts += 1

            if n_contacts > 0 and np.linalg.norm(torque) > 1e-8:
                omega = 0.5 * torque / n_contacts
                nw = np.linalg.norm(omega)
                max_step = 0.05  # ~3 degrees
                if nw > max_step:
                    omega *= max_step / nw
                    nw = max_step

                rots[i] = RotLib.from_rotvec(omega).as_matrix() @ rots[i]
                total_rot += nw
                any_updated = True

                # Refresh i's FCL Transform + AABB (cheap: just a Transform swap).
                fcl_objs[i] = _make_fcl_obj(i)
                aabbs[i] = _world_aabb_u(i)

        if not any_updated:
            break

    return total_rot


def _optimize_rotations_trimesh(centers, rots, nfs, mverts, mfaces, N,
                                  d_hat, scale, max_rot_iters):
    """Legacy trimesh.proximity.closest_point path (slow but no FCL dep)."""
    import trimesh

    def world_verts_i(i):
        return scale * nfs[i] * (rots[i] @ mverts[i].T).T + centers[i]

    total_rot = 0.0
    meshes = [trimesh.Trimesh(vertices=world_verts_i(i), faces=mfaces[i], process=False)
              for i in range(N)]

    for rot_iter in range(max_rot_iters):
        any_updated = False
        for i in range(N):
            vi = np.asarray(meshes[i].vertices)
            ai0, ai1 = vi.min(0), vi.max(0)
            torque = np.zeros(3)
            n_contacts = 0

            for j in range(N):
                if j == i:
                    continue
                vj = np.asarray(meshes[j].vertices)
                aj0, aj1 = vj.min(0), vj.max(0)
                if not (np.all(ai0 - d_hat * 2 <= aj1) and np.all(aj0 - d_hat * 2 <= ai1)):
                    continue

                cl, dists, fidx = trimesh.proximity.closest_point(meshes[i], vj)
                k = np.argmin(dists)
                d_min = dists[k]

                if d_min < d_hat * 1.5:
                    n_ij = vj[k] - cl[k]
                    nn = np.linalg.norm(n_ij)
                    if nn < 1e-12:
                        continue
                    n_ij = n_ij / nn
                    r_i = cl[k] - centers[i]
                    force_mag = max(0.0, d_hat - d_min)
                    tau = np.cross(r_i, force_mag * n_ij)
                    torque += tau
                    n_contacts += 1

            if n_contacts > 0 and np.linalg.norm(torque) > 1e-8:
                omega = 0.5 * torque / n_contacts
                nw = np.linalg.norm(omega)
                max_step = 0.05
                if nw > max_step:
                    omega *= max_step / nw
                    nw = max_step

                rots[i] = RotLib.from_rotvec(omega).as_matrix() @ rots[i]
                total_rot += nw
                any_updated = True

                meshes[i] = trimesh.Trimesh(
                    vertices=world_verts_i(i), faces=mfaces[i], process=False)

        if not any_updated:
            break

    return total_rot


def solve_dual_qp(A_contact, b_contact, P_diag, n_vars):
    """Solve QP in dual (contact force) space.

    Primal: min (1/2) x^T P x  s.t.  A x >= b
    Dual:   min (1/2) λ^T G λ - b^T λ  s.t.  λ >= 0
    where G = A P^{-1} A^T, and x* = P^{-1} A^T λ*

    Returns (x_primal, lambda_dual, qp_iters) or None if infeasible.
    """
    n_constraints = A_contact.shape[0]
    # P^{-1} is diagonal with entries 1/P_diag
    P_inv_diag = 1.0 / P_diag
    # G = A @ diag(P_inv) @ A^T
    A_scaled = A_contact * P_inv_diag[np.newaxis, :]  # A @ P^{-1}
    G = A_scaled @ A_contact.T  # (n_constraints × n_constraints)

    # Dual QP: min (1/2) λ^T G λ - b^T λ  s.t.  λ >= 0
    G_sparse = sp.csc_matrix(G)
    q_dual = -b_contact

    # Constraints: 0 ≤ λ (box constraint)
    A_dual = sp.eye(n_constraints, format='csc')
    l_dual = np.zeros(n_constraints)
    u_dual = np.full(n_constraints, np.inf)

    solver = osqp.OSQP()
    solver.setup(G_sparse, q_dual, A_dual, l_dual, u_dual,
                 verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                 max_iter=4000, polish=True)
    result = solver.solve()

    if result.info.status not in ('solved', 'solved_inaccurate'):
        return None

    lam = np.maximum(result.x, 0.0)  # ensure non-negative
    # Recover primal: x* = P^{-1} A^T λ*
    x_primal = P_inv_diag * (A_contact.T @ lam)
    return x_primal, lam, result.info.iter


def solve_s4r_qp_step_translation_only(N, active, ds, d_hat, centers,
                                         body_map, active_bodies, contact_sparsity):
    """Fallback: translation-only QP for a single step."""
    n_bodies_qp = len(active_bodies)
    n_vars = 3 * n_bodies_qp
    n_active = len(active)

    P = sp.eye(n_vars, format='csc')
    q = np.zeros(n_vars)
    A = np.zeros((n_active, n_vars))
    l = np.zeros(n_active)

    for ci, (i, j, d_curr, n_ij, ext_i, ext_j, cp_i, cp_j) in enumerate(active):
        b = ds * (ext_i + ext_j) + d_hat - d_curr
        ii = body_map[i]
        jj = body_map[j]
        A[ci, 3 * ii: 3 * ii + 3] = -n_ij
        A[ci, 3 * jj: 3 * jj + 3] = n_ij
        l[ci] = b

    A_sparse = sp.csc_matrix(A)
    u = np.full(n_active, np.inf)

    solver = osqp.OSQP()
    solver.setup(P, q, A_sparse, l, u,
                 verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                 max_iter=4000, polish=True)
    result = solver.solve()

    if result.info.status in ('solved', 'solved_inaccurate'):
        dp_compact = result.x.reshape(n_bodies_qp, 3)
        dp_full = np.zeros((N, 3))
        for idx, b in enumerate(active_bodies):
            dp_full[b] = dp_compact[idx]
        return dp_full
    return None


if __name__ == "__main__":
    import argparse

    # The canonical scene generators live in ../scenes (kubric/hy3d/thingi).
    # Add both this package and ../scenes to the path so this module is
    # runnable directly, e.g.:
    #   python s4r/s4r_qp.py --dataset kubric --n-objects 40 --seed 42
    sys.path.insert(0, _HERE)
    sys.path.insert(0, _os.path.join(_os.path.dirname(_HERE), "scenes"))

    parser = argparse.ArgumentParser(
        description="S4R-QP: Progressive Scaling + QP Solver")
    parser.add_argument('--dataset', choices=['kubric', 'hy3d', 'thingi'],
                        default='kubric',
                        help='Benchmark mesh pool (see ../data and the README)')
    parser.add_argument('--n-objects', '--N', dest='n_objects', type=int,
                        default=40, help='Number of objects')
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seed(s); the paper benchmark seeds are 42, 123, 456')
    parser.add_argument('--d-hat', type=float, default=0.02, help='Safety distance')
    parser.add_argument('--ds-max', type=float, default=0.05, help='Max scale step')
    parser.add_argument('--contact-backend', choices=['trimesh', 'fcl', 'fcl_prebuilt', 'warp'],
                        default='fcl_prebuilt',
                        help='Collision backend (fcl_prebuilt reuses scale-1 BVHs; warp = NVIDIA Warp GPU)')
    parser.add_argument('--M', type=int, default=3,
                        help='Revalidation interval: full collision detection every M steps '
                             '(benchmark default M=3; M=1 detects every step)')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--rotation', action='store_true',
                        help='Enable rotation DOFs (6-DOF QP + SO(3) optimization, slower)')
    parser.add_argument('--dual', action='store_true', help='Use dual QP formulation')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--dump-trajectory', type=str, default=None,
                        help='If set, write a per-scale-step trajectory JSON for external '
                             'viewer. May be a full path or a bare scene name (resolved under '
                             'vis/video/public/trajectories/<name>.json). Only the first seed is dumped.')
    parser.add_argument('--dump-every', type=int, default=1,
                        help='Capture every Nth scale step when --dump-trajectory is set.')
    args = parser.parse_args()

    from make_scene import make_scene

    rot_str = "6DOF" if args.rotation else "3DOF"
    dual_str = "+dual" if args.dual else ""
    print(f"=== S4R-QP {rot_str}{dual_str} M={args.M} | "
          f"{args.dataset}(N={args.n_objects}) ===")
    print(f"{'seed':<7} {'pen':>4} {'RMSD':>8} {'time':>7} {'scale':>6}")
    print("-" * 38)

    for seed in args.seed:
        objects, _sxz, _sy = make_scene(args.dataset, args.n_objects, seed, 1.0)
        r = solve_s4r_qp(
            objects, d_hat=args.d_hat, ds_max=args.ds_max,
            max_steps=args.max_steps, verbose=args.verbose,
            enable_rotation=args.rotation,
            contact_sparsity=True, adaptive_ds=True,
            use_dual=args.dual,
            revalidate_interval=args.M,
            contact_backend=args.contact_backend)

        print(f"{seed:<7} {r['pen']:>4} {r['rmsd']:>8.4f} "
              f"{r['time']:>6.2f}s {r['scale']:>6.4f}")
