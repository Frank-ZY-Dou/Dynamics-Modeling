"""GPU-native S4R-QP driver.

Pipeline (no host roundtrip in the inner loop except the contact-tuple
list returned by the oracle):

  for k = 0, 1, ..., K_max:
     contacts  ← warp oracle (template-shared V3, GPU)        # §4.4.1
     active    ← {pairs needing push at scale + ds}
     A, b      ← per-active-contact arrays
     λ_warm    ← active-set transfer of previous λ*
     Δp        ← dual APGD on GPU (this file's contribution)
     centers  += Δp;  scale += ds

Termination: scale reaches 1.0 or max_steps consumed. Optional
tail-refinement at scale=1 with ds=0 mirrors paper §4.5 (capped iters).

Translation-only QP (paper default per §4.2.2). Rotation extension can
be added later; the warm-start and oracle plumbing already returns the
data needed for the 6-DOF variant.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List

import numpy as np
import warp as wp

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(os.path.dirname(_HERE), "s4r")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dual_apgd import DualAPGD  # noqa: E402
from warp_pair_contact_v3 import S4RWarpContactOracleV3  # noqa: E402
from oracle_v4 import S4RWarpContactOracleV4  # noqa: E402


class _ObjShim:
    """Minimal struct accepted by S4RWarpContactOracleV3."""
    __slots__ = ("normalize_factor", "collision_verts_model", "collision_faces")
    def __init__(self, nf, v, f):
        self.normalize_factor = nf
        self.collision_verts_model = v
        self.collision_faces = f


def _solve_qp_osqp(ci_np, cj_np, cn_np, b_np, n_bodies):
    """Diagnostic: solve the SAME min-norm contact QP the dual APGD solves,
    but exactly on CPU with OSQP. Used only to isolate solver-numerics from
    pipeline effects (NOT a production path).

        min_{Δp ∈ R^{3N}}  ½‖Δp‖²   s.t.  n_k·(Δp_j − Δp_i) ≥ b_k  ∀k

    Returns Δp as an (N, 3) float32 array, matching DualAPGD.solve().
    """
    import numpy as _np
    import scipy.sparse as _sp
    import osqp as _osqp

    K = len(b_np)
    dim = 3 * n_bodies
    if K == 0:
        return _np.zeros((n_bodies, 3), dtype=_np.float32)

    # Build sparse A: row k has -n_k on cols (3i:3i+3), +n_k on (3j:3j+3).
    rows = _np.repeat(_np.arange(K), 6)
    ci = ci_np.astype(_np.int64)
    cj = cj_np.astype(_np.int64)
    cols = _np.empty(6 * K, dtype=_np.int64)
    vals = _np.empty(6 * K, dtype=_np.float64)
    for axis in range(3):
        cols[axis::6] = 3 * ci + axis
        vals[axis::6] = -cn_np[:, axis]
        cols[3 + axis::6] = 3 * cj + axis
        vals[3 + axis::6] = cn_np[:, axis]
    A = _sp.csc_matrix((vals, (rows, cols)), shape=(K, dim))
    P = _sp.identity(dim, format="csc", dtype=_np.float64)
    q = _np.zeros(dim)
    lo = b_np.astype(_np.float64)
    hi = _np.full(K, _np.inf)

    m = _osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=lo, u=hi, eps_abs=1e-7, eps_rel=1e-7,
            max_iter=8000, verbose=False, polish=True)
    res = m.solve()
    dp = res.x
    if dp is None or not _np.all(_np.isfinite(dp)):
        dp = _np.zeros(dim)
    return dp.reshape(n_bodies, 3).astype(_np.float32)


def solve_s4r_gpu(
    objects,
    d_hat: float = 0.02,
    ds_max: float = 0.05,
    max_steps: int = 200,
    s_min: float = 0.01,
    apgd_max_iter: int = 500,
    apgd_tol_primal: float = 1e-6,
    apgd_tol_comp: float = 1e-6,
    apgd_check_every: int = 8,
    apgd_use_graph: bool = False,
    apgd_graph_iters: int = 80,
    oracle_version: str = "v3",   # "v3" or "v4"
    warm_start_lambda: bool = True,
    enable_tail: bool = True,
    tail_max_iters: int = 50,
    enable_cleanup: bool = False,
    cleanup_max_iters: int = 50,
    eval_penetration: bool = True,
    qp_solver: str = "apgd",   # "apgd" (GPU, production) | "osqp" (CPU diagnostic)
    # --- N=30000 RMSD-regression ablation toggles (default = original behavior) ---
    tail_tol_gated: bool = False,   # fix(A): route tail through convergence-checked
                                    #   apgd.solve (max_iter ∝ √K) instead of fixed-iter
                                    #   solve_graph. Removes the large-K tail under-convergence.
    tail_lam_warmstart: bool = True,  # fix(B): when False, cold-start λ=0 each tail iter
                                      #   (mirrors paper's fresh-OSQP-per-iter; removes the
                                      #   α-scaled warm-start over-push bias).
    verbose: bool = True,
):
    """Run GPU-native S4R-QP. Returns a result dict with RMSD, timings,
    penetration stats and a per-step diagnostic list.
    """
    wp.synchronize()
    method_t0 = time.perf_counter()
    N = len(objects)
    centers0 = np.array([o.center for o in objects], dtype=np.float64)
    centers = centers0.copy()
    rots = [np.asarray(o.rotation, dtype=np.float64).copy() for o in objects]
    nfs = [float(o.normalize_factor) for o in objects]
    mverts = [np.asarray(o.collision_verts_model, dtype=np.float64).copy()
              for o in objects]
    mfaces = [np.asarray(o.collision_faces, dtype=np.int32).copy()
              for o in objects]

    # Pre-compute per-body max extent for the s_min sanity check
    # (paper Eq. 10) — bail out with jitter if violated, identical
    # logic to s4r_qp.py.
    max_extents = np.array([nfs[i] * float(np.max(
        np.linalg.norm(mverts[i], axis=1))) for i in range(N)])
    if verbose:
        print(f"  [scene] N={N} bodies, max_extent_global="
              f"{max_extents.max():.4f}", flush=True)

    s_min_safe = float("inf")
    for i in range(N):
        for j in range(i + 1, N):
            d_ij = float(np.linalg.norm(centers[i] - centers[j]))
            ext_sum = max_extents[i] + max_extents[j] + 1e-12
            s_contact = max(0.0, (d_ij - d_hat) / ext_sum)
            s_min_safe = min(s_min_safe, s_contact)
    if verbose:
        print(f"  [smin] safe_bound={s_min_safe:.4f}  using s_min={s_min:.4f} "
              f"{'OK' if s_min < s_min_safe else 'JITTER'}", flush=True)
    if s_min >= s_min_safe:
        # Jitter overlapping centroids apart by d_hat along their axis.
        for i in range(N):
            for j in range(i + 1, N):
                d_ij = float(np.linalg.norm(centers[i] - centers[j]))
                ext_sum = max_extents[i] + max_extents[j] + 1e-12
                if (d_ij - d_hat) / ext_sum < s_min:
                    axis = centers[j] - centers[i]
                    n = float(np.linalg.norm(axis)) + 1e-12
                    axis = axis / n
                    shift = (d_hat + 1e-6) * axis
                    centers[j] += 0.5 * shift
                    centers[i] -= 0.5 * shift

    # ── Warp oracle (template-shared V3, optional GPU broad-phase V4) ──
    shims = [_ObjShim(nfs[i], mverts[i], mfaces[i]) for i in range(N)]
    t_oracle_build = time.time()
    OracleCls = (S4RWarpContactOracleV4 if oracle_version == "v4"
                 else S4RWarpContactOracleV3)
    oracle = OracleCls(shims, d_hat=d_hat)
    oracle_build_time = time.time() - t_oracle_build
    if verbose:
        print(f"  [oracle] {oracle_version} built in {oracle_build_time:.2f}s "
              f"(templates={oracle.n_templates})", flush=True)

    # Pre-grow APGD capacity to an upper bound on K. The graph-fused
    # path captures kernel launches at a fixed cap_K; if K_actual ever
    # exceeds it the graph is invalidated and rebuilt (~hundreds of ms),
    # which silently dominates "QP time" in profiles. Setting cap_K
    # generously up front avoids this entirely.
    K_max_hint = max(8 * N, 4096)
    apgd = DualAPGD(capacity_K=K_max_hint, capacity_N=N)

    # ── Warm-up: build the CUDA graph once on a dummy QP so its
    # capture cost (~tens of ms) doesn't get counted into the first
    # scale step's QP wall.
    if apgd_use_graph:
        dummy_ci = np.array([0], dtype=np.int32)
        dummy_cj = np.array([1 if N > 1 else 0], dtype=np.int32)
        dummy_cn = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        dummy_b = np.array([0.0], dtype=np.float32)
        apgd.solve_graph(dummy_ci, dummy_cj, dummy_cn, dummy_b, N,
                         iters=apgd_graph_iters)

    # ── Progressive scaling loop ──
    scale = s_min
    diagnostics = []
    lam_prev_dict: dict[tuple, float] = {}

    contact_time_total = 0.0
    qp_time_total = 0.0
    tail_time_total = 0.0
    wp.synchronize()
    setup_time = time.perf_counter() - method_t0
    solve_t0 = time.perf_counter()

    for step in range(max_steps):
        if scale >= 1.0 - 1e-6:
            break
        ds = min(ds_max, 1.0 - scale)

        # Contact detection.
        t_c = time.time()
        contacts = oracle.find_contacts(scale, ds, centers, rots)
        contact_time_total += time.time() - t_c

        # Filter to active set (paper §4.2.1).
        active = []
        for (i, j, d_signed, n, ext_i, ext_j, _, _) in contacts:
            need = ds * (ext_i + ext_j) + d_hat - d_signed
            if need > 0:
                active.append((int(i), int(j), float(d_signed),
                               np.asarray(n, dtype=np.float64),
                               float(ext_i), float(ext_j)))
        if not active:
            scale += ds
            diagnostics.append({"step": step, "scale": float(scale),
                                "n_active": 0, "apgd_iters": 0,
                                "primal_viol": 0.0, "qp_ms": 0.0})
            continue

        K = len(active)
        ci_np = np.fromiter((a[0] for a in active), dtype=np.int32, count=K)
        cj_np = np.fromiter((a[1] for a in active), dtype=np.int32, count=K)
        cn_np = np.stack([a[3] for a in active]).astype(np.float32)
        b_np = np.fromiter(
            (ds * (a[4] + a[5]) + d_hat - a[2] for a in active),
            dtype=np.float32, count=K,
        )

        # Active-set transfer for warm-start.
        lam_init = None
        if warm_start_lambda and lam_prev_dict:
            lam_init = np.zeros(K, dtype=np.float32)
            for k in range(K):
                key = (int(ci_np[k]), int(cj_np[k]))
                if key in lam_prev_dict:
                    lam_init[k] = lam_prev_dict[key]
                else:
                    krev = (int(cj_np[k]), int(ci_np[k]))
                    if krev in lam_prev_dict:
                        lam_init[k] = lam_prev_dict[krev]

        # Inner QP solve. APGD (GPU) is the production path; the OSQP
        # branch is a diagnostic to isolate solver vs pipeline effects —
        # it solves the IDENTICAL dual QP (same A, b, active set) with an
        # exact CPU solver so any RMSD gap that survives points at the
        # pipeline, not the APGD numerics.
        t_qp = time.time()
        if qp_solver == "osqp":
            dp = _solve_qp_osqp(ci_np, cj_np, cn_np, b_np, N)
            # info shape compatible with the APGD path (warm-start cache +
            # diagnostics downstream). λ is not exposed by the OSQP path, so
            # disable warm-start carry-over (zeros) — exact solve doesn't need it.
            info = {"solver": "osqp", "lam": np.zeros(K, dtype=np.float32),
                    "iters": 0, "primal_viol": 0.0, "comp_viol": 0.0, "L": 0.0}
        elif apgd_use_graph:
            dp, info = apgd.solve_graph(
                ci_np, cj_np, cn_np, b_np, N,
                lam_warm=lam_init,
                iters=apgd_graph_iters,
            )
        else:
            dp, info = apgd.solve(
                ci_np, cj_np, cn_np, b_np, N,
                lam_warm=lam_init,
                max_iter=apgd_max_iter,
                tol_primal=apgd_tol_primal,
                tol_comp=apgd_tol_comp,
                check_every=apgd_check_every,
            )
        qp_time_total += time.time() - t_qp

        # Apply displacement and advance scale.
        centers += dp.astype(np.float64)
        scale += ds

        # Cache λ keyed by ordered (i, j) for next-step warm-start.
        lam_now = info["lam"]
        lam_prev_dict = {(int(ci_np[k]), int(cj_np[k])): float(lam_now[k])
                         for k in range(K)}

        diagnostics.append({
            "step": step, "scale": float(scale), "n_active": K,
            "apgd_iters": int(info["iters"]),
            "primal_viol": float(info["primal_viol"]),
            "comp_viol": float(info["comp_viol"]),
            "L": float(info["L"]),
            "max_disp": float(np.max(np.linalg.norm(dp, axis=1))),
            "qp_ms": (time.time() - t_qp) * 1000.0,
        })

        if verbose and (step + 1) % 5 == 0:
            rmsd_now = float(np.sqrt(np.mean(
                np.sum((centers - centers0) ** 2, axis=1))))
            print(
                f"  step={step+1:3d} scale={scale:.3f} K={K:4d} "
                f"apgd_iter={info['iters']:3d} "
                f"pv={info['primal_viol']:.1e} "
                f"max_disp={diagnostics[-1]['max_disp']:.4f} "
                f"RMSD={rmsd_now:.4f}", flush=True)

    main_time = time.perf_counter() - solve_t0  # MERGE-FIX: old t0 renamed to solve_t0 in the v1 timing refactor
    if verbose:
        print(f"  [main] {len(diagnostics)} steps, scale={scale:.4f}, "
              f"main_loop={main_time:.2f}s", flush=True)

    # ── Tail refinement (ds=0 at scale=1) ──
    tail_log = []
    tail_lam_prev_dict: dict[tuple, float] = {}
    if enable_tail and scale >= 1.0 - 1e-6:
        for tail_it in range(tail_max_iters):
            t_t = time.time()
            contacts_t = oracle.find_contacts(scale, 0.0, centers, rots)
            pen = [c for c in contacts_t if c[2] < 0.0]
            if not pen:
                if verbose:
                    print(f"  [tail] iter={tail_it} pen=0 — converged",
                          flush=True)
                break
            active_t = []
            for (i, j, d_signed, n, ext_i, ext_j, _, _) in contacts_t:
                # ds=0 means b = d_hat - d_signed; keep ALL near-contact
                # pairs (paper §4.5.1 full-active-set correction).
                if d_signed < d_hat:
                    active_t.append((int(i), int(j), float(d_signed),
                                     np.asarray(n, dtype=np.float64),
                                     float(ext_i), float(ext_j)))
            if not active_t:
                break
            K = len(active_t)
            ci_t = np.fromiter((a[0] for a in active_t), dtype=np.int32, count=K)
            cj_t = np.fromiter((a[1] for a in active_t), dtype=np.int32, count=K)
            cn_t = np.stack([a[3] for a in active_t]).astype(np.float32)
            b_t = np.fromiter(
                (d_hat - a[2] for a in active_t),
                dtype=np.float32, count=K,
            )

            # Warm-start λ from previous tail iter — same scale, near-
            # identical active set (the residual pen pairs barely change
            # between consecutive tail iters), so λ_prev is an excellent
            # initial guess.
            # fix(B): optionally cold-start λ each tail iter (paper parity).
            lam_init_t = None
            if tail_lam_warmstart and tail_lam_prev_dict:
                lam_init_t = np.zeros(K, dtype=np.float32)
                for k in range(K):
                    key = (int(ci_t[k]), int(cj_t[k]))
                    if key in tail_lam_prev_dict:
                        lam_init_t[k] = tail_lam_prev_dict[key]
                    else:
                        krev = (int(cj_t[k]), int(ci_t[k]))
                        if krev in tail_lam_prev_dict:
                            lam_init_t[k] = tail_lam_prev_dict[krev]

            if qp_solver == "osqp":
                dp = _solve_qp_osqp(ci_t, cj_t, cn_t, b_t, N)
                info = {"solver": "osqp",
                        "lam": np.zeros(K, dtype=np.float32),
                        "iters": 0, "primal_viol": 0.0, "comp_viol": 0.0,
                        "L": 0.0}
            elif tail_tol_gated:
                # fix(A): convergence-checked APGD with K-scaled iteration
                # budget — removes the fixed-60-iter under-convergence that
                # leaves non-min-norm (over-pushed) tail steps at large K.
                tail_max_it = max(2000, 20 * int(np.sqrt(max(K, 1))))
                dp, info = apgd.solve(
                    ci_t, cj_t, cn_t, b_t, N, lam_warm=lam_init_t,
                    max_iter=tail_max_it,
                    tol_primal=apgd_tol_primal * 0.1,
                    tol_comp=apgd_tol_comp * 0.1,
                    check_every=apgd_check_every,
                )
            elif apgd_use_graph:
                # ORIGINAL: fixed-iter fused graph (production default).
                dp, info = apgd.solve_graph(
                    ci_t, cj_t, cn_t, b_t, N, lam_warm=lam_init_t,
                    iters=apgd_graph_iters,
                )
            else:
                dp, info = apgd.solve(
                    ci_t, cj_t, cn_t, b_t, N, lam_warm=lam_init_t,
                    max_iter=apgd_max_iter * 2,
                    tol_primal=apgd_tol_primal * 0.1,
                    tol_comp=apgd_tol_comp * 0.1,
                    check_every=apgd_check_every,
                )

            # Step damping (paper Eq. 21).
            max_step = float(np.max(np.linalg.norm(dp, axis=1)))
            alpha = min(1.0, d_hat / max(max_step, 1e-12))
            centers += alpha * dp.astype(np.float64)

            # Cache λ keyed by ordered (i, j) for next tail iter's warm-
            # start. Scale by α since we damped the step.
            lam_now = info["lam"]
            tail_lam_prev_dict = {
                (int(ci_t[k]), int(cj_t[k])): float(lam_now[k] * alpha)
                for k in range(K)
            }

            tail_log.append({
                "iter": tail_it, "K": K,
                "n_pen": len(pen),
                "max_pen": float(max(-c[2] for c in pen)),
                "apgd_iters": int(info["iters"]),
                "primal_viol": float(info["primal_viol"]),
                "alpha": float(alpha),
            })
            tail_time_total += time.time() - t_t
            if verbose:
                print(f"  [tail] iter={tail_it} K={K} pen={len(pen)} "
                      f"maxpen={tail_log[-1]['max_pen']:.4f} "
                      f"alpha={alpha:.2f} apgd={info['iters']}",
                      flush=True)

    # ── Penalty cleanup (Jacobi/GS) for any residual penetration ──
    # The QP tail can stagnate when pushing one pair micro-collides a
    # neighbour. For each remaining penetrating pair we apply a direct
    # mass-balanced push of size (|depth| + ε) along the contact normal.
    # Adaptive ε; switch Jacobi→Gauss-Seidel after stagnation.
    cleanup_log = []
    cleanup_time_total = 0.0
    if enable_cleanup:
        cleanup_eps = d_hat * 0.1
        prev_pen_n = None
        stagnant = 0
        gauss_seidel = False
        for ci in range(cleanup_max_iters):
            t_cu = time.time()
            cu_contacts = oracle.find_contacts(scale, 0.0, centers, rots)
            cu_pen = [c for c in cu_contacts if c[2] < 0.0]
            if not cu_pen:
                if verbose:
                    print(f"  [cleanup] iter={ci} pen=0 — converged",
                          flush=True)
                cleanup_time_total += time.time() - t_cu
                break
            n_pen_now = len(cu_pen)
            if prev_pen_n is not None:
                if n_pen_now > prev_pen_n:
                    cleanup_eps *= 0.5  # too aggressive — pull back
                    stagnant = 0
                elif n_pen_now == prev_pen_n:
                    stagnant += 1
                    if stagnant >= 2:
                        cleanup_eps = min(cleanup_eps * 1.5, d_hat * 2.0)
                        gauss_seidel = True
                else:
                    stagnant = 0
            prev_pen_n = n_pen_now
            if gauss_seidel:
                cu_pen.sort(key=lambda c: c[2])  # most-negative first
                for (i, j, d_signed, n_ij, _, _, _, _) in cu_pen:
                    mag = (-d_signed) + cleanup_eps
                    n_arr = np.asarray(n_ij, dtype=np.float64)
                    centers[i] -= 0.5 * mag * n_arr
                    centers[j] += 0.5 * mag * n_arr
            else:
                push = np.zeros_like(centers)
                for (i, j, d_signed, n_ij, _, _, _, _) in cu_pen:
                    mag = (-d_signed) + cleanup_eps
                    n_arr = np.asarray(n_ij, dtype=np.float64)
                    push[i] -= 0.5 * mag * n_arr
                    push[j] += 0.5 * mag * n_arr
                centers += push
            cleanup_log.append({
                "iter": ci, "n_pen": n_pen_now,
                "eps": cleanup_eps,
                "mode": "GS" if gauss_seidel else "J",
            })
            cleanup_time_total += time.time() - t_cu
            if verbose and (ci + 1) % 10 == 0:
                print(f"  [cleanup] iter={ci+1} pen={n_pen_now} "
                      f"eps={cleanup_eps:.5f} "
                      f"mode={'GS' if gauss_seidel else 'J'}", flush=True)

    wp.synchronize()
    solve_time = time.perf_counter() - solve_t0
    method_total_time = setup_time + solve_time
    eval_t0 = time.perf_counter()
    rmsd = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1))))

    # ── Final penetration evaluation (shared mesh evaluator) ──
    pen_pairs = -1
    max_pen = -1.0
    if eval_penetration:
        try:
            import trimesh
            from mesh_collision import evaluate_world_collision_meshes
            final_meshes = []
            for i in range(N):
                v_world = nfs[i] * (rots[i] @ mverts[i].T).T + centers[i]
                final_meshes.append(trimesh.Trimesh(
                    vertices=v_world, faces=mfaces[i], process=False))
            stats = evaluate_world_collision_meshes(final_meshes)
            pen_pairs = int(stats.pen_pairs)
            max_pen = float(stats.max_penetration)
        except Exception as e:
            if verbose:
                print(f"  [warn] penetration eval failed: {e}", flush=True)
    evaluation_time = time.perf_counter() - eval_t0

    return {
        "centers": centers,
        "centers0": centers0,
        "rmsd": rmsd,
        "timing_policy": "per_scene_setup_plus_solve_v1",
        "setup_time": float(setup_time),
        "solve_time": float(solve_time),
        "method_total_time": float(method_total_time),
        "evaluation_time": float(evaluation_time),
        "solver_internal_time": float(solve_time),
        "wall_time": float(method_total_time),
        "contact_time": float(contact_time_total),
        "qp_time": float(qp_time_total),
        "tail_time": float(tail_time_total),
        "oracle_build_time": float(oracle_build_time),
        "n_steps": len(diagnostics),
        "scale_final": float(scale),
        "pen_pairs": pen_pairs,
        "max_penetration": max_pen,
        "diagnostics": diagnostics,
        "tail_log": tail_log,
        "cleanup_log": cleanup_log,
        "cleanup_time": float(cleanup_time_total),
    }
