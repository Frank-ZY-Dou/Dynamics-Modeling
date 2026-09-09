"""GPU-native S4R-QP driver.

Pipeline (no host round trip in the inner loop except the contact-tuple
list returned by the oracle):

  for k = 0, 1, ..., K_max:
     contacts  ← Warp oracle (template-shared V3 or V4, GPU)
     active    ← {pairs needing a push before scale + ds}
     A, b      ← per-active-contact arrays
     λ_warm    ← active-set transfer of the previous λ*
     Δp        ← dual APGD on the GPU
     centers  += Δp;  scale += ds          (only for a finite, checked step)

Termination: scale reaches 1.0, the step budget is spent, or a step fails
numerically after repeated halving of ds. At full scale a tail refinement
(ds = 0) runs on the GPU oracle, and then an exact FCL pass (triangle
crossings, containment) re-checks the scene and, if it still finds a
penetrating pair, runs correction iterations with the exact contacts. The
sampled GPU oracle is a contact generator; the FCL pass and the shared
mesh evaluator are the certificate.

Translation-only QP. The result dict uses the CPU solver's outcome rule:
``status`` is ``'converged'`` only when the continuation reached full
scale, the poses are finite and the shared mesh evaluator finds no
penetrating pair on the full-size bodies; ``native_converged`` reports
the GPU oracle's own tail test, ``fcl_verified`` the exact FCL re-check,
``unconverged_steps`` how many applied steps stayed above the primal
tolerance after the extra rounds (see the return statement).
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
from s4r_qp import separate_coincident_centroids, PrebuiltFCLOracle  # noqa: E402


class _ObjShim:
    """Minimal struct accepted by the Warp oracles."""
    __slots__ = ("normalize_factor", "collision_verts_model", "collision_faces")

    def __init__(self, nf, v, f):
        self.normalize_factor = nf
        self.collision_verts_model = v
        self.collision_faces = f


def _solve_qp_osqp(ci_np, cj_np, cn_np, b_np, n_bodies):
    """Diagnostic: solve the SAME min-norm contact QP the dual APGD solves,
    exactly on the CPU with OSQP, to separate solver numerics from pipeline
    effects (not a production path).

        min_{Δp ∈ R^{3N}}  ½‖Δp‖²   s.t.  n_k·(Δp_j − Δp_i) ≥ b_k  ∀k
    """
    import numpy as _np
    import scipy.sparse as _sp
    from s4r_qp import osqp_solve, qp_status_ok

    K = len(b_np)
    dim = 3 * n_bodies
    if K == 0:
        return _np.zeros((n_bodies, 3), dtype=_np.float32)
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
    res = osqp_solve(P, _np.zeros(dim), A, b_np.astype(_np.float64), _np.full(K, _np.inf),
                     eps_abs=1e-7, eps_rel=1e-7, max_iter=8000, verbose=False, polishing=True)
    dp = res.x if qp_status_ok(res) else None
    if dp is None or not _np.all(_np.isfinite(dp)):
        dp = _np.zeros(dim)
    return dp.reshape(n_bodies, 3).astype(_np.float32)


def _contact_arrays(active):
    K = len(active)
    ci = np.fromiter((a[0] for a in active), dtype=np.int32, count=K)
    cj = np.fromiter((a[1] for a in active), dtype=np.int32, count=K)
    cn = np.stack([a[3] for a in active]).astype(np.float32)
    return ci, cj, cn


def _warm_from(lam_dict, ci, cj):
    if not lam_dict:
        return None
    K = len(ci)
    lam = np.zeros(K, dtype=np.float32)
    for k in range(K):
        key = (int(ci[k]), int(cj[k]))
        if key in lam_dict:
            lam[k] = lam_dict[key]
        else:
            krev = (int(cj[k]), int(ci[k]))
            if krev in lam_dict:
                lam[k] = lam_dict[krev]
    return lam


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
    tail_tol_gated: bool = False,     # convergence-checked tail with a K-scaled budget
    tail_lam_warmstart: bool = True,  # False: cold-start λ each tail iteration
    verify_with_fcl: bool = True,     # exact FCL re-check and correction at full scale
    verify_tail_iters: int = 20,
    accept_primal_tol: float | None = None,  # step acceptance threshold on the primal residual
    apgd_extra_rounds: int = 2,       # extra warm-started solves before accepting an unconverged step
    max_step_failures: int = 4,
    verbose: bool = True,
):
    """Run GPU-native S4R-QP. Returns a result dict (keys documented at the
    return statement): poses, RMSD, timings, penetration statistics from the
    shared mesh evaluator at full scale, per-step diagnostics and the
    outcome contract.
    """
    if not (np.isfinite(ds_max) and ds_max > 0.0):
        raise ValueError(f"ds_max must be a positive finite scale step, got {ds_max}")
    if not (np.isfinite(d_hat) and d_hat >= 0.0):
        raise ValueError(f"d_hat must be a non-negative finite distance, got {d_hat}")
    if not (0.0 < s_min < 1.0):
        raise ValueError(f"s_min must lie in (0, 1), got {s_min}")
    if accept_primal_tol is None:
        accept_primal_tol = 10.0 * apgd_tol_primal

    wp.synchronize()
    method_t0 = time.perf_counter()
    N = len(objects)
    centers0 = np.array([o.center for o in objects], dtype=np.float64).reshape(N, 3)
    centers = centers0.copy()
    rots = [np.asarray(o.rotation, dtype=np.float64).copy() for o in objects]
    nfs = [float(o.normalize_factor) for o in objects]
    mverts = [np.asarray(o.collision_verts_model, dtype=np.float64).copy() for o in objects]
    mfaces = [np.asarray(o.collision_faces, dtype=np.int32).copy() for o in objects]
    if N and not np.all(np.isfinite(centers)):
        raise ValueError("object centers contain NaN or Inf")
    notes = []

    # Bounding-sphere radius about the scaling centre (scale 1).
    max_extents = np.array([nfs[i] * float(np.max(np.linalg.norm(mverts[i], axis=1)))
                            for i in range(N)])
    if verbose:
        print(f"  [scene] N={N} bodies, max_extent_global="
              f"{max_extents.max() if N else 0.0:.4f}", flush=True)

    # Starting-scale admissibility: no pair may already be within d_hat at
    # s_min. Shared with the CPU solver (Gauss-Seidel passes, deterministic
    # split directions for coincident centroids, re-checked until clean).
    passes, remaining = separate_coincident_centroids(centers, max_extents, d_hat, s_min)
    if verbose:
        print(f"  [smin] s_min={s_min:.4f}: {'admissible' if passes == 0 else f'centroids separated in {passes} passes'}",
              flush=True)
    if passes:
        notes.append(f"starting-scale admissibility restored in {passes} passes")
    if remaining:
        raise RuntimeError("could not separate coincident centroids to an admissible starting scale")

    # ── Warp oracle ──
    shims = [_ObjShim(nfs[i], mverts[i], mfaces[i]) for i in range(N)]
    t_oracle_build = time.time()
    OracleCls = S4RWarpContactOracleV4 if oracle_version == "v4" else S4RWarpContactOracleV3
    oracle = OracleCls(shims, d_hat=d_hat)
    oracle_build_time = time.time() - t_oracle_build
    if verbose:
        print(f"  [oracle] {oracle_version} built in {oracle_build_time:.2f}s "
              f"(templates={oracle.n_templates})", flush=True)

    # APGD capacity: an upper bound on K keeps the captured graph valid.
    K_max_hint = max(8 * N, 4096)
    apgd = DualAPGD(capacity_K=K_max_hint, capacity_N=N)
    if apgd_use_graph:
        dummy_ci = np.array([0], dtype=np.int32)
        dummy_cj = np.array([1 if N > 1 else 0], dtype=np.int32)
        dummy_cn = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        dummy_b = np.array([0.0], dtype=np.float32)
        apgd.solve_graph(dummy_ci, dummy_cj, dummy_cn, dummy_b, N, iters=apgd_graph_iters)

    def _solve_contacts(ci_np, cj_np, cn_np, b_np, lam_init, tail=False):
        """One dual solve; returns (dp float32 (N,3), info)."""
        K = len(b_np)
        if qp_solver == "osqp":
            dp = _solve_qp_osqp(ci_np, cj_np, cn_np, b_np, N)
            return dp, {"solver": "osqp", "lam": np.zeros(K, dtype=np.float32),
                        "iters": 0, "primal_viol": 0.0, "comp_viol": 0.0, "L": 0.0,
                        "finite": bool(np.all(np.isfinite(dp)))}
        if tail and tail_tol_gated:
            return apgd.solve(ci_np, cj_np, cn_np, b_np, N, lam_warm=lam_init,
                              max_iter=max(2000, 20 * int(np.sqrt(max(K, 1)))),
                              tol_primal=apgd_tol_primal * 0.1, tol_comp=apgd_tol_comp * 0.1,
                              check_every=apgd_check_every)
        if apgd_use_graph:
            return apgd.solve_graph(ci_np, cj_np, cn_np, b_np, N, lam_warm=lam_init,
                                    iters=apgd_graph_iters)
        return apgd.solve(ci_np, cj_np, cn_np, b_np, N, lam_warm=lam_init,
                          max_iter=apgd_max_iter * (2 if tail else 1),
                          tol_primal=apgd_tol_primal * (0.1 if tail else 1.0),
                          tol_comp=apgd_tol_comp * (0.1 if tail else 1.0),
                          check_every=apgd_check_every)

    def _solve_checked(ci_np, cj_np, cn_np, b_np, lam_init, tail=False):
        """Solve, then re-solve warm-started while the primal residual is
        above the acceptance threshold (up to apgd_extra_rounds)."""
        dp, info = _solve_contacts(ci_np, cj_np, cn_np, b_np, lam_init, tail)
        rounds = 0
        while (info.get("finite", True) and info["primal_viol"] > accept_primal_tol
               and rounds < apgd_extra_rounds and qp_solver != "osqp"):
            dp, info = _solve_contacts(ci_np, cj_np, cn_np, b_np, info["lam"], tail)
            rounds += 1
        info["extra_rounds"] = rounds
        info["converged"] = bool(info.get("finite", True) and info["primal_viol"] <= accept_primal_tol)
        return dp, info

    # ── Progressive scaling loop ──
    scale = s_min
    diagnostics = []
    lam_prev_dict: dict = {}
    contact_time_total = 0.0
    qp_time_total = 0.0
    tail_time_total = 0.0
    unconverged_steps = 0
    step_failures = 0
    ds_retry_cap = float("inf")
    stop_reason = None
    steps_used = 0
    wp.synchronize()
    setup_time = time.perf_counter() - method_t0
    solve_t0 = time.perf_counter()

    for step in range(max_steps):
        if scale >= 1.0 - 1e-6:
            break
        steps_used = step + 1
        ds = min(ds_max, 1.0 - scale, ds_retry_cap)

        t_c = time.time()
        contacts = oracle.find_contacts(scale, ds, centers, rots)
        contact_time_total += time.time() - t_c

        active = []
        for (i, j, d_signed, n, ext_i, ext_j, _, _) in contacts:
            if ds * (ext_i + ext_j) + d_hat - d_signed > 0:
                active.append((int(i), int(j), float(d_signed),
                               np.asarray(n, dtype=np.float64), float(ext_i), float(ext_j)))
        if not active:
            scale += ds
            ds_retry_cap = float("inf")
            diagnostics.append({"step": step, "scale": float(scale), "ds": float(ds),
                                "n_active": 0, "apgd_iters": 0,
                                "primal_viol": 0.0, "qp_ms": 0.0})
            continue

        K = len(active)
        ci_np, cj_np, cn_np = _contact_arrays(active)
        b_np = np.fromiter((ds * (a[4] + a[5]) + d_hat - a[2] for a in active),
                           dtype=np.float32, count=K)
        lam_init = _warm_from(lam_prev_dict, ci_np, cj_np) if warm_start_lambda else None

        t_qp = time.time()
        dp, info = _solve_checked(ci_np, cj_np, cn_np, b_np, lam_init)
        qp_time_total += time.time() - t_qp

        # Acceptance gate: a non-finite step is never applied; the scale
        # step is halved and the step retried, then the run stops with an
        # explicit failure.
        if not info.get("finite", True):
            step_failures += 1
            ds_retry_cap = 0.5 * ds
            if verbose:
                print(f"  step={step+1}: non-finite APGD step, retrying with ds={ds_retry_cap:.4f}", flush=True)
            if step_failures > max_step_failures or ds_retry_cap < 1e-6:
                stop_reason = "numerical_failure"
                break
            continue
        if not info["converged"]:
            unconverged_steps += 1

        centers += dp.astype(np.float64)
        scale += ds
        ds_retry_cap = float("inf")
        step_failures = 0

        lam_now = info["lam"]
        lam_prev_dict = {(int(ci_np[k]), int(cj_np[k])): float(lam_now[k]) for k in range(K)}

        diagnostics.append({
            "step": step, "scale": float(scale), "ds": float(ds), "n_active": K,
            "apgd_iters": int(info["iters"]),
            "primal_viol": float(info["primal_viol"]),
            "comp_viol": float(info["comp_viol"]),
            "L": float(info["L"]),
            "extra_rounds": int(info.get("extra_rounds", 0)),
            "converged": bool(info["converged"]),
            "max_disp": float(np.max(np.linalg.norm(dp, axis=1))),
            "qp_ms": (time.time() - t_qp) * 1000.0,
        })

        if verbose and (step + 1) % 5 == 0:
            rmsd_now = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1))))
            print(f"  step={step+1:3d} scale={scale:.3f} K={K:4d} "
                  f"apgd_iter={info['iters']:3d} pv={info['primal_viol']:.1e} "
                  f"max_disp={diagnostics[-1]['max_disp']:.4f} RMSD={rmsd_now:.4f}", flush=True)

    continuation_complete = bool(scale >= 1.0 - 1e-6)
    if continuation_complete:
        scale = 1.0
    elif stop_reason is None:
        stop_reason = "max_steps"
    main_time = time.perf_counter() - solve_t0
    if verbose:
        print(f"  [main] {len(diagnostics)} steps, scale={scale:.4f}, "
              f"main_loop={main_time:.2f}s", flush=True)

    # ── Tail refinement (ds = 0 at scale 1) on the GPU oracle ──
    tail_log = []
    tail_stop_reason = "not_run"
    tail_lam_prev_dict: dict = {}
    if enable_tail and continuation_complete:
        tail_stop_reason = "tail_iter_cap" if tail_max_iters > 0 else "tail_disabled"
        for tail_it in range(tail_max_iters):
            t_t = time.time()
            contacts_t = oracle.find_contacts(scale, 0.0, centers, rots)
            pen = [c for c in contacts_t if c[2] < 0.0]
            if not pen:
                tail_stop_reason = "feasible"
                if verbose:
                    print(f"  [tail] iter={tail_it} pen=0", flush=True)
                break
            # ds = 0: b = d_hat − d; keep every near-contact pair so that
            # pushing one pair does not flip a neighbour.
            active_t = [(int(i), int(j), float(d), np.asarray(n, dtype=np.float64), float(ei), float(ej))
                        for (i, j, d, n, ei, ej, _, _) in contacts_t if d < d_hat]
            if not active_t:
                break
            K = len(active_t)
            ci_t, cj_t, cn_t = _contact_arrays(active_t)
            b_t = np.fromiter((d_hat - a[2] for a in active_t), dtype=np.float32, count=K)
            lam_init_t = _warm_from(tail_lam_prev_dict, ci_t, cj_t) if tail_lam_warmstart else None
            dp, info = _solve_checked(ci_t, cj_t, cn_t, b_t, lam_init_t, tail=True)
            if not info.get("finite", True):
                tail_stop_reason = "tail_numerical_failure"
                break
            # Step damping: no body moves more than d_hat per iteration.
            max_step = float(np.max(np.linalg.norm(dp, axis=1)))
            alpha = min(1.0, d_hat / max(max_step, 1e-12))
            centers += alpha * dp.astype(np.float64)
            lam_now = info["lam"]
            tail_lam_prev_dict = {(int(ci_t[k]), int(cj_t[k])): float(lam_now[k] * alpha) for k in range(K)}
            tail_log.append({
                "iter": tail_it, "K": K, "n_pen": len(pen),
                "max_pen": float(max(-c[2] for c in pen)),
                "apgd_iters": int(info["iters"]),
                "primal_viol": float(info["primal_viol"]),
                "alpha": float(alpha),
            })
            tail_time_total += time.time() - t_t
            if verbose:
                print(f"  [tail] iter={tail_it} K={K} pen={len(pen)} "
                      f"maxpen={tail_log[-1]['max_pen']:.4f} alpha={alpha:.2f} apgd={info['iters']}", flush=True)

    # ── Penalty cleanup (Jacobi / Gauss-Seidel) for residual penetration ──
    cleanup_log = []
    cleanup_time_total = 0.0
    if enable_cleanup and continuation_complete:
        cleanup_eps = d_hat * 0.1
        prev_pen_n = None
        stagnant = 0
        gauss_seidel = False
        for ci in range(cleanup_max_iters):
            t_cu = time.time()
            cu_contacts = oracle.find_contacts(scale, 0.0, centers, rots)
            cu_pen = [c for c in cu_contacts if c[2] < 0.0]
            if not cu_pen:
                cleanup_time_total += time.time() - t_cu
                break
            n_pen_now = len(cu_pen)
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
            if gauss_seidel:
                cu_pen.sort(key=lambda c: c[2])
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
            cleanup_log.append({"iter": ci, "n_pen": n_pen_now, "eps": cleanup_eps,
                                "mode": "GS" if gauss_seidel else "J"})
            cleanup_time_total += time.time() - t_cu

    # ── Exact re-check and correction with the FCL oracle ──
    # The sampled oracle can miss a crossing between its samples. At full
    # scale the exact FCL oracle (triangle crossings, containment) re-checks
    # every pair and, when it still finds penetration, runs correction
    # iterations with its own contacts.
    verify_log = []
    fcl_verified = None
    verify_time = 0.0
    if verify_with_fcl and continuation_complete:
        t_v = time.time()
        fcl_oracle = PrebuiltFCLOracle(nfs, mverts, mfaces, d_hat)
        fcl_verified = False
        v_lam: dict = {}
        for v_it in range(verify_tail_iters + 1):
            contacts_f = fcl_oracle.find_contacts(1.0, 0.0, centers, rots, incremental=True)
            pen_f = [c for c in contacts_f if c[2] < 0.0]
            if not pen_f:
                fcl_verified = True
                break
            if v_it == verify_tail_iters:
                break
            active_f = [(int(i), int(j), float(d), np.asarray(n, dtype=np.float64), float(ei), float(ej))
                        for (i, j, d, n, ei, ej, _, _) in contacts_f if d < d_hat]
            K = len(active_f)
            ci_f, cj_f, cn_f = _contact_arrays(active_f)
            b_f = np.fromiter((d_hat - a[2] for a in active_f), dtype=np.float32, count=K)
            dp, info = _solve_checked(ci_f, cj_f, cn_f, b_f, _warm_from(v_lam, ci_f, cj_f), tail=True)
            if not info.get("finite", True):
                break
            max_step = float(np.max(np.linalg.norm(dp, axis=1)))
            alpha = min(1.0, d_hat / max(max_step, 1e-12))
            step_f = alpha * dp.astype(np.float64)
            step_f[np.linalg.norm(step_f, axis=1) <= 1e-10] = 0.0
            centers += step_f
            v_lam = {(int(ci_f[k]), int(cj_f[k])): float(info["lam"][k] * alpha) for k in range(K)}
            verify_log.append({"iter": v_it, "K": K, "n_pen": len(pen_f),
                               "max_pen": float(max(-c[2] for c in pen_f)),
                               "alpha": float(alpha), "apgd_iters": int(info["iters"])})
            if verbose:
                print(f"  [verify] iter={v_it} exact pen={len(pen_f)} "
                      f"maxpen={verify_log[-1]['max_pen']:.4f} alpha={alpha:.2f}", flush=True)
        verify_time = time.time() - t_v
        if verbose:
            print(f"  [verify] exact FCL check: {'penetration-free' if fcl_verified else 'residual penetration'} "
                  f"after {len(verify_log)} correction iterations ({verify_time:.2f}s)", flush=True)

    wp.synchronize()
    solve_time = time.perf_counter() - solve_t0
    method_total_time = setup_time + solve_time
    eval_t0 = time.perf_counter()
    rmsd = float(np.sqrt(np.mean(np.sum((centers - centers0) ** 2, axis=1)))) if N else 0.0
    poses_finite = bool(np.all(np.isfinite(centers)))

    # ── Final penetration evaluation (shared mesh evaluator, full scale) ──
    pen_pairs = -1
    max_pen = -1.0
    if eval_penetration:
        try:
            import trimesh
            from mesh_collision import evaluate_world_collision_meshes
            final_meshes = []
            for i in range(N):
                v_world = nfs[i] * (rots[i] @ mverts[i].T).T + centers[i]
                final_meshes.append(trimesh.Trimesh(vertices=v_world, faces=mfaces[i], process=False))
            stats = evaluate_world_collision_meshes(final_meshes)
            pen_pairs = int(stats.pen_pairs)
            max_pen = float(stats.max_penetration)
        except Exception as e:
            if verbose:
                print(f"  [warn] penetration eval failed: {e}", flush=True)
    evaluation_time = time.perf_counter() - eval_t0

    native_converged = bool(continuation_complete and tail_stop_reason == "feasible")
    if not continuation_complete:
        status = stop_reason
    elif not poses_finite:
        status = "numerical_failure"
    elif pen_pairs > 0:
        status = "residual_penetration"
    elif pen_pairs == 0:
        status = "converged"
    else:
        status = "unevaluated"

    return {
        "centers": centers,
        "final_centers": centers.copy(),
        "final_rotations": np.stack(rots, axis=0) if N else np.zeros((0, 3, 3)),
        "centers0": centers0,
        "rmsd": rmsd,
        "timing_policy": "per_scene_setup_plus_solve_v1",
        "setup_time": float(setup_time),
        "solve_time": float(solve_time),
        "method_total_time": float(method_total_time),
        "evaluation_time": float(evaluation_time),
        "solver_internal_time": float(solve_time),
        "wall_time": float(method_total_time),
        "time": float(method_total_time),
        "contact_time": float(contact_time_total),
        "qp_time": float(qp_time_total),
        "tail_time": float(tail_time_total),
        "verify_time": float(verify_time),
        "oracle_build_time": float(oracle_build_time),
        "n_steps": len(diagnostics),
        "steps": steps_used,
        "scale_final": float(scale),
        "scale": float(scale),
        # Scored at full scale by the shared mesh evaluator (-1 = not run).
        "pen_pairs": pen_pairs,
        "pen": pen_pairs,
        "max_penetration": max_pen,
        "max_pen": max_pen,
        "evaluated_at_scale": 1.0,
        # Outcome contract.
        "status": status,
        "continuation_complete": continuation_complete,
        "stop_reason": tail_stop_reason if continuation_complete else stop_reason,
        "tail_stop_reason": tail_stop_reason,
        "native_converged": native_converged,
        "fcl_verified": fcl_verified,
        "poses_finite": poses_finite,
        "unconverged_steps": unconverged_steps,
        "notes": notes,
        "diagnostics": diagnostics,
        "tail_log": tail_log,
        "verify_log": verify_log,
        "cleanup_log": cleanup_log,
        "cleanup_time": float(cleanup_time_total),
    }
