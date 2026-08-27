"""Scene generation for the S4R benchmarks.

Builds a randomized interpenetrating scene from one of three mesh pools
(kubric, hy3d, thingi) with fixed seeds, runs the S4R solver on it, and
reports penetrating pairs / RMSD / wall time as scored by the shared
mesh-level evaluator.

Usage:
  python make_scene.py --dataset kubric --N 40 --seed 42
"""
import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "s4r"))

KUBRIC_POOL_DIR = (
    SCRIPT_DIR.parent / "data" / "kubric_pool"
)


def make_scene(dataset: str, N: int, seed: int, spawn_xz_mul: float = 1.0):
    sxz_base = 0.15 * (N / 40) ** (1 / 3)
    sy_base = 0.35 * (N / 40) ** (1 / 3)
    sxz = sxz_base * spawn_xz_mul
    sy = sy_base * spawn_xz_mul
    if dataset == "kubric":
        from mesh_collision import generate_kubric_scene
        return generate_kubric_scene(
            n_objects=N, target_size=0.1, spawn_range_xz=sxz, spawn_range_y=sy,
            seed=seed, kubric_dir=str(KUBRIC_POOL_DIR), max_verts=5000,
            allow_repeat=True,
        ), sxz, sy
    if dataset == "hy3d":
        from generate_hy3d_scene import generate_hy3d_scene
        HY3D_DIR = (SCRIPT_DIR.parent / "data" / "hy3d_processed").resolve()
        return generate_hy3d_scene(
            n_objects=N, target_size=0.1, spawn_range_xz=sxz,
            spawn_range_y=sy, seed=seed, hy3d_dir=str(HY3D_DIR),
        ), sxz, sy
    if dataset == "thingi":
        from generate_thingi10k_scene import generate_thingi10k_scene
        return generate_thingi10k_scene(
            n_objects=N, target_size=0.1, spawn_range_xz=sxz,
            spawn_range_y=sy, seed=seed,
        ), sxz, sy
    if dataset == "thingi":
        from generate_thingi10k_scene import generate_thingi10k_scene
        return generate_thingi10k_scene(
            n_objects=N, target_size=0.1, spawn_range_xz=sxz,
            spawn_range_y=sy, seed=seed,
        ), sxz, sy
    raise ValueError(f"unknown dataset: {dataset}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", choices=["kubric", "hy3d", "thingi"],
                    default="kubric")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--spawn-xz-mul", type=float, default=1.0,
                    help="Spawn-box width multiplier for density sweep")
    args = ap.parse_args()

    # Variant -> solver kwargs.
    kw = dict(
        d_hat=0.02, ds_max=0.05, max_steps=200, verbose=False,
        contact_sparsity=True, adaptive_ds=True, revalidate_interval=3,
        contact_backend="fcl", enable_rotation=False,
    )
    audit_mode = False
    if args.variant == "tail_on":
        os.environ["S4R_MAX_TAIL_ITERS"] = "20"
    elif args.variant == "tail_off":
        os.environ["S4R_MAX_TAIL_ITERS"] = "0"
    elif args.variant.startswith("cache_M"):
        M = int(args.variant.split("M")[1])
        kw["revalidate_interval"] = M
        kw["audit"] = True
        audit_mode = True
    elif args.variant == "density":
        # No audit: density sweep only needs pen/rmsd/time, and the
        # per-step audit pass (O(N^2) trimesh contains-checks) makes
        # large-N runs intractable.
        audit_mode = False
    else:
        raise ValueError(f"unknown variant: {args.variant}")

    objs, sxz, sy = make_scene(
        args.dataset, args.N, args.seed, args.spawn_xz_mul,
    )

    from s4r_qp import solve_s4r_qp
    t0 = time.time()
    r = solve_s4r_qp(objs, **kw)
    elapsed = time.time() - t0

    out = {
        "variant": args.variant,
        "dataset": args.dataset,
        "N": args.N,
        "seed": args.seed,
        "spawn_xz_mul": args.spawn_xz_mul,
        "spawn_xz": sxz,
        "spawn_y": sy,
        "pen": int(r.get("pen", -1)),
        "max_pen": float(r.get("max_pen", -1.0)),
        "rmsd": float(r.get("rmsd", -1.0)),
        "time": float(r.get("time", elapsed)),
        "wall_with_setup": float(elapsed),
        "steps": int(r.get("steps", -1)),
    }
    if audit_mode:
        audit = r.get("audit_log", [])
        if audit:
            mb = sum(int(a.get("missed_by_solver", 0)) for a in audit)
            fa = sum(int(a.get("false_active", 0)) for a in audit)
            out["audit_missed_by_solver_total"] = mb
            out["audit_false_active_total"] = fa
            out["audit_steps"] = len(audit)
            out["audit_max_pen_before_max"] = max(
                (a.get("max_pen_before", 0.0) for a in audit), default=0.0,
            )
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    print(
        f"[{args.variant} {args.dataset} N={args.N} seed={args.seed}]  "
        f"pen={out['pen']}  rmsd={out['rmsd']:.4f}  "
        f"time={out['time']:.1f}s  steps={out['steps']}"
    )


if __name__ == "__main__":
    main()
