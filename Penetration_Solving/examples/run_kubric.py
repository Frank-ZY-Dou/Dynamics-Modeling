"""Minimal end-to-end example: resolve one Kubric scene with S4R (CPU).

Generates a randomized interpenetrating scene from the bundled 41-mesh
Kubric pool, runs the progressive-scaling solver, and scores the result
with the shared mesh-level evaluator.

Usage:
  python run_kubric.py --N 40 --seed 42
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s4r"))
sys.path.insert(0, str(HERE.parent / "scenes"))

import numpy as np
from make_scene import make_scene
from s4r_qp import solve_s4r_qp
from mesh_collision import evaluate_mesh_object_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="kubric", choices=["kubric", "hy3d", "thingi"])
    args = ap.parse_args()

    objs, _, _ = make_scene(args.dataset, args.N, args.seed, 1.0)
    init = evaluate_mesh_object_scene(objs)
    print(f"[init]  N={args.N} seed={args.seed}  penetrating pairs = {init.pen_pairs}")

    t0 = time.time()
    res = solve_s4r_qp(objs, d_hat=0.02, ds_max=0.05, max_steps=200,
                       adaptive_ds=True, contact_sparsity=True,
                       revalidate_interval=3,
                       contact_backend="fcl_prebuilt", verbose=False)
    dt = time.time() - t0

    # The solver returns the resolved poses; write them back and re-score
    # with the shared evaluator so the report is independent of the solver.
    for o, c in zip(objs, res["final_centers"]):
        o.center = np.asarray(c, dtype=float)
    if res.get("final_rotations") is not None:
        for o, R in zip(objs, res["final_rotations"]):
            o.rotation = np.asarray(R, dtype=float)
    final = evaluate_mesh_object_scene(objs)
    print(f"[final] penetrating pairs = {final.pen_pairs}   "
          f"RMSD = {res['rmsd']:.4f}   solve = {dt:.2f}s")


if __name__ == "__main__":
    main()
