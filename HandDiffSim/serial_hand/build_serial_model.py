#!/usr/bin/env python
"""Build the simplified serial-joint AmazingHand from the official closed-chain MJCF.

Steps: probe linkage -> sample its quasi-static motion on a motor grid -> extract equivalent serial joint angles
-> identify couplings / motor maps / RSS closure parameters -> generate serial MJCF -> validate motion.

    python build_serial_model.py --side both            # ~1 min per hand
    python build_serial_model.py --side right --quick   # coarse grid for a fast check
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ah_serial.build_mjcf import build_serial_mjcf
from ah_serial.identify import format_report, identify_finger
from ah_serial.kinematics import SerialHandKinematics
from ah_serial.linkage import LinkageHand
from ah_serial.validate import SerialModel, compare_with_linkage, dynamic_coupling_check, format_validation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "AmazingHand"
SCENES = {"right": REPO / "Demo/AHSimulation/AHSimulation/AH_Right/mjcf/scene.xml",
          "left": REPO / "Demo/AHSimulation/AHSimulation/AH_Left/mjcf/scene.xml"}


def build(side: str, step_deg: float, theta_max_deg: float, out_root: Path):
    t0 = time.time()
    H = LinkageHand(SCENES[side])
    print(H.describe())
    grid = np.radians(np.arange(-theta_max_deg, theta_max_deg + 1e-9, step_deg))
    theta = np.array([(a, b) for a in grid for b in grid])
    print(f"\nsampling {len(theta)} motor pairs ({len(grid)}x{len(grid)}, step {step_deg} deg) ...")
    S = H.sample_motor_grid(theta)
    print(f"ok fraction per finger: {S.ok.mean(axis=1).round(3)}, max closure error (ok) {S.closure_mm[S.ok].max():.4f} mm, "
          f"max hinge residual {np.degrees(S.resid[S.ok].max()):.4f} deg")
    signs = H.apply_sign_conventions(S)
    print(f"axis sign conventions (1 = as exported): {signs}\n")

    fits, valid, margin = {}, {}, {}
    for i, f in enumerate(H.fingers):
        fits[f.n], valid[f.n], margin[f.n] = identify_finger(H, S, i)
        print(format_report(f.n, H.geometry[i], fits[f.n]))

    params = {"side": side, "source_scene": str(H.scene_xml), "palm_body": H.palm_name,
              "sampling": {"step_deg": step_deg, "theta_max_deg": theta_max_deg, "n_pairs": int(len(theta))},
              "fingers": {str(f.n): {"geometry": g.to_json(), "fits": fits[f.n]} for f, g in zip(H.fingers, H.geometry)}}
    model_dir = out_root / "models" / f"AH_{side.capitalize()}"
    model_path, scene_path = build_serial_mjcf(H, fits, model_dir, f"AH_{side}_serial")
    print(f"\nwrote {model_path.relative_to(out_root.parent)} and {scene_path.relative_to(out_root.parent)}")

    K = SerialHandKinematics(params)
    serial = SerialModel(scene_path)
    m = serial.model
    print(f"serial model: nbody={m.nbody} njnt={m.njnt} nq={m.nq} nu={m.nu} neq={m.neq} ngeom={m.ngeom} nmesh={m.nmesh}")
    summary = compare_with_linkage(H, S, valid, K, serial)
    dyn = dynamic_coupling_check(serial, K)
    print(format_validation(summary, dyn))
    params["validation"] = {"tip_error": {mth: {str(n): v for n, v in d.items()} for mth, d in summary.items()
                                          if mth in ("exact_joints", "motors_newton", "motors_poly")},
                            "body_pos_max_mm_exact_joints": {str(n): v for n, v in summary["body_pos_max_mm_exact_joints"].items()},
                            "numpy_fk_vs_mujoco_max_mm": {str(n): v for n, v in summary["numpy_fk_vs_mujoco_max_mm"].items()},
                            "dynamic_coupling": dyn}

    params_dir = out_root / "params"
    params_dir.mkdir(exist_ok=True)
    with open(params_dir / f"identified_{side}.json", "w") as fh:
        json.dump(params, fh, indent=1)
    np.savez_compressed(params_dir / f"samples_{side}.npz", theta=S.theta, ok=S.ok, q=S.q, resid=S.resid, tip=S.tip,
                        closure_mm=S.closure_mm, valid=np.stack([valid[n] for n in (1, 2, 3, 4)]),
                        dc_margin=np.stack([margin[n] for n in (1, 2, 3, 4)]),
                        **{f"pose_{k}": v for k, v in S.pose.items()})
    print(f"wrote params/identified_{side}.json and params/samples_{side}.npz   ({time.time() - t0:.1f}s total)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["right", "left", "both"], default="both")
    ap.add_argument("--step-deg", type=float, default=5.0, help="motor grid step")
    ap.add_argument("--theta-max-deg", type=float, default=90.0)
    ap.add_argument("--quick", action="store_true", help="15 deg grid")
    args = ap.parse_args()
    step = 15.0 if args.quick else args.step_deg
    for side in (["right", "left"] if args.side == "both" else [args.side]):
        print("=" * 100 + f"\n{side.upper()} HAND\n" + "=" * 100)
        build(side, step, args.theta_max_deg, HERE)


if __name__ == "__main__":
    main()
