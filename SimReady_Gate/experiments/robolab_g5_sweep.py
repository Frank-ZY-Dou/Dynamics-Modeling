"""G5 on RoboLab pre-settle layouts: settle in MuJoCo and measure, three states per cell.

  raw          the layout as the skill writes it (bodies hover at z = dims/2 + 2 mm)
  raw_seated   the same x, y, yaw dropped onto the table (the like-for-like baseline: the repair
               starts from this state, so any difference to `repaired` is the repair's doing)
  repaired     after S4R on the seated layout
Every state uses the identical harness (base scene from base_empty.usda, table slab at the
measured top, floor, proxies, density, timestep). Per body peak speeds are kept so that a cell's
verdict can be attributed (hover drop, toppling, a body off the table, an ejection).
Writes results/robolab_g5.json. Pass criterion used in the notes: no body off the table AND
peak speed <= 1.0 m/s AND peak displacement <= 0.15 m (the CLI defaults), stated explicitly.
"""
import json, sys, time, statistics as st
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); import simready  # noqa
from simready.gates import verify_scene
from simready.gates.settle_mujoco import settle_and_measure
from simready.dsl import parse_program
from robolab_diagnostic import scene_from_layout, seat_scene, program_from_layout, repair_scene
from robolab_offline_layouts import get_layout

V_MAX, D_MAX = 1.0, 0.15


def g5(sc, tops, mode):
    r = settle_and_measure(sc, seconds=2.0, support_tops=tops, decompose_free=(mode == "coacd"))
    worst = max(r.per_body_peak_speed.items(), key=lambda kv: kv[1]) if r.per_body_peak_speed else ("-", 0.0)
    return {"vmax": r.peak_speed, "dmax": r.peak_disp, "left": list(r.left_support), "worst_body": worst[0], "worst_v": worst[1],
            "pass": (not r.left_support) and r.peak_speed <= V_MAX and r.peak_disp <= D_MAX}


def cell(n, seed):
    L = get_layout(n, seed)
    raw = scene_from_layout(L); seated = seat_scene(raw)
    prog = parse_program(program_from_layout(L))
    rep, res, dt, tops = repair_scene(seated, prog)
    out = {"n": n, "seed": seed, "pen_raw": verify_scene(raw).pen_pairs, "pen_seated": verify_scene(seated).pen_pairs, "pen_rep": verify_scene(rep).pen_pairs,
           "rmsd_xy": res.rmsd, "repair_s": dt}
    for tag, sc in (("raw", raw), ("seated", seated), ("rep", rep)):
        for mode in ("hull", "coacd"):
            out[f"{tag}_{mode}"] = g5(sc, tops, mode)
    return out


if __name__ == "__main__":
    rows = []
    for n in (4, 6, 8, 10):
        for s in (0, 1, 2):
            t0 = time.time(); r = cell(n, s); rows.append(r)
            print(f"N={n} s={s} pen raw {r['pen_raw']} seated {r['pen_seated']} -> rep {r['pen_rep']} | " +
                  " | ".join(f"{k}: {r[k]['vmax']:.2f} m/s {r[k]['dmax']:.3f} m left={r[k]['left']} ({r[k]['worst_body']})"
                             for k in ("raw_coacd", "seated_coacd", "rep_coacd")) + f"  ({time.time()-t0:.0f}s)", flush=True)
    Path(HERE.parent / "results").mkdir(exist_ok=True); json.dump(rows, open(HERE.parent / "results" / "robolab_g5.json", "w"), indent=1)
    for key in ("raw_hull", "raw_coacd", "seated_hull", "seated_coacd", "rep_hull", "rep_coacd"):
        print(f"{key:13s} vmax median {st.median(r[key]['vmax'] for r in rows):.3f} max {max(r[key]['vmax'] for r in rows):.3f} | "
              f"dmax median {st.median(r[key]['dmax'] for r in rows):.4f} max {max(r[key]['dmax'] for r in rows):.4f} | "
              f"cells with a body off the table {sum(1 for r in rows if r[key]['left'])} | pass {sum(1 for r in rows if r[key]['pass'])}/{len(rows)}")
