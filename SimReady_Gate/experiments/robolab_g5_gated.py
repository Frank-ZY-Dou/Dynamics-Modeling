"""Agent route on the G5 sweep: for every cell whose repaired scene has a body in `left_support`,
add `within(<body>, table.top, inset=0.10)` to the program, repair again, settle again.
Reads results/robolab_g5.json (from robolab_g5_sweep.py), writes results/robolab_g5_gated.json."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "experiments"); import simready  # noqa: E402,F401
from simready.scene import MeshProxy
from simready.gates import verify_scene
from simready.gates.settle_mujoco import settle_and_measure
from simready.dsl import parse_program, compile_program, check_predicates
from simready.repair import repair_upright
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh
from robolab_offline_layouts import make_layout
from robolab_diagnostic import scene_from_layout, program_from_layout

INSET = 0.10
rows = json.load(open("results/robolab_g5.json")); out = []
for r in rows:
    off_support = sorted(set(r.get("rep_hull_left", [])) | set(r.get("rep_coacd_left", [])))
    if not off_support:
        continue
    n, seed = r["n"], r["seed"]
    L = make_layout(np.random.RandomState(seed), n_objects=n); sc = scene_from_layout(L)
    text = program_from_layout(L).rstrip() + "".join(f"\n  within({c}, table.top, inset={INSET})" for c in off_support) + "\n"
    prog = parse_program(text); spec = compile_program(prog, sc, s_min=0.05, ds_max=0.05, tail_iters=30)
    tops = {sup: min(h for b, h in spec.supports.items() if spec.support_of.get(b) == sup) for sup in set(spec.support_of.values())}
    t0 = time.time()
    with MeshProxy(sc, faces=2000, slab_tops=tops):
        repair_upright(sc, spec)
    reseat_on_supports(sc, spec)
    if verify_scene(sc).pen_pairs > 0:
        polish_full_mesh(sc, spec, iters=15); reseat_on_supports(sc, spec)
    rec = {"n": n, "seed": seed, "off_support": off_support, "pen": verify_scene(sc).pen_pairs, "repair_s": round(time.time() - t0, 1),
           "fails": [p[0] for p in check_predicates(prog, sc) if not p[1]]}
    for mode in ("hull", "coacd"):
        g = settle_and_measure(sc, seconds=2.0, support_tops=tops, decompose_free=(mode == "coacd"))
        rec[f"{mode}_vmax"], rec[f"{mode}_dmax"], rec[f"{mode}_left"] = g.peak_speed, g.peak_disp, list(g.left_support)
    out.append(rec)
    print(f"N={n} s={seed} off_support={off_support} -> pen={rec['pen']} fails={rec['fails']} repair {rec['repair_s']}s | "
          f"hull {rec['hull_vmax']:.2f} m/s {rec['hull_dmax']:.3f} m left={rec['hull_left']} | coacd {rec['coacd_vmax']:.2f} m/s {rec['coacd_dmax']:.3f} m left={rec['coacd_left']}", flush=True)
Path("results").mkdir(exist_ok=True); json.dump(out, open("results/robolab_g5_gated.json", "w"), indent=1)
print("cells routed:", len(out), "| pass after route:", sum(1 for o in out if not o["hull_left"] and not o["coacd_left"]))
