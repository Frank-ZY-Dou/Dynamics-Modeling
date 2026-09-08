"""Diagnostic: RoboLab placement (disc solver) versus mesh-level truth, then S4R repair.

For each (n_objects, seed): take the persisted pre-settle layout from RoboLab's own
SpatialSolver (experiments/robolab_offline_layouts.py), stage it in RoboLab's base scene
(base_empty.usda: `table` = table_oak at its authored pose, the robot's `franka_table`,
`GroundPlane`), and report three states with the same evaluator:
  raw          the layout exactly as the skill writes it (z = dims/2 + 2 mm above the table top)
  raw_seated   the same x, y, yaw with every body dropped onto the table (what the repair starts from)
  repaired     S4R under the DSL program implied by the layout's predicates
Writes results/robolab_diag.json and prints a table.
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import simready  # noqa: F401
from simready.scene import Scene, MeshProxy
from simready.gates import verify_scene
from simready.dsl import parse_program, compile_program, check_predicates
from simready.repair import repair_upright
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh
from simready.io.usd_io import body_from_object_usd, load_scene_usda
from robolab_offline_layouts import get_layout, BASE_SCENE, ROBOLAB

REL = {"left-of": "left_of", "right-of": "right_of", "front-of": "in_front_of", "back-of": "behind"}
_BASE = None


def base_scene():
    """RoboLab's base scene: table_oak (`table`, the support), franka_table (robot pedestal), GroundPlane."""
    global _BASE
    if _BASE is None:
        _BASE = load_scene_usda(str(BASE_SCENE))
    return copy.deepcopy(_BASE)


def scene_from_layout(L):
    sc = base_scene()
    bodies = list(sc.bodies)
    for o in L["objects"]:
        bodies.append(body_from_object_usd(o["name"], o["usd_path"], (o["x"], o["y"], o["z"]), yaw_deg=o["yaw"], tags={"object"}))
    return Scene(bodies)


def seat_scene(scene, support="table"):
    """Drop every free body onto the support at its own x, y (no in-plane change)."""
    sc = copy.deepcopy(scene)
    sup = sc[support]
    from simready.dsl.compile import support_height_for
    for b in sc.free():
        h0 = support_height_for(sc, b, sup)
        b.center = np.array([b.center[0], b.center[1], h0 + b.support_offset(sc.up)])
    return sc


def program_from_layout(L, margin=0.005, support="table"):
    lines = ["program", f"  no_penetration(*, margin={margin})", "  fixed(table)  fixed(franka_table)  fixed(GroundPlane)",
             f"  on_support(*, {support})   upright(*)",
             f"  within(*, {support}.top, inset=0.0)"]
    for a, b, rt, dist in L["relations"]:
        lines.append(f"  {REL[rt]}({a}, {b}, gap={dist:.3f})")
    lines.append("  minimize displacement(*)")
    return "\n".join(lines)


def repair_scene(scene, prog, faces=2000, verbose=False):
    sc = copy.deepcopy(scene)
    spec = compile_program(prog, sc, s_min=0.05, ds_max=0.05, tail_iters=30, max_xy_step=0.03)
    tops = {sup: min(h for b, h in spec.supports.items() if spec.support_of.get(b) == sup) for sup in set(spec.support_of.values())}
    t0 = time.time()
    with MeshProxy(sc, faces=faces, slab_tops=tops):
        res = repair_upright(sc, spec, verbose=verbose)
    reseat_on_supports(sc, spec)
    if verify_scene(sc).pen_pairs > 0:
        res2 = polish_full_mesh(sc, spec, iters=15); reseat_on_supports(sc, spec); res.steps += res2.steps
    return sc, res, time.time() - t0, tops


def obj_obj_pairs(rep, scene):
    return sum(1 for x, y, s in rep.pairs if s < 0 and not scene[x].fixed and not scene[y].fixed)


def run_cell(n, seed, verbose=False):
    L = get_layout(n, seed)
    raw = scene_from_layout(L)
    seated = seat_scene(raw)
    prog = parse_program(program_from_layout(L))
    v_raw, v_seated = verify_scene(raw), verify_scene(seated)
    rep, res, dt, tops = repair_scene(seated, prog, verbose=verbose)
    v_rep = verify_scene(rep)
    preds = check_predicates(prog, rep)
    hover = [round(float(b.world_aabb()[0][2] - raw.support_height(raw["table"], at=b.center[:2], radius=0.1)), 4) for b in raw.free()]
    return {"n": n, "seed": seed, "solver_ok": L["ok"], "solver_msg": L["msg"][:60], "objects": [o["name"] for o in L["objects"]],
            "raw": {"pen": v_raw.pen_pairs, "pen_obj_obj": obj_obj_pairs(v_raw, raw), "hover_m": hover},
            "raw_seated": {"pen": v_seated.pen_pairs, "pen_obj_obj": obj_obj_pairs(v_seated, seated)},
            "repaired": {"pen": v_rep.pen_pairs, "pen_obj_obj": obj_obj_pairs(v_rep, rep), "min_signed": v_rep.min_signed,
                         "rmsd_xy": res.rmsd, "steps": res.steps, "time_s": dt, "preds_fail": [p[0] for p in preds if not p[1]]}}


def main():
    ns = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["4", "6", "8", "10"])]
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2"])]
    out = []
    for n in ns:
        for s in seeds:
            r = run_cell(n, s); out.append(r)
            print(f"N={n} seed={s} solver={'ok ' if r['solver_ok'] else 'FAIL'} raw pen {r['raw']['pen']:2d} (obj-obj {r['raw']['pen_obj_obj']}) "
                  f"seated {r['raw_seated']['pen']:2d} ({r['raw_seated']['pen_obj_obj']}) -> repaired {r['repaired']['pen']:2d} "
                  f"rmsd_xy={r['repaired']['rmsd_xy']:.3f} steps={r['repaired']['steps']} {r['repaired']['time_s']:.1f}s predsFail={r['repaired']['preds_fail']}", flush=True)
    Path(HERE.parent / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE.parent / "results" / "robolab_diag.json", "w"), indent=1)
    ok_cells = [r for r in out if r["solver_ok"]]
    print(f"\nsolver-ok cells: {len(ok_cells)}/{len(out)}; with mesh penetration when seated: {sum(1 for r in ok_cells if r['raw_seated']['pen']>0)}"
          f" | all cells repaired to pen=0: {sum(1 for r in out if r['repaired']['pen']==0)}/{len(out)}")


if __name__ == "__main__":
    main()
