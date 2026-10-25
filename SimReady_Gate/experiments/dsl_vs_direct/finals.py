"""The final scene of every trial, rebuilt from its recorded answer (the pipelines are deterministic;
no model is called), settled in MuJoCo (G5, convex-hull proxies, 2 s, the assets' friction and mass)
and drawn from above.

    python experiments/dsl_vs_direct/finals.py results/dsl_vs_direct/claude-sonnet-5-5 [--draw task_key ...]

Writes finals.json next to the trials: per trial whether the rubric passed and whether the scene
stays at rest (no body leaving the support, peak speed <= 1.0 m/s, peak displacement <= 0.15 m).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def final_scene(task, trial, layout_dir):
    import scenes as S, pipelines as P
    path = S.build(task, trial["seed"], layout_dir)
    answer = trial["rounds"][-1]["answer"]
    if answer is None:
        return None
    if trial["condition"] == "direct":
        return P.run_poses(path, answer, task.support)[0]
    return P.run_program(path, answer, v0_only=trial["condition"] == "dsl_v0")[0]


def settle(sc, support):
    from simready.gates.settle_mujoco import settle_and_measure
    from simready.cli import quiet_stdout
    tops = {support: sc.support_height(sc[support], at=np.mean([b.center[:2] for b in sc.free()], axis=0), radius=1.0)}
    with quiet_stdout():
        r = settle_and_measure(sc, seconds=2.0, support_tops=tops)
    return {"pass": (not r.left_support) and r.peak_speed <= 1.0 and r.peak_disp <= 0.15 and not r.engine_warnings,
            "peak_speed": round(r.peak_speed, 3), "peak_disp": round(r.peak_disp, 4), "left_support": r.left_support}


def draw(sc, path, title):
    """Top view: the support's top in grey, every free body's footprint hull."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from simready.dsl.statements import footprint_hull
    fig, ax = plt.subplots(figsize=(4.6, 2.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for b in sc.fixed_bodies():
        if "support" in b.tags:
            lo, hi = b.world_aabb()
            ax.add_patch(Polygon([[lo[1], lo[0]], [hi[1], lo[0]], [hi[1], hi[0]], [lo[1], hi[0]]], closed=True, fc="#ecebe7", ec="#c9c8c2", lw=0.8))
    for b in sc.free():
        h = footprint_hull(b)
        ax.add_patch(Polygon(np.column_stack([h[:, 1], h[:, 0]]), closed=True, fc="#2a78d6", ec="#fcfcfb", lw=1.0, alpha=0.85))
    ax.set_aspect("equal"); ax.invert_xaxis()                      # robot at the bottom, +y to the left
    ax.set_xlim(0.95, -0.95); ax.set_ylim(0.15, 0.95)
    ax.set_xlabel("y (m)", fontsize=6, color="#52514e"); ax.set_ylabel("x (m)", fontsize=6, color="#52514e")
    ax.set_title(title, fontsize=8, color="#0b0b0b")
    ax.tick_params(labelsize=6, colors="#52514e")
    for s_ in ax.spines.values():
        s_.set_color("#c9c8c2")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--draw", nargs="*", default=[]); ap.add_argument("--no-settle", action="store_true")
    ap.add_argument("--tasks", nargs="*", help="only these task keys")
    a = ap.parse_args()
    import tasks as T
    run = Path(a.run)
    by_key = {t.key: t for t in T.all_tasks()}
    out = []
    for p in sorted(run.glob("trials/*.json")):
        tr = json.loads(p.read_text())
        if a.tasks and tr["task"] not in a.tasks:
            continue
        task = by_key[tr["task"]]
        sc = final_scene(task, tr, run / "finals_layouts")
        rec = {"task": tr["task"], "condition": tr["condition"], "seed": tr["seed"], "rubric_ok": tr["success"]}
        if sc is not None and not a.no_settle:
            rec["settle"] = settle(sc, task.support)
        if sc is not None and tr["task"] in a.draw and tr["seed"] == 0:
            (run / "figures").mkdir(exist_ok=True)
            draw(sc, run / "figures" / f"{tr['task']}_{tr['condition']}.png", f"{tr['condition']}: {'passes' if tr['success'] else 'fails'} the rubric")
        out.append(rec)
        print(json.dumps(rec), flush=True)
    summary = {}
    for cond in sorted({r["condition"] for r in out}):
        rs = [r for r in out if r["condition"] == cond and "settle" in r]
        if rs:
            summary[cond] = {"trials": len(rs), "settle_pass": float(np.mean([r["settle"]["pass"] for r in rs])),
                             "rubric_and_settle": float(np.mean([r["settle"]["pass"] and r["rubric_ok"] for r in rs]))}
    (run / "finals.json").write_text(json.dumps({"summary": summary, "trials": out}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
