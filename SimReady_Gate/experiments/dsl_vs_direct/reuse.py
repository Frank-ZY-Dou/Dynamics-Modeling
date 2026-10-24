"""One answer, many scenes: what a passing answer is worth beyond the scene it was written for.

For every trial of a run whose final scene passed its rubric, the final answer is applied again,
with no further call to the model:
  new starts   to K new starting heaps of the same task (other seeds): a program is solved again
               from each; poses are set again, and give the same layout every time;
  new assets   to the task's scene with every object replaced by another package of its category
               (another can, another apple): the program is solved for the new geometry; the poses are
               set as they were.
Reported per condition: the share of the new scenes that pass the rubric, the solver time per scene
and the spread of the layouts (mean distance between the same body's positions across the new
starts; poses typed once have none). Writes reuse.json next to the trials.

    python experiments/dsl_vs_direct/reuse.py results/dsl_vs_direct/sonnet --starts 5
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def swapped(task):
    """The task with every object drawn from the next package of its category (same names)."""
    import tasks as T
    where = {p: (cat, i) for cat, pk in T.ASSETS["pool"].items() for i, p in enumerate(pk)}
    t2 = copy.copy(task)
    objs = []
    for name, pkg in task.objects:
        if pkg in where:
            cat, i = where[pkg]
            pool = T.ASSETS["pool"][cat]
            objs.append((name, pool[(i + 1) % len(pool)]))
        else:
            objs.append((name, pkg))
    t2.objects = objs
    return t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--starts", type=int, default=5); ap.add_argument("--first-seed", type=int, default=100)
    a = ap.parse_args()
    import tasks as T, scenes as S, pipelines as P
    from simready.cli import load_any
    run = Path(a.run)
    by_key = {t.key: t for t in T.all_tasks()}
    out = []
    for p in sorted(run.glob("trials/*.json")):
        tr = json.loads(p.read_text())
        if not tr["success"] or tr["condition"] not in ("dsl_v1", "direct"):
            continue
        task = by_key[tr["task"]]
        answer = tr["rounds"][-1]["answer"]
        rec = {"task": tr["task"], "condition": tr["condition"], "seed": tr["seed"], "starts": [], "assets": None}
        positions = []
        for k in range(a.starts):
            seed = a.first_seed + k
            path = S.build(task, seed, run / "reuse_layouts")
            start = load_any(str(path))
            if tr["condition"] == "dsl_v1":
                final, report, ok, t = P.run_program(path, answer)
            else:
                final, report, ok, t = P.run_poses(path, answer, task.support)
            passed, checks = P.grade(task, final, start)
            rec["starts"].append({"seed": seed, "rubric_ok": passed, "solve_s": t, "failed": [c[0] for c in checks if not c[1]]})
            positions.append({b.name: b.center[:2].copy() for b in final.free()})
        names = sorted(positions[0])
        spread = [float(np.mean([np.linalg.norm(positions[i][n] - positions[j][n]) for n in names]))
                  for i in range(len(positions)) for j in range(i + 1, len(positions))]
        rec["spread_m"] = float(np.mean(spread)) if spread else 0.0
        t2 = swapped(task)
        path = S.build(t2, tr["seed"], run / "reuse_layouts_assets")
        start = load_any(str(path))
        if tr["condition"] == "dsl_v1":
            final, report, ok, t = P.run_program(path, answer)
        else:
            final, report, ok, t = P.run_poses(path, answer, task.support)
        passed, checks = P.grade(t2, final, start)
        rec["assets"] = {"rubric_ok": passed, "solve_s": t, "failed": [c[0] for c in checks if not c[1]],
                         "swapped": {n: q for (n, q), (_, q0) in zip(t2.objects, task.objects) if q != q0}}
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("task", "condition")} | {"starts_ok": sum(s["rubric_ok"] for s in rec["starts"]),
                          "spread_m": round(rec["spread_m"], 3), "assets_ok": rec["assets"]["rubric_ok"]}), flush=True)
    summary = {}
    for cond in ("dsl_v1", "direct"):
        rs = [r for r in out if r["condition"] == cond]
        if not rs:
            continue
        starts = [s for r in rs for s in r["starts"]]
        summary[cond] = {"answers": len(rs), "new_start_scenes": len(starts), "new_start_pass_rate": float(np.mean([s["rubric_ok"] for s in starts])),
                         "solve_s_per_scene": float(np.mean([s["solve_s"] for s in starts])), "spread_m": float(np.mean([r["spread_m"] for r in rs])),
                         "new_asset_pass_rate": float(np.mean([r["assets"]["rubric_ok"] for r in rs]))}
    (run / "reuse.json").write_text(json.dumps({"summary": summary, "answers": out}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
