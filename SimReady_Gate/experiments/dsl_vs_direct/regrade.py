"""Grade a recorded run again with the present rubric, where that can change a verdict.

The rubric's line check ("centres within 1 cm of a common line") measured each centre from the
centres' mean during the runs, which asks for more than the request says; it now measures half the
range of the centres (from the best common line). The check can only be more lenient now, so a
verdict changes only where every check a round failed was a line check. Those rounds' scenes are
rebuilt from the recorded answer (the pipelines are deterministic; no model is called) and graded
again, and every other check of a rebuilt scene must come out as it did in the run (`mismatches`
lists any that does not). Writes regrade.json next to the trials: per trial the verdicts as run and
as regraded, of the final round and of the first, and of direct's generic repair; a finals.json
already written takes the regraded verdicts too.

    python experiments/dsl_vs_direct/regrade.py results/dsl_vs_direct/claude-sonnet-5-5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))

LINE_CHECKS = ("row_aligned", "rows_aligned")


def only_line_checks(failed) -> bool:
    return bool(failed) and all(c[0].startswith(LINE_CHECKS) for c in failed)


def other_failures(checks) -> list:
    return [c[0] for c in checks if not c[1] and not c[0].startswith(LINE_CHECKS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    import tasks as T, scenes as S, pipelines as P
    from simready.cli import load_any
    run = Path(a.run)
    by_key = {t.key: t for t in T.all_tasks()}
    layouts = run / "regrade_layouts"
    trials, mismatches, rebuilt = {}, [], 0
    for p in sorted(run.glob("trials/*.json")):
        tr = json.loads(p.read_text())
        task = by_key[tr["task"]]
        rounds = []
        for r in tr["rounds"]:
            item = {"round": r["round"], "ok_run": bool(r["rubric_ok"]), "ok": bool(r["rubric_ok"])}
            dr = r.get("direct_repair")
            if dr is not None:
                item.update({"repair_ok_run": bool(dr["rubric_ok"]), "repair_ok": bool(dr["rubric_ok"])})
            redo = r["answer"] is not None and only_line_checks(r.get("rubric_failed"))
            redo_repair = dr is not None and only_line_checks(dr.get("rubric_failed"))
            if redo or redo_repair:
                rebuilt += 1
                path = S.build(task, tr["seed"], layouts)
                start = load_any(str(path))
                if tr["condition"] == "direct":
                    final = P.run_poses(path, r["answer"], task.support)[0]
                else:
                    final = P.run_program(path, r["answer"], v0_only=tr["condition"] == "dsl_v0")[0]
                if redo:
                    ok, checks = P.grade(task, final, start)
                    if other_failures(checks):
                        mismatches.append({"trial": p.stem, "round": r["round"], "checks": other_failures(checks)})
                    item["ok"] = ok
                if redo_repair:
                    fixed = P.generic_repair(final, task.support, on=P.typed_supports(r["answer"]))[0]
                    ok2, checks2 = P.grade(task, fixed, start)
                    if other_failures(checks2):
                        mismatches.append({"trial": p.stem, "round": r["round"], "repair": True, "checks": other_failures(checks2)})
                    item["repair_ok"] = ok2
            rounds.append(item)
        rec = {"success_run": bool(tr["success"]), "success": rounds[-1]["ok"],
               "first_run": bool(tr["first_round_success"]), "first": rounds[0]["ok"], "rounds": rounds}
        if "repair_ok" in rounds[-1]:
            rec.update({"repair_success_run": rounds[-1]["repair_ok_run"], "repair_success": rounds[-1]["repair_ok"],
                        "repair_first_run": rounds[0].get("repair_ok_run", False), "repair_first": rounds[0].get("repair_ok", False)})
        trials[p.stem] = rec
    changed = sorted(k for k, v in trials.items()
                     if any(v[f] != v[f + "_run"] for f in ("success", "first", "repair_success", "repair_first") if f in v))
    out = {"rule": "a line check measures half the range of the centres (from the best common line), no longer the distance from their mean",
           "rebuilt_rounds": rebuilt, "changed": changed, "mismatches": mismatches, "trials": trials}
    (run / "regrade.json").write_text(json.dumps(out, indent=1))
    if (run / "finals.json").exists():
        sync_finals(run / "finals.json", trials)
    print(json.dumps({"rebuilt_rounds": rebuilt, "changed": changed, "mismatches": mismatches}, indent=1))


def sync_finals(path: Path, trials: dict) -> None:
    """finals.json with the regraded verdicts: each trial's rubric_ok and the rubric-and-settle rates."""
    import numpy as np
    fin = json.loads(path.read_text())
    for rec in fin["trials"]:
        rec["rubric_ok"] = trials[f"{rec['task']}_{rec['condition']}_s{rec['seed']}"]["success"]
    for cond, s in fin["summary"].items():
        rs = [r for r in fin["trials"] if r["condition"] == cond and "settle" in r]
        s["rubric_and_settle"] = float(np.mean([r["settle"]["pass"] and r["rubric_ok"] for r in rs]))
    path.write_text(json.dumps(fin, indent=1))


if __name__ == "__main__":
    main()
