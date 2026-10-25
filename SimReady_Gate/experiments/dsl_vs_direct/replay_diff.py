"""A replayed run against the run it replays, trial by trial: what passes in each (the first run as regraded,
when it has a regrade.json), which trials changed, how many rounds kept their recorded answer and how many called
the model again, and the solver's seconds (wall-clock and CPU) per trial in each.

    python experiments/dsl_vs_direct/replay_diff.py <first run dir> <replay dir> [<first run dir> <replay dir> ...] --md diff.md --json diff.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS = ["dsl_v1", "dsl_v0", "direct", "direct+repair"]


def verdicts(run: Path) -> dict:
    """trial name -> (passed, passed with the generic repair or None), the regraded verdicts when the run has them."""
    rg = json.loads((run / "regrade.json").read_text())["trials"] if (run / "regrade.json").exists() else {}
    out = {}
    for p in sorted(run.glob("trials/*.json")):
        t = json.loads(p.read_text())
        g = rg.get(p.stem)
        rep = None
        if t["condition"] == "direct":
            dr = t["rounds"][-1].get("direct_repair") or {}
            rep = bool(g["repair_success"]) if g and "repair_success" in g else bool(dr.get("rubric_ok"))
        out[p.stem] = (bool(g["success"]) if g else bool(t["success"]), rep, t)
    return out


def solver(t: dict, repair: bool = False):
    wall = sum(float(r.get("solve_s") or 0) for r in t["rounds"])
    cpu = sum(float(r.get("solve_cpu_s") or 0) for r in t["rounds"])
    if repair:
        dr = t["rounds"][-1].get("direct_repair") or {}
        wall += float(dr.get("solve_s") or 0); cpu += float(dr.get("solve_cpu_s") or 0)
    return wall, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="first run dir, replay dir, ...")
    ap.add_argument("--md"); ap.add_argument("--json")
    a = ap.parse_args()
    assert len(a.pairs) % 2 == 0, "give the dirs in pairs: first run, replay"
    report, lines = {}, ["| model | condition | trials | pass, first run | pass, replay | changed (lost / gained) | rounds kept / called | solver s / trial, first run (CPU) | solver s / trial, replay (CPU) |",
                         "|---|---|---|---|---|---|---|---|---|"]
    for first, replay in zip(a.pairs[::2], a.pairs[1::2]):
        old, new = verdicts(Path(first)), verdicts(Path(replay))
        model = json.loads((Path(replay) / "run_meta.json").read_text())["model"]
        acc = defaultdict(lambda: {"trials": 0, "pass_old": 0, "pass_new": 0, "lost": [], "gained": [], "kept": 0, "called": 0,
                                   "wall_old": 0.0, "cpu_old": 0.0, "wall_new": 0.0, "cpu_new": 0.0})
        for name, (ok_new, rep_new, t_new) in new.items():
            ok_old, rep_old, t_old = old[name]
            for cond, o, n, repair in ((t_new["condition"], ok_old, ok_new, False),) + (
                    (("direct+repair", rep_old, rep_new, True),) if t_new["condition"] == "direct" else ()):
                c = acc[cond]
                c["trials"] += 1; c["pass_old"] += o; c["pass_new"] += n
                if o and not n:
                    c["lost"].append(name)
                if n and not o:
                    c["gained"].append(name)
                c["kept"] += sum(r.get("answer_from") == "recorded" for r in t_new["rounds"])
                c["called"] += sum(r.get("answer_from") == "called" for r in t_new["rounds"])
                w, u = solver(t_old, repair); c["wall_old"] += w; c["cpu_old"] += u
                w, u = solver(t_new, repair); c["wall_new"] += w; c["cpu_new"] += u
        report[model] = acc
        for cond in CONDITIONS:
            if cond not in acc:
                continue
            c = acc[cond]; n = c["trials"]
            lines.append(f"| {model} | {cond} | {n} | {c['pass_old']}/{n} | {c['pass_new']}/{n} | {len(c['lost'])} / {len(c['gained'])} | "
                         f"{c['kept']} / {c['called']} | {c['wall_old'] / n:.1f} ({c['cpu_old'] / n:.1f}) | {c['wall_new'] / n:.1f} ({c['cpu_new'] / n:.1f}) |")
    text = "\n".join(lines) + "\n"
    print(text)
    for model, acc in report.items():
        for cond, c in acc.items():
            for kind in ("lost", "gained"):
                for name in c[kind]:
                    print(f"{model} {cond} {kind}: {name}")
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
