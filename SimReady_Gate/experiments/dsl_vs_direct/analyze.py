"""Tables of the comparison from the trial files run.py writes.

Per condition: the share of trials whose final scene passes the rubric, the share that pass at the
first round, rounds used, and what a trial costs: prompt tokens (fresh and read from the prompt
cache), output tokens (thinking included), the CLI's cost at list price, the time spent in the model
and in the solver. "Per success" divides a condition's total by its number of passing trials: what
one passing scene costs. direct+repair is direct's last answer repaired by S4R with the shared
geometric statements only; its cost is direct's plus the repair time. Solver time is wall-clock, with
the worker's CPU seconds beside it; the held-out tasks are also tabulated apart.

    python experiments/dsl_vs_direct/analyze.py results/dsl_vs_direct/sonnet [more run dirs] --md summary.md --json summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["dsl_v1", "dsl_v0", "direct", "direct+repair"]
HELD_OUT = {"tool_layout", "desk"}          # tasks.HELD_OUT: written after the language was frozen


def load(dirs):
    trials = []
    for d in dirs:
        for p in sorted(Path(d).glob("trials/*.json")):
            trials.append(json.loads(p.read_text()))
    return trials


def per_trial(t):
    rounds = t["rounds"]
    u = defaultdict(float)
    for r in rounds:
        us = (r["llm"] or {}).get("usage") or {}
        for k in ("input", "cache_creation", "cache_read", "output", "thinking"):
            u[k] += float(us.get(k) or 0)
        u["cost"] += float((r["llm"] or {}).get("cost_usd") or 0)
        u["llm_s"] += float((r["llm"] or {}).get("duration_ms") or 0) / 1000.0
        u["solve_s"] += float(r.get("solve_s") or 0)
        u["solve_cpu_s"] += float(r.get("solve_cpu_s") or 0)
    rows = [{"condition": t["condition"], "task": t["task"], "family": t["family"], "n": t["n"], "seed": t["seed"], "model": t["model"],
             "success": bool(t["success"]), "first": bool(t["first_round_success"]), "rounds": len(rounds),
             "self_ok": bool(t["self_ok"]), "prompt": u["input"] + u["cache_creation"] + u["cache_read"],
             "prompt_fresh": u["input"] + u["cache_creation"], "output": u["output"], "thinking": u["thinking"], "cost": u["cost"],
             "llm_s": u["llm_s"], "solve_s": u["solve_s"], "solve_cpu_s": u["solve_cpu_s"], "items": rounds[-1].get("answer_items"),
             "split": "held_out" if t["family"] in HELD_OUT else "development"}]
    if t["condition"] == "direct" and rounds[-1].get("direct_repair"):
        dr = rounds[-1]["direct_repair"]
        r2 = dict(rows[0]); r2["condition"] = "direct+repair"
        r2["success"] = bool(dr["rubric_ok"]); r2["solve_s"] = rows[0]["solve_s"] + float(dr.get("solve_s") or 0)
        r2["solve_cpu_s"] = rows[0]["solve_cpu_s"] + float(dr.get("solve_cpu_s") or 0)
        r2["first"] = bool((rounds[0].get("direct_repair") or {}).get("rubric_ok"))
        rows.append(r2)
    return rows


def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson interval of a pass rate k / n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows, key=lambda r: r["condition"]):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for k, g in groups.items():
        n, s = len(g), sum(r["success"] for r in g)
        tot = {f: float(sum(r[f] for r in g)) for f in ("prompt", "prompt_fresh", "output", "thinking", "cost", "llm_s", "solve_s", "solve_cpu_s")}
        out[k] = {"trials": n, "passed": s, "success_rate": s / n, "success_ci95": wilson(s, n), "first_round_rate": sum(r["first"] for r in g) / n,
                  "self_check_rate": sum(r["self_ok"] for r in g) / n, "rounds_mean": float(np.mean([r["rounds"] for r in g])),
                  "per_trial": {f: v / n for f, v in tot.items()}, "per_success": {f: (v / s if s else None) for f, v in tot.items()},
                  "time_per_trial_s": (tot["llm_s"] + tot["solve_s"]) / n,
                  "time_per_success_s": (tot["llm_s"] + tot["solve_s"]) / s if s else None}
    return out


def md_table(summ, title):
    lines = [f"### {title}", "", "| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER + sorted(k for k in summ if k not in ORDER):
        if c not in summ:
            continue
        v = summ[c]; p, q = v["per_trial"], v["per_success"]
        fmt = lambda x, f="{:.0f}": "—" if x is None else f.format(x)
        lo, hi = v["success_ci95"]
        lines.append(f"| {c} | {v['trials']} | {v['passed']}/{v['trials']} ({100 * lo:.0f}-{100 * hi:.0f}%) | {100 * v['first_round_rate']:.0f}% | {v['rounds_mean']:.2f} | "
                     f"{p['output']:.0f} | {p['prompt']:.0f} ({p['prompt_fresh']:.0f}) | ${p['cost']:.3f} | {p['llm_s']:.0f} s | {p['solve_s']:.1f} s ({p['solve_cpu_s']:.1f} s) | "
                     f"{fmt(q['output'])} | {fmt(v['time_per_success_s'], '{:.0f} s')} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--md"); ap.add_argument("--json")
    a = ap.parse_args()
    trials = load(a.dirs)
    rows = [r for t in trials for r in per_trial(t)]
    models = sorted({r["model"] for r in rows})
    report = {"models": models, "overall": {}, "by_family": {}, "by_size": {}}
    md = []
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        report["overall"][m] = summarize(mr)
        md.append(md_table(report["overall"][m], f"{m}: all tasks"))
        for split in ("development", "held_out"):
            sr = [r for r in mr if r["split"] == split]
            if sr and len(sr) < len(mr):
                report.setdefault("by_split", {}).setdefault(m, {})[split] = summarize(sr)
                md.append(md_table(report["by_split"][m][split], f"{m}: {split.replace('_', '-')} tasks"))
        fam = summarize(mr, key=lambda r: (r["family"], r["n"], r["condition"]))
        report["by_family"][m] = {f"{f}_n{n}|{c}": v for (f, n, c), v in fam.items()}
        lines = [f"### {m}: pass rate by task", "", "| task | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
        for (f, n) in sorted({(f, n) for (f, n, c) in fam}):
            cells = []
            for c in ORDER:
                v = fam.get((f, n, c))
                cells.append("—" if v is None else f"{100 * v['success_rate']:.0f}% ({v['per_trial']['output']:.0f} tok, {v['per_trial']['llm_s'] + v['per_trial']['solve_s']:.0f} s)")
            lines.append(f"| {f} (N={n}) | " + " | ".join(cells) + " |")
        md.append("\n".join(lines) + "\n")
        # scaling: the families run at several sizes
        sizes = sorted({(f, n) for (f, n, c) in fam})
        multi = sorted({f for f, n in sizes if sum(1 for g, _ in sizes if g == f) > 1})
        lines = [f"### {m}: by scene size", "", "| task | N | " + " | ".join(f"{c} pass, output tokens, model s" for c in ("dsl_v1", "direct")) + " |",
                 "|---|---|---|---|"]
        for f in multi:
            for (g, n) in sizes:
                if g != f:
                    continue
                cells = []
                for c in ("dsl_v1", "direct"):
                    v = fam.get((g, n, c))
                    cells.append("—" if v is None else f"{v['passed']}/{v['trials']}, {v['per_trial']['output']:.0f}, {v['per_trial']['llm_s']:.0f}")
                lines.append(f"| {f} | {n} | " + " | ".join(cells) + " |")
        md.append("\n".join(lines) + "\n")
    text = "\n".join(md)
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
