"""The next round of the agent loop of an example, written by a model: the request and the scene, the previous program and
what the repair found, as the comparison's dsl_v1_rewrites condition is told it (the report with the rewrites), in one
isolated call of the comparison (experiments/dsl_vs_direct/llm.py). The system prompt asks for the justifications, as
the CLI's prompt does. Writes program_round<K+1>.json and agent_round<K+1>_call.json (the message and the call).

    python viz/agent_revise.py results/videos/pile_n20 1 --model claude-opus-5-5

The answer format carries the schedule's margin only; the other schedule entries of the previous program (the
example's s_min, which sets how large the bodies are while they are placed) are kept, and the record says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "dsl_vs_direct"))
from simready.cli import load_any  # noqa: E402
from simready.dsl.schema import op_schema, system_prompt, user_prompt  # noqa: E402
import llm  # noqa: E402
from pipelines import REWRITES_NOTE, report_of  # noqa: E402
from run import round_message, sanitize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("round", type=int, help="the round whose program and repair report are read")
    ap.add_argument("--model", required=True); ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    d, k = Path(a.dir), a.round
    prev = json.loads((d / f"program_round{k}.json").read_text())
    text = (d / f"agent_round{k}_repair.txt").read_text()
    rep = json.loads(text[:text.rindex("}") + 1])
    report = report_of(rep["ok"], rep["pen_after"], rep["predicates"], rep["preferences"], rep["solver_notes"])
    if rep["failed_predicates"]:
        report.update({key: rep[key] for key in ("rewrites", "accepted_if_relaxed", "accepted_if_weakened")}, rewrites_note=REWRITES_NOTE)
        report["schedule_rewrites"] = rep["schedule_rewrites"]
    msg = round_message(user_prompt((d / "request.txt").read_text().strip(), load_any(str(d / "layout.json"))), prev, report)
    system = system_prompt(with_schema=False)
    call = llm.call(system, msg, a.model, a.effort, schema=op_schema())
    answer = call.get("structured")
    if not call.get("ok") or not isinstance(answer, dict):
        sys.exit(f"the call gave no program: {call.get('error') or call.get('subtype')}")
    given = json.loads(json.dumps(answer))
    kept = {key: v for key, v in prev.get("schedule", {}).items() if key != "margin"}
    if kept:
        answer["schedule"] = {**answer.get("schedule", {}), **kept}
    (d / f"program_round{k + 1}.json").write_text(json.dumps(answer, indent=1) + "\n")
    record = {"model": a.model, "effort": a.effort, "cli": llm.cli_version(), "flags": llm.FLAGS, "system_sha256": call["system_sha256"],
              "message": msg, "answer": given, "schedule_kept": kept,
              **{key: call.get(key) for key in ("usage", "cost_usd", "duration_ms", "wall_s", "stop_reason")}}
    (d / f"agent_round{k + 1}_call.json").write_text(json.dumps(sanitize(record), indent=1) + "\n")
    print(json.dumps({"program": str(d / f"program_round{k + 1}.json"), "statements": len(answer.get("statements", [])),
                      "schedule": answer.get("schedule"), "usage": call.get("usage")}))


if __name__ == "__main__":
    main()
