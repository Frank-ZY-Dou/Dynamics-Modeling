"""The comparison: the same requests on the same SimReady scenes, answered by the same model through
isolated CLI calls (llm.py), in three conditions.

  dsl_v1   the model writes a SimReady program (every statement); the gate compiles and repairs it
  dsl_v0   the same with the statements the language had before v1 (no rows, grids, headings, regions)
  direct   the model writes the final pose of every free body; no program, no solver
  (direct+repair, computed from direct's answers without another call: its poses repaired by S4R with
   only the geometric statements every task shares)

A trial is one task, one seed, one condition. It has up to --rounds rounds; each round is a new call
whose message repeats the request with the previous answer and what that condition's own check found
(the gate's report on the program, or the verifier's report on the poses). A trial stops when that
check passes. The hidden rubric (tasks.py) grades every round's scene; no model sees it.

    python experiments/dsl_vs_direct/run.py --model claude-sonnet-5-5 --seeds 0 1 2 --workers 4 --out results/dsl_vs_direct/sonnet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))

DIRECT_SYSTEM = """You lay out objects on a table for a robot. Output ONLY a JSON object in the required format: the final pose
of every free body listed in the scene.

Frame: metres and degrees, z up. The robot base is at (0, 0) and looks along +x: +x is away from the
robot ("far"), -x toward it ("near"), +y is left, -y right.

For every free body give x and y, the final position of its reference centre (the centre of its
bounding box, as listed with the scene), and yaw, the final heading in degrees: the turn about z of
the body's local frame from the scene frame (the scene lists each body's local size and present
heading; at yaw 0 its local x axis points along +x). The body is set down upright on the surface
under its footprint, so z is computed for you; to rest a body on top of another, give "on": <that body>.
Bodies must not interpenetrate and must stay on the support: work out each footprint from the local
size and the yaw. Satisfy every part of the request; its numbers are in metres and degrees.
"""


# The answer formats carry no scene names, so that a condition's prompt prefix (format and system
# prompt) is the same for every call and the prompt cache serves it; names are checked on arrival.
DIRECT_SCHEMA = {"type": "object", "properties": {"poses": {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}, "yaw": {"type": "number"},
    "on": {"type": "string", "description": "the body this one rests on, when not the support"}},
    "required": ["name", "x", "y", "yaw"], "additionalProperties": False}}}, "required": ["poses"], "additionalProperties": False}


def conditions():
    from simready.dsl.schema import system_prompt, V0_OPS
    return {"dsl_v1": system_prompt(with_schema=False, reasons=False), "dsl_v0": system_prompt(ops=V0_OPS, with_schema=False, reasons=False),
            "direct": DIRECT_SYSTEM}


def schemas():
    from simready.dsl.schema import op_schema, V0_OPS
    return {"dsl_v1": op_schema(), "dsl_v0": op_schema(ops=V0_OPS), "direct": DIRECT_SCHEMA}


def sanitize(obj, home=str(Path.home())):
    """Results keep no e-mail address and no local directory."""
    text = json.dumps(obj)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<email>", text)
    text = text.replace(home, "~")
    return json.loads(text)


def run_trial(job):
    import tasks as T, scenes as S, pipelines as P, llm
    from simready.cli import load_any
    from simready.dsl.schema import user_prompt
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    task = {t.key: t for t in T.all_tasks()}[job["task"]]
    cond, seed = job["condition"], job["seed"]
    layout = S.build(task, seed, Path(job["out"]) / "layouts")
    start = load_any(str(layout))
    system = conditions()[cond]
    schema = schemas()[cond]
    base = user_prompt(task.request, start)
    # a replay (--replay) keeps the recorded answer of every round whose message is the recorded one: the first
    # (task and scene alone), and a later one when the answer before it was the recorded one and checking it found
    # word for word what it found then; any other round calls the model, as a run does
    recorded = json.loads(Path(job["replay_of"]).read_text())["rounds"] if job.get("replay_of") else None
    rounds, prev, prev_recorded = [], None, True
    canon = lambda x: json.dumps(x, sort_keys=True)
    for r in range(1, job["rounds"] + 1):
        msg = base if prev is None else (base + "\nYour previous answer:\n" + json.dumps(prev["answer"]) +
                                         "\n\nWhat checking it found:\n" + json.dumps(prev["report"]) + "\n\nGive a corrected, complete answer.\n")
        old = recorded[r - 1] if recorded is not None and r <= len(recorded) else None
        same = (old is not None and prev_recorded and len(msg) == old["message_chars"]
                and (r == 1 or canon(sanitize(prev["report"])) == canon(recorded[r - 2]["report"])))
        if same:
            answer, llm_rec = old["answer"], old["llm"]
        else:
            call = llm.call(system, msg, job["model"], job["effort"], schema=schema, work_root=job["work_root"])
            answer = call.get("structured")
            llm_rec = {k: call.get(k) for k in ("ok", "wall_s", "duration_ms", "duration_api_ms", "cost_usd", "usage", "num_turns", "stop_reason", "error")}
        prev_recorded = same
        rec = {"round": r, "llm": llm_rec, "message_chars": len(msg), "answer": answer}
        if recorded is not None:
            rec["answer_from"] = "recorded" if same else "called"
        clock = {}
        if answer is None:
            report, self_ok, final, t_solve = {"ok": False, "error": "no structured answer"}, False, start, 0.0
        elif cond == "direct":
            final, report, self_ok, t_solve = P.run_poses(layout, answer, task.support, clock=clock)
        else:
            final, report, self_ok, t_solve = P.run_program(layout, answer, v0_only=cond == "dsl_v0", clock=clock)
        passed, checks = P.grade(task, final, start)
        rec.update({"report": report, "self_ok": self_ok, "solve_s": t_solve, "solve_cpu_s": clock.get("cpu_s", 0.0), "rubric_ok": passed,
                    "rubric_failed": [c for c in checks if not c[1]],
                    "answer_items": len(answer.get("statements", answer.get("poses", []))) if isinstance(answer, dict) else None})
        if cond == "direct" and answer is not None:
            fix_clock = {}
            fixed, gr_ok, t_fix = P.generic_repair(final, task.support, clock=fix_clock)
            p2, c2 = P.grade(task, fixed, start)
            rec["direct_repair"] = {"rubric_ok": p2, "self_ok": gr_ok, "solve_s": t_fix, "solve_cpu_s": fix_clock.get("cpu_s", 0.0),
                                    "rubric_failed": [c for c in c2 if not c[1]]}
        rounds.append(rec)
        prev = {"answer": answer, "report": report}
        if self_ok:
            break
    out = {"task": task.key, "family": task.name, "n": task.n, "condition": cond, "seed": seed, "model": job["model"], "effort": job["effort"],
           "rounds": rounds, "n_rounds": len(rounds), "success": rounds[-1]["rubric_ok"], "first_round_success": rounds[0]["rubric_ok"],
           "self_ok": rounds[-1]["self_ok"]}
    path = Path(job["out"]) / "trials" / f"{task.key}_{cond}_s{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(out), indent=1, default=str))
    return {k: out[k] for k in ("task", "condition", "seed", "n_rounds", "success", "first_round_success")}


def code_version() -> dict:
    """The commit the harness runs from, and whether the package's tree differs from it."""
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=10).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=HERE.parents[1], capture_output=True, text=True, timeout=10).stdout
        return {"commit": head or None, "local_changes": bool(changes.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "local_changes": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--effort", default="medium")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--conditions", nargs="+", default=["dsl_v1", "dsl_v0", "direct"]); ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out", required=True)
    ap.add_argument("--work-root", default=None, help="where the empty per-call working directories go")
    ap.add_argument("--replay", help="a recorded run: its trials again with the present pipelines, keeping each recorded "
                                     "answer whose message is unchanged (model, effort, rounds and seeds come from its record)")
    a = ap.parse_args()
    replay_meta = None
    if a.replay:
        replay_meta = json.loads((Path(a.replay) / "run_meta.json").read_text())
        a.model, a.effort, a.rounds = replay_meta["model"], replay_meta["effort"], replay_meta["rounds"]
        a.seeds, a.conditions = replay_meta["seeds"], replay_meta["conditions"]
    import tasks as T, llm
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = [t.key for t in T.all_tasks() if not a.tasks or t.key in a.tasks or t.name in a.tasks]
    jobs = [{"task": k, "condition": c, "seed": s, "model": a.model, "effort": a.effort, "rounds": a.rounds, "out": str(out),
             "work_root": a.work_root} | ({"replay_of": str(Path(a.replay) / "trials" / f"{k}_{c}_s{s}.json")} if a.replay else {})
            for k in keys for s in a.seeds for c in a.conditions
            if not (out / "trials" / f"{k}_{c}_s{s}.json").exists()]
    meta = {"cli": llm.cli_version(), "model": a.model, "effort": a.effort, "rounds": a.rounds, "seeds": a.seeds, "conditions": a.conditions,
            "flags": llm.FLAGS, "env_allowlist": list(llm.ENV_KEEP) + ["CLAUDE_CODE_DISABLE_AUTO_MEMORY=1"],
            "system_prompts": {k: {"chars": len(v)} for k, v in conditions().items()}, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "code": code_version(), "assets_revision": T.ASSETS["revision"], "tasks": keys, "held_out": sorted(T.HELD_OUT),
            "workers": a.workers, "cpus": sorted(os.sched_getaffinity(0)), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    if replay_meta is not None:
        meta["replay_of"] = {"run": os.path.basename(os.path.normpath(a.replay)), "started": replay_meta["started"],
                             "code": replay_meta.get("code"), "cli": replay_meta.get("cli")}
    if (out / "run_meta.json").exists():                 # a rerun of the trials a crash left out: keep the first record
        first = json.loads((out / "run_meta.json").read_text())
        first.setdefault("resumed", []).append({k: meta[k] for k in ("started", "code", "cli", "cpus", "workers")} | {"trials": len(jobs)})
        meta = first
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    for k, v in conditions().items():
        (out / f"system_{k}.txt").write_text(v)
        (out / f"schema_{k}.json").write_text(json.dumps(schemas()[k], indent=1))
    print(f"{len(jobs)} trials", flush=True)
    crashed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_trial, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                print(json.dumps(f.result()), flush=True)
            except BrokenProcessPool:        # a worker died (killed, out of memory): stop, do not grade a partial run
                print(json.dumps({"task": j["task"], "condition": j["condition"], "seed": j["seed"], "crash": "a worker process died"}), flush=True)
                for g in futs:
                    g.cancel()
                sys.exit(4)
            except Exception:  # noqa: BLE001  a crashed trial is reported and left to a rerun
                crashed += 1
                print(json.dumps({"task": j["task"], "condition": j["condition"], "seed": j["seed"], "crash": traceback.format_exc()[-1500:]}), flush=True)
    if crashed:
        sys.exit(3)                         # the trials without a file run again when run.py is started again


if __name__ == "__main__":
    main()
