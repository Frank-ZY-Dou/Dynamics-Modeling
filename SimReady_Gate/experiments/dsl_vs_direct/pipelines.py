"""What happens to a model's answer in each condition, and how a final scene is graded.

dsl:     the answer is a SimReady program: validated against the scene, compiled, repaired by S4R on
         the starting scene (`simready.cli.repair_scene`, exactly what `repair` runs), verified.
direct:  the answer is the final pose of every free body: x, y of its reference centre and its yaw;
         the body is set down upright on the surface under it (or on the body it names), as the repair
         seats a body last, and verified.
The report each pipeline returns is what the next round of that condition is told.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from simready.cli import load_any, repair_scene, INPUT_ERRORS
from simready.dsl.schema import validate_program_json, ProgramError, V0_OPS, V0_FIELDS
from simready.dsl.model import parse_program
from simready.dsl.compile import support_height_for
from simready.gates.verify import verify_scene
from simready.repair.upright_s4r import rotz, _lift_clear

SOLVER = {"s_min": 0.05, "ds_max": 0.05, "tail": 40, "max_xy_step": 0.05}


def _elapsed(t0, c0, clock):
    """Wall seconds since t0; with a `clock` dict, also the process's CPU seconds since c0 (the
    worker's own computing, unaffected by other processes on its core; the model calls are child
    processes and not counted)."""
    if clock is not None:
        clock["cpu_s"] = time.process_time() - c0
    return time.time() - t0


def run_program(layout_path, obj: dict, v0_only: bool = False, clock: dict | None = None):
    """(final scene, report, self_ok, solver seconds)."""
    sc = load_any(str(layout_path))
    t0, c0 = time.time(), time.process_time()
    try:
        if not isinstance(obj, dict):
            raise ProgramError("the answer must be a JSON object")
        if v0_only:
            for i, st in enumerate(obj.get("statements", [])):
                op = st.get("op") if isinstance(st, dict) else None
                if op not in V0_OPS:
                    raise ProgramError(f"statement {i}: '{op}' is not one of the statements you may use")
                extra = set(st) - V0_FIELDS[op] - {"op", "reason"}
                if extra:
                    raise ProgramError(f"statement {i} ({op}): fields {sorted(extra)} are not available")
        dsl, notes = validate_program_json(obj, sc)
        prog = parse_program(dsl)
        out = repair_scene(sc, prog, s_min=SOLVER["s_min"], ds_max=SOLVER["ds_max"], tail=SOLVER["tail"], max_xy_step=SOLVER["max_xy_step"])
    except INPUT_ERRORS as e:
        return sc, {"ok": False, "error": f"{type(e).__name__}: {e}"}, False, _elapsed(t0, c0, clock)
    report = {"ok": bool(out["ok"]), "pen_after": out["after"].pen_pairs,
              "failed_predicates": [(n, None if v is None else round(float(v), 4)) for n, ok, v in out["hard"] if not ok],
              "failed_preferences": [n for n, ok, _ in out["prefs"] if not ok], "solver_notes": out["solver_notes"][:4]}
    return sc, report, bool(out["ok"]), _elapsed(t0, c0, clock)


def run_poses(layout_path, obj: dict, support: str = "table", clock: dict | None = None):
    """(final scene, report, self_ok, seconds). The pose check is what a geometric verifier can say
    about poses without knowing the request: interpenetrating pairs, bodies off the support or not
    resting on anything."""
    sc = load_any(str(layout_path))
    t0, c0 = time.time(), time.process_time()
    names = {b.name for b in sc.free()}
    try:
        poses = obj.get("poses") if isinstance(obj, dict) else None
        if not isinstance(poses, list):
            raise ValueError("the answer must be an object with a 'poses' list")
        given = {}
        for i, p in enumerate(poses):
            n = p.get("name") if isinstance(p, dict) else None
            if n not in names:
                raise ValueError(f"pose {i}: unknown body '{n}'")
            if n in given:
                raise ValueError(f"pose {i}: '{n}' given twice")
            vals = [p.get(k) for k in ("x", "y", "yaw")]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals):
                raise ValueError(f"pose {i} ({n}): x, y and yaw must be finite numbers")
            on = p.get("on", support)
            if on != support and on not in names:
                raise ValueError(f"pose {i} ({n}): 'on' must be {support} or a body")
            given[n] = (float(vals[0]), float(vals[1]), float(vals[2]), on)
    except (ValueError, AttributeError) as e:
        return sc, {"ok": False, "error": str(e)}, False, _elapsed(t0, c0, clock)
    placed = set()

    def seat(n, depth=0):
        if n in placed or depth > len(given):
            return
        x, y, yaw, on = given[n]
        if on in given:
            seat(on, depth + 1)
        b = sc[n]
        b.rotation = rotz(math.radians(yaw))
        # set down from above: the drop meets the top of what it rests on, wherever the body started (left at its
        # starting height, a body moved onto a taller one, a carton onto a carton, met the inside of it)
        b.center = np.array([x, y, float(sc[on].world_aabb()[1][2]) + b.support_offset(sc.up) + 0.01])
        h = support_height_for(sc, b, sc[on])
        b.center = np.array([x, y, h + b.support_offset(sc.up)])
        _lift_clear(sc, b, sc[on])      # as the repair's last seat: a residual of the sampled drop is raised clear
        placed.add(n)
    for n in given:
        seat(n)
    rep = verify_scene(sc)
    lo, hi = sc[support].world_aabb()
    off = [b.name for b in sc.free() if (b.world_aabb()[0][0] < lo[0] or b.world_aabb()[1][0] > hi[0]
                                          or b.world_aabb()[0][1] < lo[1] or b.world_aabb()[1][1] > hi[1])]
    missing = sorted(names - set(given))
    report = {"ok": rep.pen_pairs == 0 and not rep.floating and not off and not missing,
              "penetrating_pairs": [(a, b, round(1000 * s, 1)) for a, b, s in rep.pairs][:40],
              "not_resting_on_anything": rep.floating, "off_the_support": off, "missing_poses": missing}
    return sc, report, bool(report["ok"]), _elapsed(t0, c0, clock)


def typed_supports(obj) -> dict:
    """body -> the body a pose answer rests it on (the "on" of its pose)."""
    poses = obj.get("poses") if isinstance(obj, dict) else None
    return {p["name"]: p["on"] for p in poses if isinstance(p, dict) and isinstance(p.get("name"), str)
            and isinstance(p.get("on"), str)} if isinstance(poses, list) else {}


def generic_repair(sc, support: str = "table", on: dict | None = None, clock: dict | None = None):
    """The ablation: S4R with only the geometric statements every task shares (no penetration,
    upright on the support, over its top), on poses a model typed; a body the answer rests on
    another (`on`, typed_supports) stays on it."""
    names = {b.name for b in sc.free()}
    rest = "".join(f"  on_support({n}, {s})\n" for n, s in sorted((on or {}).items()) if n in names and s in names and s != n)
    prog = parse_program(f"program\n  no_penetration(*, margin=0.005)\n  on_support(*, {support})\n{rest}  upright(*)\n"
                         f"  within(*, {support}.top, inset=0.01)\n")
    sc2 = copy.deepcopy(sc)
    t0, c0 = time.time(), time.process_time()
    try:
        out = repair_scene(sc2, prog, s_min=SOLVER["s_min"], ds_max=SOLVER["ds_max"], tail=SOLVER["tail"], max_xy_step=SOLVER["max_xy_step"])
        ok = bool(out["ok"])
    except INPUT_ERRORS:
        ok = False
    return sc2, ok, _elapsed(t0, c0, clock)


def grade(task, final, start):
    """(passed, [(name, ok, value)]) of the task's rubric on a final scene."""
    checks = task.checks(final, start)
    return all(bool(c[1]) for c in checks), [(c[0], bool(c[1]), c[2] if not isinstance(c[2], (np.floating, float)) else round(float(c[2]), 4))
                                              for c in checks]
