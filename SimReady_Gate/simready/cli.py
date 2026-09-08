"""simready CLI: summarize | prompt | check | verify | repair | settle.

Exit codes: 0 = the requested outcome holds (repair: pen 0 and every predicate satisfied;
settle: G5 pass); 1 = it does not; 2 = the program is invalid (ProgramError / CompileError /
SyntaxError). Reports are JSON on stdout; solver output goes to stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import simready  # noqa: F401  (S4R path)
from .scene.model import Scene, Body, slab_proxy, MeshProxy
from .gates.verify import verify_scene, PEN_EPS, TOUCH_EPS
from .dsl.model import parse_program
from .dsl.compile import compile_program, check_predicates, CompileError
from .dsl.schema import prompt_for, validate_program_json, scene_summary, ProgramError, DEFAULTS
from .repair.upright_s4r import repair_upright, reseat_on_supports, polish_full_mesh

SCENE_EXT = (".usda", ".usd", ".usdc")


def load_any(path: str) -> Scene:
    if path.endswith(SCENE_EXT):
        from .io.usd_io import load_scene_usda
        return load_scene_usda(path)
    if path.endswith(".json"):
        from .io.usd_io import body_from_object_usd
        L = json.load(open(path))
        bodies = []
        tb = L.get("table")
        if tb and tb.get("usd_path") and os.path.exists(tb["usd_path"]):
            bodies.append(body_from_object_usd("table", tb["usd_path"], tb.get("position", (0, 0, 0)), fixed=True, tags={"support", "fixture"}))
        elif L.get("base_scene") and os.path.exists(L["base_scene"]):
            from .io.usd_io import load_scene_usda
            bodies.extend(load_scene_usda(L["base_scene"]).bodies)
        else:
            import trimesh
            slab = trimesh.creation.box(extents=(0.8, 1.0, 0.04))
            bodies.append(Body.from_mesh("table", slab.vertices, slab.faces, center=np.array([0.274, 0.0, -0.02]), fixed=True, tags={"support", "fixture"}))
        for o in L["objects"]:
            bodies.append(body_from_object_usd(o["name"], o["usd_path"], (o["x"], o["y"], o["z"]), yaw_deg=o.get("yaw", 0.0), tags={"object"}))
        return Scene(bodies, meta={"path": path})
    raise SystemExit(f"unsupported scene file {path}")


def load_program(path: str, scene: Scene):
    """(program, dsl_text, notes, raw_json_or_None)."""
    text = open(path).read()
    if path.endswith(".json"):
        obj = json.loads(text)
        dsl, notes = validate_program_json(obj, scene)
        return parse_program(dsl), dsl, notes, obj
    return parse_program(text), text, [], None


@contextlib.contextmanager
def quiet_stdout():
    """Route C-level stdout (OSQP prints from its C core) to stderr while a solver runs."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1); os.close(saved)


def provenance(scene_path, program_path=None, dsl=None, raw=None, params=None):
    def sha(path):
        try:
            return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        except OSError:
            return None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True).stdout.strip()[:12]
    except Exception:  # noqa: BLE001
        git = None
    versions = {}
    for mod in ("numpy", "trimesh", "fcl", "osqp", "mujoco", "pxr"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    return {"scene": {"path": scene_path, "sha256": sha(scene_path)}, "program": {"path": program_path, "sha256": sha(program_path) if program_path else None,
            "dsl": dsl, "json": raw}, "params": params or {}, "tolerances": {"pen_eps": PEN_EPS, "touch_eps": TOUCH_EPS, "predicate_tol": 2e-3,
            "verify_gap_tol": 2e-3, "verify_sink_tol": 1e-3, "defaults": DEFAULTS}, "software": {"git": git, "python": platform.python_version(), **versions},
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def cmd_summarize(a):
    sc = load_any(a.scene); print(scene_summary(sc))


def cmd_prompt(a):
    sc = load_any(a.scene); print(prompt_for(a.request, sc))


def cmd_check(a):
    """Validate, parse AND compile (on a copy) so that every error the repair would hit is
    reported here as ProgramError, with a dry run of the rows and predicates."""
    import copy
    sc = load_any(a.scene)
    try:
        prog, dsl, notes, _ = load_program(a.program, sc)
        spec = compile_program(prog, copy.deepcopy(sc))
        from .repair.upright_s4r import State, decompose_zyx
        sc2 = copy.deepcopy(sc); compile_program(prog, sc2)
        free = [k for k, b in enumerate(sc2.bodies) if not b.fixed]
        st = State(sc2, free, {k: 3 * i for i, k in enumerate(free)}, np.array([sc2.bodies[k].center[:2] for k in free]) if free else np.zeros((0, 2)),
                   np.zeros(len(free)), np.zeros(len(free)), np.zeros(len(free)), 1.0, 0.0)
        rows = spec.rows_fn(st) if spec.rows_fn else []
        preds = check_predicates(prog, sc2)
    except (ProgramError, CompileError, SyntaxError, KeyError, ValueError) as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})); sys.exit(2)
    print(dsl)
    for n in notes:
        print("note:", n)
    print(json.dumps({"ok": True, "statements": len(prog.statements), "rows": len(rows), "predicates": len(preds),
                      "supports": spec.supports, "upright": sorted(spec.upright), "margin": spec.d_hat}))


def cmd_verify(a):
    sc = load_any(a.scene); rep = verify_scene(sc); print(rep.summary())
    for x, y, s in sorted(rep.pairs, key=lambda t: t[2]):
        print(f"  {x:20s} {y:20s} {1000*s:+.2f} mm")
    if sc.meta.get("dropped"):
        print("dropped children (unresolved assets):", sc.meta["dropped"])
    sys.exit(0 if rep.pen_pairs == 0 else 1)


def cmd_repair(a):
    sc = load_any(a.scene)
    try:
        prog, dsl, notes, raw = load_program(a.program, sc)
        before = verify_scene(sc)
        spec = compile_program(prog, sc, s_min=a.s_min, ds_max=a.ds_max, tail_iters=a.tail)
    except (ProgramError, CompileError, SyntaxError, KeyError, ValueError) as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})); sys.exit(2)
    t0 = time.time()
    tops = {sup: min(h for b, h in spec.supports.items() if spec.support_of.get(b) == sup) for sup in set(spec.support_of.values())}
    with quiet_stdout():
        with MeshProxy(sc, faces=a.proxy_faces, slab_tops=tops):   # repair on decimated meshes + support slabs; verify on full meshes
            res = repair_upright(sc, spec, verbose=a.verbose)
        reseat_on_supports(sc, spec)
        after = verify_scene(sc)
        if after.pen_pairs > 0:                       # proxy residuals: polish on the full meshes
            res2 = polish_full_mesh(sc, spec, iters=a.tail, verbose=a.verbose)
            reseat_on_supports(sc, spec); res.steps += res2.steps
            after = verify_scene(sc)
    preds = check_predicates(prog, sc)
    ok = after.pen_pairs == 0 and all(p[1] for p in preds)
    params = {"s_min": a.s_min, "ds_max": a.ds_max, "tail_iters": a.tail, "proxy_faces": a.proxy_faces}
    report = {"ok": ok, "scene": a.scene, "program": a.program, "pen_before": before.pen_pairs, "pen_after": after.pen_pairs,
              "min_score_after": after.min_signed, "contained_after": after.contained, "floating_after": after.floating,
              "rmsd_xy": res.rmsd, "steps": res.steps, "time_s": round(time.time() - t0, 2),
              "predicates": [(n, bool(okp), float(v)) for n, okp, v in preds], "failed_predicates": [n for n, okp, _ in preds if not okp],
              "notes": notes, "dropped_children": sc.meta.get("dropped", []),
              "provenance": provenance(a.scene, a.program, dsl, raw, params)}
    if a.out:
        out_is_usd = a.out.endswith(SCENE_EXT)
        if out_is_usd != a.scene.endswith(SCENE_EXT):
            print(json.dumps({"ok": False, "error": f"--out {a.out} must have the same kind of extension as the scene {a.scene}"})); sys.exit(2)
        if out_is_usd:
            from .io.usd_io import write_scene_poses
            write_scene_poses(sc, a.scene, a.out)          # asset paths are re-anchored when --out is elsewhere
            back = load_any(a.out)                               # round trip must reproduce the repaired poses
            worst = max((float(np.abs(back[b.name].center - b.center).max()) for b in sc.free()), default=0.0)
            report["written_pose_error_m"] = worst
            if worst > 1e-4:          # float32 orient ops reproduce poses to ~1e-6 m; 0.1 mm is far inside every tolerance
                report["ok"] = ok = False; report["error"] = f"written scene does not reproduce the repaired poses ({worst:.2e} m)"
        else:
            import math
            L = json.load(open(a.scene))
            for o in L["objects"]:
                b = sc[o["name"]]
                t = b.center - b.rotation @ b.meta.get("c_model", np.zeros(3))
                o["x"], o["y"], o["z"] = [float(v) for v in t]
                o["yaw"] = math.degrees(math.atan2(b.rotation[1, 0], b.rotation[0, 0]))
            L.pop("ok", None); L.pop("msg", None)
            json.dump(L, open(a.out, "w"), indent=1)
        cert = Path(a.out).with_suffix(".certificate.json")
        report["written"] = a.out
        json.dump(report, open(cert, "w"), indent=1)
        report["certificate"] = str(cert)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
    sys.exit(0 if ok else 1)


def cmd_settle(a):
    """G5: settle the scene in MuJoCo with engine-like proxies and measure; exit 1 when it is not
    simulation-ready. Thresholds come from the program's gate section when given, else the CLI."""
    from .gates.settle_mujoco import settle_and_measure
    sc = load_any(a.scene)
    v_max, d_max, min_gap = a.v_max, a.d_max, None
    if a.program:
        try:
            prog, dsl, notes, _ = load_program(a.program, sc)
            spec = compile_program(prog, sc)
        except (ProgramError, CompileError, SyntaxError, KeyError, ValueError) as e:
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})); sys.exit(2)
        tops = {sup: min(h for b, h in spec.supports.items() if spec.support_of.get(b) == sup) for sup in set(spec.support_of.values())}
        g5 = prog.gate.get("G5", {})
        if "v_max" in g5:
            v_max = g5["v_max"][1]
        if "dx" in g5:
            d_max = g5["dx"][1]
        g2 = prog.gate.get("G2", {})
        if "min_gap" in g2:
            min_gap = g2["min_gap"][1]
    else:   # no program: every fixed body tagged support is a slab at the plate height under the free bodies
        tops = {}
        for sup in sc.bodies:
            if sup.fixed and "support" in sup.tags:
                hs = [sc.support_height(sup, at=b.center[:2], radius=0.1) for b in sc.free()]
                tops[sup.name] = min(hs) if hs else sc.support_height(sup)
    # the largest hover above a support bounds the speed a plain drop can reach
    hover = 0.0
    for b in sc.free():
        for sup, top in tops.items():
            hover = max(hover, float(b.world_aabb()[0][2] - top))
    free_fall_bound = float(np.sqrt(2 * 9.81 * max(hover, 0.0)))
    modes = ("hull", "coacd") if a.proxy == "both" else (a.proxy,)
    report, ok = {"scene": a.scene, "seconds": a.seconds, "supports": tops, "max_hover_m": hover, "free_fall_bound_m_s": free_fall_bound}, True
    for mode in modes:
        with quiet_stdout():
            r = settle_and_measure(sc, seconds=a.seconds, support_tops=tops, decompose_free=(mode == "coacd"), max_hull_verts=a.hull_verts)
        passed = (not r.left_support) and r.peak_speed <= v_max and r.peak_disp <= d_max
        ok &= passed
        report[mode] = {"peak_speed": round(r.peak_speed, 3), "peak_disp": round(r.peak_disp, 4), "left_support": r.left_support,
                        "per_body_peak_speed": {k: round(v, 3) for k, v in r.per_body_peak_speed.items()},
                        "final_disp": {k: round(v, 4) for k, v in r.final_disp.items()},
                        "faster_than_free_fall": r.peak_speed > free_fall_bound + 0.05, "pass": passed}
    if min_gap is not None:
        rep = verify_scene(sc)
        gaps = [s for x, y, s in rep.pairs]   # negative pairs only; positive gaps are not stored per pair
        report["g2_min_gap_required"] = min_gap; report["g2_pen_pairs"] = rep.pen_pairs
        ok &= rep.pen_pairs == 0
    report["pass"] = ok; report["thresholds"] = {"v_max": v_max, "d_max": d_max, "source": "program gate" if a.program and prog.gate else "cli"}
    print(json.dumps(report, indent=1))
    cert = Path(a.scene).with_suffix(".certificate.json")
    if cert.exists():
        c = json.load(open(cert)); c["g5"] = report; json.dump(c, open(cert, "w"), indent=1); print("updated", cert, file=sys.stderr)
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simready")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize"); s.add_argument("scene"); s.set_defaults(fn=cmd_summarize)
    s = sub.add_parser("prompt"); s.add_argument("scene"); s.add_argument("--request", required=True); s.set_defaults(fn=cmd_prompt)
    s = sub.add_parser("check"); s.add_argument("scene"); s.add_argument("program"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("verify"); s.add_argument("scene"); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("repair"); s.add_argument("scene"); s.add_argument("program"); s.add_argument("--out")
    s.add_argument("--s-min", type=float, default=0.05); s.add_argument("--ds-max", type=float, default=0.05)
    s.add_argument("--tail", type=int, default=30); s.add_argument("--verbose", action="store_true")
    s.add_argument("--proxy-faces", type=int, default=2000); s.set_defaults(fn=cmd_repair)
    s = sub.add_parser("settle"); s.add_argument("scene"); s.add_argument("--program"); s.add_argument("--seconds", type=float, default=2.0)
    s.add_argument("--proxy", choices=("hull", "coacd", "both"), default="both"); s.add_argument("--hull-verts", type=int, default=256)
    s.add_argument("--v-max", type=float, default=1.0, help="peak body speed allowed (m/s) unless the program's gate says otherwise")
    s.add_argument("--d-max", type=float, default=0.15, help="peak displacement allowed (m)"); s.set_defaults(fn=cmd_settle)
    a = ap.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()
