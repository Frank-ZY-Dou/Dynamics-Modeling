"""Regression tests for the input validation, geometry queries, gates and solver acceptance
rules. Run with `python -m unittest discover tests` (pytest also collects them). The tests
build their own box scenes; MuJoCo, FCL and OSQP are used where the behaviour under test
lives in them.
"""
import inspect
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simready.scene.model import Scene, Body  # noqa: E402
from simready.errors import GeometryQueryError  # noqa: E402
from simready.dsl.schema import validate_program_json, ProgramError  # noqa: E402
from simready.dsl.model import parse_program  # noqa: E402
from simready.dsl.compile import compile_program, check_predicates  # noqa: E402
from simready.dsl.text2dsl import extract_json, strict_json_loads  # noqa: E402
from simready.repair.upright_s4r import RepairSpec, State, repair_upright, rotz, _qp_solved  # noqa: E402
from simready.gates.verify import pair_signed_distances, _contained  # noqa: E402
from simready.gates.clearance import check_clearance  # noqa: E402


def box(name, extents, center, fixed=False, tags=(), yaw=0.0):
    m = trimesh.creation.box(extents=extents)
    return Body.from_mesh(name, m.vertices, m.faces, center=np.array(center, dtype=float), rotation=rotz(yaw),
                          fixed=fixed, tags=set(tags))


def table_scene(*objects):
    table = box("table", (1.0, 1.0, 0.05), (0.0, 0.0, -0.025), fixed=True, tags=("support", "fixture"))
    return Scene([table, *objects])


class InputValidation(unittest.TestCase):
    def test_backend_accepts_scene(self):
        try:
            from simready.dsl import anthropic_backend
        except ImportError:
            self.skipTest("anthropic SDK not installed")
        self.assertIn("scene", inspect.signature(anthropic_backend.complete).parameters)

    def test_string_coordinate_cannot_add_statements(self):
        sc = table_scene(box("mug", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)))
        bad = {"statements": [{"op": "place", "a": "mug", "x": "0, y=0)\n  fixed(mug)\n  place(mug, x=0", "y": 0}]}
        with self.assertRaises(ProgramError):
            validate_program_json(bad, sc)
        good = {"statements": [{"op": "place", "a": "mug", "x": 0.1, "y": 0.2}]}
        dsl, _ = validate_program_json(good, sc)
        self.assertNotIn("fixed", dsl)
        for value in (True, float("nan"), float("inf"), None):
            with self.assertRaises(ProgramError):
                validate_program_json({"statements": [{"op": "place", "a": "mug", "x": value, "y": 0}]}, sc)

    def test_gate_values_must_be_finite(self):
        sc = table_scene(box("mug", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)))
        with self.assertRaises(ProgramError):
            validate_program_json({"statements": [], "gate": {"settle_v_max": float("inf")}}, sc)
        with self.assertRaises(ProgramError):
            strict_json_loads('{"statements": [], "gate": {"settle_v_max": 1e999}}')
        with self.assertRaises(ProgramError):
            strict_json_loads('{"statements": [], "gate": {"settle_v_max": NaN}}')

    def test_fields_belong_to_their_op(self):
        sc = table_scene(box("mug", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)))
        with self.assertRaises(ProgramError):
            validate_program_json({"statements": [{"op": "fixed", "a": "mug", "gap": 0.1}]}, sc)
        with self.assertRaises(ProgramError):
            validate_program_json({"statements": [], "extra": 1}, sc)

    def test_inside_inset_is_kept(self):
        tray = box("tray", (0.4, 0.4, 0.05), (0.0, 0.0, 0.025), tags=("container",))
        sc = table_scene(tray, box("mug", (0.1, 0.1, 0.1), (0.0, 0.0, 0.1)))
        dsl, _ = validate_program_json({"statements": [{"op": "inside", "a": "mug", "b": "tray", "inset": 0.25}]}, sc)
        self.assertIn("inside(mug, tray, inset=0.25)", dsl)

    def test_text_gate_errors_are_explicit(self):
        with self.assertRaises(SyntaxError):
            parse_program("program\n  no_penetration(*)\ngate\n  G5: v_max <= nan")
        with self.assertRaises(SyntaxError):
            parse_program("program\n  no_penetration(*)\ngate\n  G5: v_max <= inf")
        with self.assertRaises(SyntaxError):
            parse_program("program\n  no_penetration(*)\ngate\n  G7: foo >= 1")
        with self.assertRaises(SyntaxError):
            parse_program("program\n  no_penetration(*)\ngate\n  G5: v_max >= 1")
        prog = parse_program("program\n  no_penetration(*)\ngate\n  G2: min_gap >= 1e-3\n  G5: v_max <= 0.5, dx <= 0.1")
        self.assertEqual(prog.gate["G5"]["v_max"], ("<=", 0.5))
        self.assertEqual(prog.gate["G2"]["min_gap"], (">=", 1e-3))

    def test_text_arity_and_numbers(self):
        for text in ("program\n  upright()", "program\n  on_support(mug)", "program\n  place(mug, x=1, x=2, y=0)",
                     "program\n  place(mug, x=abc, y=0)", "program\n  left_of(a, b, gap=-1)", "program\n  minimize displacement(a)"):
            with self.assertRaises(SyntaxError, msg=text):
                parse_program(text)
        prog = parse_program("program\n  place(mug, x=-0.3, y=0.1, yaw=-45)")
        self.assertEqual(prog.statements[0].kw["yaw"], -45.0)

    def test_extract_json_handles_escapes_and_prose(self):
        obj = extract_json('Here it is:\n```json\n{"intent": "say \\"}\\" please", "statements": []}\n```\nDone.')
        self.assertEqual(obj["intent"], 'say "}" please')
        obj = extract_json('{"intent": "a { b", "statements": [{"op": "minimize"}]}')
        self.assertEqual(len(obj["statements"]), 1)
        with self.assertRaises(ProgramError):
            extract_json('{"statements": [], "statements": []}')
        with self.assertRaises(ProgramError):
            extract_json('{"x": Infinity}')
        with self.assertRaises(ProgramError):
            extract_json("no object here")


class GeometryQueries(unittest.TestCase):
    def test_capsule_has_its_caps(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("mujoco not installed")
        from simready.io.mjcf_io import _geom_mesh
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><geom type="capsule" size="0.05 0.1"/><geom type="cylinder" size="0.05 0.1"/></body></worldbody></mujoco>')
        v, *_ = _geom_mesh(model, 0)
        self.assertAlmostEqual(float(v[:, 2].max() - v[:, 2].min()), 0.30, places=3)
        self.assertAlmostEqual(float(v[:, 2].max() + v[:, 2].min()), 0.0, places=6)
        v, *_ = _geom_mesh(model, 1)
        self.assertAlmostEqual(float(v[:, 2].max() - v[:, 2].min()), 0.20, places=3)

    def test_on_support_needs_the_support_under_the_body(self):
        far = box("far", (0.1, 0.1, 0.1), (10.0, 0.0, 0.05))         # level with the table top, 10 m away
        near = box("near", (0.1, 0.1, 0.1), (0.2, 0.1, 0.05))
        sc = table_scene(far, near)
        prog = parse_program("program\n  on_support(far, table)\n  on_support(near, table)")
        preds = dict((n, (ok, v)) for n, ok, v in check_predicates(prog, sc))
        self.assertEqual(preds["on_support(far,table)"], (False, None))
        self.assertTrue(preds["on_support(near,table)"][0])

    def test_ray_cache_follows_the_support_pose(self):
        sup = box("shelf", (2.0, 1.0, 0.05), (0.0, 0.0, 0.0), fixed=True, tags=("support",))
        sc = Scene([sup])
        self.assertEqual(sc._surface_hits(sup, np.array([[0.9, 0.0]])).size, 1)
        sup.rotation = rotz(math.pi / 2)                              # now 1 m along x, 2 m along y
        self.assertEqual(sc._surface_hits(sup, np.array([[0.9, 0.0]])).size, 0)
        self.assertEqual(sc._surface_hits(sup, np.array([[0.0, 0.9]])).size, 1)
        m = trimesh.creation.box(extents=(0.2, 0.2, 0.05))           # a different mesh with the same vertex count
        sup.rotation = np.eye(3); sup.verts = np.asarray(m.vertices, dtype=np.float64) - m.vertices.mean(0)
        self.assertEqual(sc._surface_hits(sup, np.array([[0.9, 0.0]])).size, 0)

    def test_failed_queries_are_errors_not_absence(self):
        class Broken:
            def contains(self, pts):
                raise RuntimeError("backend missing")
        with self.assertRaises(GeometryQueryError):
            _contained(np.zeros((4, 3)), Broken())
        sup = box("shelf", (1.0, 1.0, 0.05), (0.0, 0.0, 0.0), fixed=True)
        sc = Scene([sup])
        sc._surface_hits(sup, np.array([[0.0, 0.0]]))
        (key, entry), = sup.meta["_ray_cache"].items()

        class BrokenRay:
            def intersects_location(self, *a, **k):
                raise RuntimeError("no ray backend")
        sup.meta["_ray_cache"][key] = (entry[0], BrokenRay(), entry[2], entry[3])
        with self.assertRaises(GeometryQueryError):
            sc._surface_hits(sup, np.array([[0.0, 0.0]]))

    def test_same_centre_min_distance_has_a_direction(self):
        a = box("a", (0.1, 0.1, 0.1), (0.2, 0.0, 0.05)); b = box("b", (0.1, 0.1, 0.1), (0.2, 0.0, 0.05))
        sc = table_scene(a, b)
        prog = parse_program("program\n  min_distance(a, b, r=0.3)")
        spec = compile_program(prog, sc)
        free = [k for k, bd in enumerate(sc.bodies) if not bd.fixed]
        st = State(sc, free, {k: 3 * i for i, k in enumerate(free)}, np.array([sc.bodies[k].center[:2] for k in free]),
                   np.zeros(2), np.zeros(2), np.zeros(2), 1.0, 0.0)
        rows = spec.rows_fn(st)
        self.assertEqual(len(rows), 1)
        coeffs, lo, hi = rows[0]
        self.assertGreater(max(abs(v) for v in coeffs.values()), 0.5)
        self.assertAlmostEqual(lo, 0.3)


class Gates(unittest.TestCase):
    def test_clearance_is_compared_pair_by_pair(self):
        a = box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)); b = box("b", (0.1, 0.1, 0.1), (0.12, 0.0, 0.05))   # 2 cm apart
        sc = table_scene(a, b)
        support_of = {"a": "table", "b": "table"}
        loose = check_clearance(sc, 0.01, support_of)
        self.assertTrue(loose["pass"]); self.assertEqual(loose["pairs_exempt_support"], 2); self.assertEqual(loose["pairs_checked"], 1)
        tight = check_clearance(sc, 0.05, support_of)
        self.assertFalse(tight["pass"]); self.assertEqual(len(tight["violations"]), 1)
        self.assertAlmostEqual(tight["violations"][0]["gap_m"], 0.02, places=3)
        no_exempt = check_clearance(sc, 0.01, {})
        self.assertFalse(no_exempt["pass"])                        # the resting contacts sit at zero gap
        with self.assertRaises(ValueError):
            check_clearance(sc, float("nan"), support_of)

    def test_settle_refuses_a_zero_length_run(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from simready.gates.settle_mujoco import settle_and_measure
        sc = table_scene(box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)))
        for seconds in (0.0, -1.0, 1e-4, float("nan")):
            with self.assertRaises(ValueError, msg=seconds):
                settle_and_measure(sc, seconds=seconds)
        rep = settle_and_measure(sc, seconds=0.01, timestep=2e-3, support_tops={"table": 0.0})
        self.assertEqual(rep.steps, 5); self.assertAlmostEqual(rep.sim_seconds, 0.01)
        self.assertEqual(rep.engine_warnings, {})                         # a quiet run reports no engine warning

    def test_certificate_binding(self):
        from simready.cli import certificate_bound, sha16
        with tempfile.TemporaryDirectory() as d:
            scene = os.path.join(d, "s.json"); prog = os.path.join(d, "p.json"); out = os.path.join(d, "o.json")
            for p, txt in ((scene, "{}"), (prog, '{"statements": []}'), (out, '{"objects": []}')):
                open(p, "w").write(txt)
            asset = os.path.join(d, "mug.usda"); open(asset, "w").write("#usda 1.0\n")
            assets = {asset: sha16(asset)}
            cert = {"provenance": {"scene": {"sha256": sha16(scene)}, "program": {"sha256": sha16(prog)}, "assets": assets}, "written_sha256": sha16(out)}
            self.assertTrue(certificate_bound(cert, out, prog, assets))
            self.assertFalse(certificate_bound(cert, out, prog, None))                 # the asset check cannot be skipped
            self.assertFalse(certificate_bound(cert, out, prog, {asset: None}))        # nor passed with an unreadable asset
            self.assertFalse(certificate_bound(cert, scene, prog, assets))   # the scene the repair read is not the repaired scene
            self.assertFalse(certificate_bound(cert, out, None, assets))     # a settle without the program judges by other thresholds
            open(asset, "w").write("#usda 1.0\n# changed\n")
            self.assertFalse(certificate_bound(cert, out, prog, {asset: sha16(asset)}))   # an asset changed under the scene
            open(asset, "w").write("#usda 1.0\n")
            with open(prog, "w") as f:
                f.write('{"statements": [{"op": "minimize"}]}')
            self.assertFalse(certificate_bound(cert, out, prog, assets))    # the program changed
            self.assertFalse(certificate_bound({"provenance": {}}, scene, None))
            self.assertFalse(certificate_bound({"provenance": {"program": {"sha256": None}}, "written_sha256": sha16(out)}, out, None))   # no assets record
            bare = {"provenance": {"program": {"sha256": None}, "assets": {}}, "written_sha256": sha16(out)}
            self.assertTrue(certificate_bound(bare, out, None, {}))

    def test_reports_are_valid_json(self):
        from simready.cli import dump_json
        text = dump_json({"a": float("inf"), "b": np.float64(1.5), "c": np.array([1, 2]), "d": (1, float("nan"))})
        self.assertEqual(json.loads(text), {"a": None, "b": 1.5, "c": [1, 2], "d": [1, None]})


class Solver(unittest.TestCase):
    def test_qp_status_codes(self):
        class Info:
            def __init__(self, status, status_val=None):
                self.status = status
                if status_val is not None:
                    self.status_val = status_val
        self.assertTrue(_qp_solved(Info("solved", 1)))
        self.assertTrue(_qp_solved(Info("solved inaccurate", 2)))
        self.assertTrue(_qp_solved(Info("solved inaccurate")))
        self.assertFalse(_qp_solved(Info("primal infeasible", 3)))
        self.assertFalse(_qp_solved(Info("maximum iterations reached", -2)))
        from simready.repair.upright_s4r import _qp_infeasible
        self.assertTrue(_qp_infeasible(Info("primal infeasible", 3)))
        self.assertTrue(_qp_infeasible(Info("primal infeasible inaccurate", 4)))
        self.assertTrue(_qp_infeasible(Info("", 4)))
        self.assertFalse(_qp_infeasible(Info("solved inaccurate", 2)))
        self.assertFalse(_qp_infeasible(Info("dual infeasible", 5)))

    def test_spec_validation(self):
        sc = table_scene(box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)))
        for bad in (dict(ds_max=0.0), dict(s_min=0.0), dict(s_min=1.5), dict(tail_iters=-1), dict(max_xy_step=0.0), dict(d_hat=float("nan"))):
            with self.assertRaises(ValueError, msg=bad):
                repair_upright(sc, RepairSpec(supports={"a": 0.0}, **bad))

    def test_overlapping_boxes_are_separated(self):
        a = box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)); b = box("b", (0.1, 0.1, 0.1), (0.07, 0.0, 0.05))   # 3 cm overlap
        sc = table_scene(a, b)
        prog = parse_program("program\n  no_penetration(*, margin=0.005)\n  on_support(*, table)\n  upright(*)\n  within(*, table.top, inset=0.05)")
        spec = compile_program(prog, sc, ds_max=0.1, tail_iters=20)
        res = repair_upright(sc, spec)
        self.assertEqual(res.pen_before, 1)
        self.assertEqual(res.pen_after, 0)
        self.assertTrue(bool(np.all(np.isfinite(res.displacement))))
        self.assertTrue(all(ok for _, ok, _ in check_predicates(prog, sc)))
        gap = min(s for i, j, s, *_ in pair_signed_distances(sc.bodies, prefilter=1.0) if not (sc.bodies[i].fixed or sc.bodies[j].fixed))
        self.assertGreaterEqual(gap, 0.0)
        self.assertLessEqual(max(abs(t[4]) for t in res.trace), 1e-6)      # no DSL row had to be relaxed

    def test_program_yields_to_contacts(self):
        # a placement target inside a fixed obstacle: the contacts win, the predicate reports it
        a = box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)); b = box("b", (0.1, 0.1, 0.1), (0.3, 0.0, 0.05))
        sc = table_scene(a, b)
        prog = parse_program("program\n  no_penetration(*, margin=0.005)\n  fixed(b)\n  on_support(*, table)\n  place(a, x=0.3, y=0.0, w=1.0)")
        spec = compile_program(prog, sc, ds_max=0.1, tail_iters=20)
        res = repair_upright(sc, spec)
        self.assertEqual(res.pen_after, 0)
        preds = {n: ok for n, ok, _ in check_predicates(prog, sc)}
        self.assertFalse(preds["place(a)"])
        self.assertTrue(preds["on_support(a,table)"])
        self.assertGreaterEqual(min(s for i, j, s, *_ in pair_signed_distances(sc.bodies, prefilter=1.0)
                                    if not (sc.bodies[i].fixed and sc.bodies[j].fixed)), -1e-6)

    def test_program_row_yields_when_a_contact_blocks_it(self):
        # a wall across the whole table: the relation cannot be met, the contacts hold, the predicate says so
        wall = box("wall", (0.02, 1.0, 0.2), (0.2, 0.0, 0.1), fixed=True)
        a = box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05))
        sc = table_scene(wall, a)
        prog = parse_program("program\n  no_penetration(*, margin=0.005)\n  on_support(a, table)\n  within(a, table.top, inset=0.0)\n  in_front_of(a, wall, gap=0.1)")
        spec = compile_program(prog, sc, ds_max=0.1, tail_iters=10)
        res = repair_upright(sc, spec)
        self.assertEqual(res.pen_after, 0)
        self.assertGreaterEqual(res.relaxed_steps, 1)
        self.assertTrue(any(t[4] > 1e-3 for t in res.trace))
        preds = {n: ok for n, ok, _ in check_predicates(prog, sc)}
        self.assertFalse(preds["in_front_of(a,wall,gap=0.1)"])
        self.assertTrue(preds["within(a,table.top)"] and preds["on_support(a,table)"])
        self.assertLess(sc["a"].center[0], 0.2)                           # it stayed behind the wall

    def test_applied_step_satisfies_the_contact_rows(self):
        # the trust region is inside the QP: whatever is applied still satisfies every contact row
        from simready.repair.upright_s4r import _solve_qp
        rows = [([(0, 1.0)], 0.004), ([(0, -1.0), (3, 1.0)], 0.006)]      # a: dx >= 4 mm; b - a >= 6 mm
        status, step, c_sl, d_sl = _solve_qp(2, rows, [], False, True, 0.03, 0.08, [1.0, 1.0, 0.01] * 2, np.zeros(6), False, 1.0)
        self.assertEqual(status, 1)
        x = step.reshape(-1)
        self.assertGreaterEqual(x[0], 0.004 - 1e-6)
        self.assertGreaterEqual(x[3] - x[0], 0.006 - 1e-6)
        self.assertLessEqual(float(np.abs(step[:, :2]).max()), 0.03 + 1e-6)
        status, *_ = _solve_qp(2, [([(0, 1.0)], 0.05)], [], False, True, 0.03, 0.08, [1.0, 1.0, 0.01] * 2, np.zeros(6), False, 1.0)
        self.assertEqual(status, -2)                                     # more than the region allows: reported, not clipped

    def test_failed_solve_does_not_advance_the_scale(self):
        from simready.repair import upright_s4r as U
        a = box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)); b = box("b", (0.1, 0.1, 0.1), (0.02, 0.0, 0.05))   # overlap even at s = 0.9
        sc = table_scene(a, b)
        prog = parse_program("program\n  no_penetration(*, margin=0.005)\n  on_support(*, table)")
        spec = compile_program(prog, sc, s_min=0.9, ds_max=0.05, tail_iters=0)
        original = U._solve_qp
        U._solve_qp = lambda *args, **kw: (-1, None, 0.0, 0.0)
        try:
            res = repair_upright(sc, spec)
        finally:
            U._solve_qp = original
        self.assertEqual(res.trace, [])                                   # no step was accepted
        self.assertTrue(any("stopped at scale 0.900" in n for n in res.notes))
        self.assertEqual(res.steps, U.MAX_STEP_FAILURES + 1)
        self.assertFalse(res.continuation_complete)
        self.assertEqual(res.termination, "qp_failures")
        self.assertAlmostEqual(res.last_accepted_scale, 0.9)
        self.assertTrue(np.allclose(sc["a"].center[:2], [0.0, 0.0]) and np.allclose(sc["b"].center[:2], [0.02, 0.0]))

    def test_steps_respect_the_trust_region(self):
        a = box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)); b = box("b", (0.1, 0.1, 0.1), (0.02, 0.0, 0.05))   # 8 cm overlap
        sc = table_scene(a, b)
        prog = parse_program("program\n  no_penetration(*, margin=0.005)\n  on_support(*, table)")
        spec = compile_program(prog, sc, s_min=1.0, tail_iters=1, max_xy_step=0.01)
        before = np.array([bd.center[:2].copy() for bd in sc.free()])
        repair_upright(sc, spec)
        after = np.array([bd.center[:2] for bd in sc.free()])
        self.assertLessEqual(float(np.abs(after - before).max()), 0.6 * 0.01 + 1e-6)   # tail cap: 0.6 x max_xy_step per axis, solver tolerance


def write_usda(path, meters_per_unit=None, up_axis=None, ghost=True):
    """A minimal Z-up scene: a fixed table slab, a free cube on it and, optionally, a prim whose
    payload does not resolve."""
    def mesh(extents, indent="        "):
        m = trimesh.creation.box(extents=extents)
        pts = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in m.vertices)
        idx = ", ".join(str(int(i)) for i in m.faces.reshape(-1))
        return (f'{indent}def Mesh "geo" {{\n{indent}    point3f[] points = [{pts}]\n'
                f'{indent}    int[] faceVertexCounts = [{", ".join(["3"] * len(m.faces))}]\n'
                f'{indent}    int[] faceVertexIndices = [{idx}]\n{indent}}}\n')
    meta = ['    defaultPrim = "World"']
    if meters_per_unit is not None:
        meta.append(f"    metersPerUnit = {meters_per_unit}")
    if up_axis is not None:
        meta.append(f'    upAxis = "{up_axis}"')
    text = "#usda 1.0\n(\n" + "\n".join(meta) + "\n)\n\ndef Xform \"World\" {\n"
    for name, ext, z in (("table", (1.0, 1.0, 0.05), -0.025), ("mug", (0.1, 0.1, 0.1), 0.05)):
        text += (f'    def Xform "{name}" {{\n        double3 xformOp:translate = (0, 0, {z})\n'
                 f'        uniform token[] xformOpOrder = ["xformOp:translate"]\n' + mesh(ext) + "    }\n")
    if ghost:
        text += '    def Xform "ghost" (\n        payload = @missing_asset.usda@\n    ) {\n    }\n'
    text += "}\n"
    open(path, "w").write(text)


class Scenes(unittest.TestCase):
    def setUp(self):
        try:
            import pxr  # noqa: F401
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("usd-core or mujoco not installed")

    def test_usd_stage_metadata(self):
        from simready.io.usd_io import load_scene_usda
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.usda")
            write_usda(p)                                                 # nothing authored: read as metres, Z up
            sc = load_scene_usda(p)
            self.assertEqual(sorted(b.name for b in sc.bodies), ["mug", "table"])
            self.assertEqual(sc.meta["dropped"], ["ghost"])
            self.assertTrue(sc["table"].fixed and not sc["mug"].fixed)
            write_usda(p, meters_per_unit=1, up_axis="Z")
            self.assertEqual(len(load_scene_usda(p).bodies), 2)
            write_usda(p, meters_per_unit=0.01)
            with self.assertRaises(ValueError):
                load_scene_usda(p)
            write_usda(p, up_axis="Y")
            with self.assertRaises(ValueError):
                load_scene_usda(p)

    def test_unresolved_asset_blocks_ok(self):
        import contextlib, io
        from simready.cli import main
        with tempfile.TemporaryDirectory() as d:
            scene = os.path.join(d, "s.usda"); prog = os.path.join(d, "p.json"); out = os.path.join(d, "s_repaired.usda")
            write_usda(scene)
            json.dump({"statements": [{"op": "no_penetration"}, {"op": "on_support", "a": "mug", "b": "table"}, {"op": "upright", "a": "mug"}]}, open(prog, "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
                main(["repair", scene, prog, "--out", out, "--tail", "2"])
            self.assertEqual(cm.exception.code, 1)
            report = json.loads(buf.getvalue())
            self.assertFalse(report["ok"])
            self.assertEqual(report["pen_after"], 0)
            self.assertEqual(report["dropped_children"], ["ghost"])
            self.assertTrue(any("unresolved" in n for n in report["notes"]))
            cert = json.load(open(os.path.join(d, "s_repaired.certificate.json")))
            self.assertFalse(cert["ok"]); self.assertIsNotNone(cert["written_sha256"])
            from simready.cli import certificate_bound, asset_hashes, load_any
            self.assertTrue(certificate_bound(cert, out, prog, asset_hashes(load_any(out))))   # the written scene binds, inline bodies included
            self.assertFalse(certificate_bound(cert, scene, prog, asset_hashes(load_any(scene))))

    def test_settle_cleanup_fallback_and_one_report(self):
        import contextlib, glob, io, types
        from unittest import mock
        from simready.io.usd_io import load_scene_usda
        from simready.gates import settle_mujoco as S
        from simready.cli import main
        with tempfile.TemporaryDirectory() as d:
            scene = os.path.join(d, "s.usda"); write_usda(scene, ghost=False)
            sc = load_scene_usda(scene)
            fake = types.ModuleType("coacd")
            fake.Mesh = lambda v, f: None
            def failing(*a, **k):
                raise RuntimeError("decomposition failed")
            fake.run_coacd = failing
            before = set(glob.glob(os.path.join(tempfile.gettempdir(), "simready_mj_*")))
            S._COACD_CACHE.clear()
            with mock.patch.dict(sys.modules, {"coacd": fake}):
                rep = S.settle_and_measure(sc, seconds=0.05, support_tops={"table": 0.0}, decompose_free=True)
            self.assertEqual(rep.proxy_fallback, ["mug"])                 # the fallback is recorded, not hidden
            self.assertEqual(rep.steps, 25)
            self.assertEqual(set(glob.glob(os.path.join(tempfile.gettempdir(), "simready_mj_*"))), before)   # nothing left behind
            cert = os.path.join(d, "s.certificate.json"); open(cert, "w").write("{not json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main(["settle", scene, "--proxy", "hull", "--seconds", "0.05"])
            self.assertIn(cm.exception.code, (0, 1))
            report = json.loads(buf.getvalue())                           # exactly one JSON document on stdout
            self.assertIn("hull", report)
            self.assertEqual(open(cert).read(), "{not json")              # the broken certificate is untouched


class Export(unittest.TestCase):
    def test_pieces_never_share_a_file_with_another_body(self):
        from unittest import mock
        from simready.io.export import export_scene
        from simready.gates import settle_mujoco as S
        cup = box("cup", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05)); other = box("cup_p0", (0.4, 0.4, 0.4), (2.0, 0.0, 0.2))
        sc = table_scene(cup, other)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "coacd_pieces_info", lambda b: ([(b.verts.copy(), b.faces.copy())], False)):
            m = export_scene(sc, d, decompose=True)
            piece = os.path.join(d, m["bodies"][1]["collision_pieces"][0])
            v = np.array([[float(x) for x in ln.split()[1:]] for ln in open(piece) if ln.startswith("v ")])
            self.assertTrue(np.allclose(np.ptp(v, axis=0), [0.1, 0.1, 0.1]))
            paths = [b["mesh"] for b in m["bodies"]] + [p for b in m["bodies"] for p in b["collision_pieces"]]
            self.assertEqual(len(paths), len(set(paths)))
            import mujoco
            mujoco.MjModel.from_xml_path(os.path.join(d, "scene.xml"))

    def test_vertical_sheet_is_kept_as_a_thin_solid(self):
        from simready.io.export import export_scene
        import mujoco
        v = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, -1.0, 2.0]]); f = np.array([[0, 1, 2], [0, 2, 3]])
        wall = Body.from_mesh("wall", v, f, fixed=True, tags={"fixture"})
        sc = table_scene(wall, box("a", (0.1, 0.1, 0.1), (0.3, 0.0, 0.05)))
        with tempfile.TemporaryDirectory() as d:
            m = export_scene(sc, d)
            entry = next(b for b in m["bodies"] if b["name"] == "wall")
            self.assertFalse(entry["flat"]); self.assertAlmostEqual(entry["thickened_m"], 0.01)
            model = mujoco.MjModel.from_xml_path(os.path.join(d, "scene.xml"))
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wall"), 1)
            v2 = np.array([[float(x) for x in ln.split()[1:]] for ln in open(os.path.join(d, entry["mesh"])) if ln.startswith("v ")])
            self.assertAlmostEqual(float(np.ptp(v2[:, 0])), 0.01, places=6)

    def test_declared_missing_fixture_is_an_error(self):
        from simready.cli import load_any
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            json.dump({"table": {"usd_path": os.path.join(d, "missing.usda")}, "objects": []}, open(p, "w"))
            with self.assertRaises(FileNotFoundError):
                load_any(p)
            json.dump({"base_scene": os.path.join(d, "missing_scene.usda"), "objects": []}, open(p, "w"))
            with self.assertRaises(FileNotFoundError):
                load_any(p)
            json.dump({"objects": []}, open(p, "w"))
            sc = load_any(p)                                              # nothing declared: a synthetic slab, and it says so
            self.assertIn("synthetic_table", sc.meta)

    def test_settled_check_sees_a_toppled_body(self):
        from simready.cli import settled_check
        from types import SimpleNamespace
        a = box("a", (0.05, 0.05, 0.2), (0.0, 0.0, 0.1))
        sc = table_scene(a)
        prog = parse_program("program\n  on_support(a, table)\n  upright(a)")
        # the body ends lying on its side: rotated 90 degrees about x, its centre 2.5 cm above the table
        r = SimpleNamespace(final_pose={"a": (np.array([0.0, 0.0, 0.025]), np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]))})
        out = settled_check(sc, r, prog)
        self.assertIn("upright(a)", out["failed_predicates"])
        self.assertNotIn("on_support(a,table)", out["failed_predicates"])
        # rocked by 5 degrees on a hull facet, resting 4 mm high: still upright and on the table at the settle tolerances
        r2 = SimpleNamespace(final_pose={"a": (np.array([0.0, 0.0, 0.104]), np.array([math.cos(math.radians(2.5)), math.sin(math.radians(2.5)), 0.0, 0.0]))})
        self.assertEqual(settled_check(sc, r2, prog)["failed_predicates"], [])

    def test_export_loads_in_mujoco_and_usd(self):
        try:
            import mujoco
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("mujoco or usd-core not installed")
        from simready.io.export import export_scene
        sheet = box("GroundPlane", (4.0, 4.0, 1e-9), (0.0, 0.0, -0.7), fixed=True, tags=("ground", "fixture"))
        sheet.verts[:, 2] = 0.0                                       # a flat sheet: no volume to hull
        sc = table_scene(box("mug", (0.1, 0.1, 0.1), (0.0, 0.0, 0.05), tags=("object",)), box("odd name-1", (0.05, 0.05, 0.05), (0.2, 0.0, 0.025)), sheet)
        with tempfile.TemporaryDirectory() as d:
            m = export_scene(sc, d)
            self.assertEqual([b["prim"] for b in m["bodies"]], ["table", "mug", "odd_name_1", "GroundPlane"])
            self.assertEqual([b["name"] for b in m["bodies"] if b["flat"]], ["GroundPlane"])
            self.assertAlmostEqual(m["ground_z"], -0.705)
            model = mujoco.MjModel.from_xml_path(os.path.join(d, "scene.xml"))     # MuJoCo compiles it
            self.assertEqual(model.nbody, 1 + 3)                                     # world + table, mug, odd_name_1
            self.assertEqual(model.njnt, 2)                                          # two free joints
            data = mujoco.MjData(model); mujoco.mj_forward(model, data)
            mug = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
            self.assertTrue(np.allclose(data.xpos[mug], [0.0, 0.0, 0.05], atol=1e-6))
            for _ in range(100):
                mujoco.mj_step(model, data)
            self.assertLess(abs(data.xpos[mug][2] - 0.05), 2e-3)                    # it rests on the table
            stage = Usd.Stage.Open(os.path.join(d, "scene.usda"))                    # the USD stage carries the physics schemas
            self.assertTrue(stage.GetPrimAtPath("/World/mug").HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(stage.GetPrimAtPath("/World/table").HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertTrue(stage.GetPrimAtPath("/World/table/mesh").HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(stage.GetPrimAtPath("/World/GroundPlane").IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/physicsScene").IsValid())
            mesh = stage.GetPrimAtPath("/World/mug/mesh")
            self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), "convexDecomposition")
            manifest = json.load(open(os.path.join(d, "manifest.json")))
            self.assertEqual(manifest["bodies"][1]["position"], [0.0, 0.0, 0.05])
            self.assertTrue(manifest["usd_written"])


if __name__ == "__main__":
    unittest.main()


ROBOLAB_DIR = Path(os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "RoboLab")))
_REMOTE_USD = ROBOLAB_DIR / "assets" / "objects" / "hot3d" / "remote_control.usd"
_WORKDESK = ROBOLAB_DIR / "assets" / "scenes" / "workdesk_snacks.usda"


class RestOrientation(unittest.TestCase):
    """Assets whose authored pose cannot stand are handled in their resting frame."""

    def test_table_and_rotation(self):
        from simready.io.asset_rest import rest_rotation, rest_dims, rotation_to_up, table
        self.assertIn("remote_control", table())
        self.assertEqual(table()["remote_control"]["up_axis"], "-y")
        R = rest_rotation("/some/checkout/assets/objects/hot3d/remote_control.usd")
        np.testing.assert_allclose(R @ np.array([0.0, -1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
        np.testing.assert_allclose(rest_dims([0.036, 0.025, 0.164], R), [0.036, 0.164, 0.025], atol=1e-12)
        self.assertIsNone(rest_rotation("pitcher"))
        self.assertIsNotNone(rest_rotation("remote_control_01"))
        for axis in ("+x", "-x", "+y", "-y", "+z", "-z"):
            R = rotation_to_up(axis)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    @unittest.skipUnless(_REMOTE_USD.exists(), "needs the RoboLab checkout")
    def test_catalog_remote_rests_flat(self):
        from simready.io.usd_io import body_from_object_usd
        from simready.repair.upright_s4r import decompose_zyx
        b = body_from_object_usd("remote_control", str(_REMOTE_USD), (0.5, 0.0, 0.3), yaw_deg=30.0, tags={"object"})
        v = b.world_vertices()
        ext = v.max(0) - v.min(0)
        self.assertLess(ext[2], 0.03, "the remote control must lie flat in its resting frame")
        self.assertGreater(max(ext[0], ext[1]), 0.15)
        _, roll, pitch = decompose_zyx(b.rotation)
        self.assertLess(max(abs(roll), abs(pitch)), 1e-9)
        self.assertIn("rest_rotation", b.meta)

    @unittest.skipUnless(_WORKDESK.exists(), "needs the RoboLab checkout")
    def test_shipped_remote_is_upright_and_geometry_unchanged(self):
        from simready.io import usd_io
        from simready.repair.upright_s4r import decompose_zyx
        sc = usd_io.load_scene_usda(str(_WORKDESK))
        b = sc["remote_control"]
        _, roll, pitch = decompose_zyx(b.rotation)
        self.assertLess(math.degrees(max(abs(roll), abs(pitch))), 1.0, "lying as shipped reads as upright in the resting frame")
        v_rest = b.world_vertices()
        saved = usd_io.rest_rotation
        try:
            usd_io.rest_rotation = lambda key: None            # load once more in the authored frame
            v_auth = usd_io.load_scene_usda(str(_WORKDESK))["remote_control"].world_vertices()
        finally:
            usd_io.rest_rotation = saved
        np.testing.assert_allclose(np.sort(v_rest, axis=0), np.sort(v_auth, axis=0), atol=1e-9)

    @unittest.skipUnless(_WORKDESK.exists(), "needs the RoboLab checkout")
    def test_written_pose_round_trip(self):
        from simready.io.usd_io import load_scene_usda, write_scene_poses
        sc = load_scene_usda(str(_WORKDESK))
        b = sc["remote_control"]
        yaw = math.radians(10.0)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        b.rotation = Rz @ b.rotation
        b.center = b.center + np.array([0.01, -0.02, 0.0])
        want = b.world_vertices()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "moved.usda")
            write_scene_poses(sc, str(_WORKDESK), out)
            got = load_scene_usda(out)["remote_control"].world_vertices()
        np.testing.assert_allclose(np.sort(got, axis=0), np.sort(want, axis=0), atol=1e-6)

    @unittest.skipUnless(_REMOTE_USD.exists(), "needs the RoboLab checkout")
    def test_layout_dims_are_resting_dims(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        import robolab_offline_layouts as L
        c = L.BY_NAME["remote_control"]
        from simready.io.asset_rest import rest_dims, rest_rotation
        d = rest_dims(c["dims"], rest_rotation("remote_control"))
        self.assertLess(d[2], 0.03)


def write_op_order_usda(path):
    """A Z-up scene whose free bodies author their xform ops in orders other than translate,
    orient, scale: a scale applied after the rotation (which reads as shear in the local map) and a
    rotation authored after the translation."""
    def mesh(extents, indent="        "):
        m = trimesh.creation.box(extents=extents)
        pts = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in m.vertices)
        idx = ", ".join(str(int(i)) for i in m.faces.reshape(-1))
        return (f'{indent}def Mesh "geo" {{\n{indent}    point3f[] points = [{pts}]\n'
                f'{indent}    int[] faceVertexCounts = [{", ".join(["3"] * len(m.faces))}]\n'
                f'{indent}    int[] faceVertexIndices = [{idx}]\n{indent}}}\n')
    c, s = math.cos(math.radians(20.0) / 2), math.sin(math.radians(20.0) / 2)
    text = ('#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Xform "World" {\n'
            '    def Xform "table" {\n        double3 xformOp:translate = (0, 0, -0.025)\n'
            '        uniform token[] xformOpOrder = ["xformOp:translate"]\n' + mesh((1.0, 1.0, 0.05)) + "    }\n"
            '    def Xform "scaled_after_turn" {\n        double3 xformOp:translate = (0.2, 0, 0.05)\n'
            f'        quatf xformOp:orient = ({c:.7f}, 0, 0, {s:.7f})\n        float3 xformOp:scale = (2, 1, 1)\n'
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale", "xformOp:orient"]\n' + mesh((0.1, 0.1, 0.1)) + "    }\n"
            '    def Xform "turned_after_move" {\n        double3 xformOp:translate = (-0.2, 0.1, 0.05)\n'
            f'        quatf xformOp:orient = ({c:.7f}, 0, 0, {s:.7f})\n'
            '        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:translate"]\n' + mesh((0.1, 0.1, 0.1)) + "    }\n}\n")
    open(path, "w").write(text)


class PiecesSheetsAndFrames(unittest.TestCase):
    def test_piece_of_a_body_inside_another_body_is_contained(self):
        # a body made of two closed pieces, one at the bin's centre and one 3 m away: the bodies'
        # AABBs do not nest and the surfaces are apart, only the piece-wise test can see the inner one
        bin_ = box("bin", (2.0, 2.0, 2.0), (0.0, 0.0, 1.0), fixed=True, tags=("fixture",))
        two = trimesh.util.concatenate([trimesh.creation.box(extents=(0.2, 0.2, 0.2)).apply_translation([-1.5, 0, 0]),
                                        trimesh.creation.box(extents=(0.2, 0.2, 0.2)).apply_translation([1.5, 0, 0])])
        body = Body.from_mesh("two", two.vertices, two.faces, center=np.array([1.5, 0.0, 1.0]))
        pairs = pair_signed_distances([bin_, body])
        self.assertEqual(len(pairs), 1)
        self.assertLess(pairs[0][2], 0.0)
        self.assertTrue(pairs[0].contained)
        outside = Body.from_mesh("two", two.vertices, two.faces, center=np.array([4.0, 0.0, 1.0]))
        self.assertFalse(any(p[2] < 0.0 for p in pair_signed_distances([bin_, outside])))

    def test_turned_sheet_is_a_wall_not_the_ground(self):
        from simready.io.export import export_scene
        from simready.gates.settle_mujoco import settle_and_measure
        # a square authored in its local xy plane, stood upright by its pose: a wall on the table
        v = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]); f = np.array([[0, 1, 2], [0, 2, 3]])
        Ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        wall = Body.from_mesh("wall", v, f, center=np.array([0.5, 0.0, 1.0]), rotation=Ry, fixed=True, tags={"fixture"})
        sc = table_scene(wall, box("a", (0.05, 0.05, 0.05), (0.5, 0.0, 2.025)))   # a small cube standing on the wall's top edge
        with tempfile.TemporaryDirectory() as d:
            m = export_scene(sc, d)
            entry = next(b for b in m["bodies"] if b["name"] == "wall")
            self.assertFalse(entry["flat"]); self.assertAlmostEqual(entry["thickened_m"], 0.01)
        r = settle_and_measure(sc, seconds=0.3, timestep=0.002)
        self.assertLess(r.peak_disp, 0.2)          # the wall holds the cube up; without it the cube falls 0.44 m

    def test_settle_measures_the_state_it_integrated(self):
        from simready.gates.settle_mujoco import settle_and_measure
        sc = table_scene(box("a", (0.1, 0.1, 0.1), (0.0, 0.0, 1.0)))      # a cube 1 m above the table
        r = settle_and_measure(sc, seconds=0.002, timestep=0.002)
        self.assertGreater(r.peak_speed, 0.015)    # one step of free fall: g dt
        self.assertGreater(r.peak_disp, 0.0)

    def test_layout_keeps_the_base_scene_unresolved_children(self):
        from simready.cli import load_any
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "base.usda"); write_usda(base, ghost=True)
            layout = os.path.join(d, "layout.json")
            json.dump({"base_scene": base, "objects": []}, open(layout, "w"))
            sc = load_any(layout)
            self.assertEqual(sc.meta.get("dropped"), ["ghost"])
            self.assertEqual(sc.meta.get("base_scene"), base)

    def test_object_usd_frame_is_checked(self):
        from simready.io.usd_io import body_from_object_usd
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cube.usda"); write_usda(path, meters_per_unit=0.01, ghost=False)
            with self.assertRaises(ValueError):
                body_from_object_usd("cube", path, (0, 0, 0))

    def test_written_pose_keeps_the_geometry_for_other_op_orders(self):
        from simready.io.usd_io import load_scene_usda, write_scene_poses
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "ops.usda"); write_op_order_usda(src)
            sc = load_scene_usda(src)
            yaw = math.radians(30.0)
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            want = {}
            for b in sc.free():
                b.rotation = Rz @ b.rotation
                b.center = b.center + np.array([0.03, -0.02, 0.0])
                want[b.name] = b.world_vertices()
            out = os.path.join(d, "moved.usda")
            write_scene_poses(sc, src, out)
            back = load_scene_usda(out)
            self.assertEqual(sorted(want), sorted(b.name for b in back.free()))
            for name, w in want.items():
                got = back[name].world_vertices()
                np.testing.assert_allclose(np.sort(got, axis=0), np.sort(w, axis=0), atol=1e-6, err_msg=name)
