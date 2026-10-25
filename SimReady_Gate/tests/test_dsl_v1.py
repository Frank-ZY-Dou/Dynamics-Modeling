"""Regression tests for the v1 statements: parsing and JSON validation, the rows they compile to
(checked against finite differences of their own residuals), the repair reaching them on small box
scenes, preferences giving way to requirements, and programs the contacts contradict. Run with
`python -m unittest discover tests`.
"""
import contextlib
import copy
import io
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simready.scene.model import Scene, Body  # noqa: E402
from simready.dsl.schema import validate_program_json, ProgramError, api_schema, prompt_for  # noqa: E402
from simready.dsl.model import parse_program  # noqa: E402
from simready.dsl.compile import compile_program, check_predicates, effective_supports, CompileError  # noqa: E402
from simready.dsl import statements as V1  # noqa: E402
from simready.repair.upright_s4r import State, repair_upright, rotz  # noqa: E402
from simready.gates.verify import verify_scene  # noqa: E402


def box(name, extents, center, fixed=False, tags=(), yaw=0.0):
    m = trimesh.creation.box(extents=extents)
    return Body.from_mesh(name, m.vertices, m.faces, center=np.array(center, dtype=float), rotation=rotz(yaw),
                          fixed=fixed, tags=set(tags))


def table_scene(*objects):
    table = box("table", (0.7, 1.0, 0.05), (0.55, 0.0, -0.025), fixed=True, tags=("support", "fixture"))
    return Scene([table, *objects])


HEAD = "program\n  no_penetration(*, margin=0.005)\n  on_support(*, table)\n  upright(*)\n  within(*, table.top, inset=0.02)\n"


def state_of(sc, scale=1.0):
    free = [k for k, b in enumerate(sc.bodies) if not b.fixed]
    return State(sc, free, {k: 3 * i for i, k in enumerate(free)}, np.array([sc.bodies[k].center[:2] for k in free]),
                 np.zeros(len(free)), np.zeros(len(free)), np.zeros(len(free)), scale, 0.0)


def repaired(sc, text, **kw):
    prog = parse_program(text)
    spec = compile_program(prog, sc, ds_max=0.05, tail_iters=30, **kw)
    res = repair_upright(sc, spec)
    preds = check_predicates(prog, sc)
    return res, {p[0]: p for p in preds}, verify_scene(sc)


class Parsing(unittest.TestCase):
    def test_v1_statements_parse(self):
        text = ("program\n  reach(mug, r=0.75, min=0.3)  clearance(mug, gap=0.03)  approach(mug, length=0.2, width=0.1, toward=robot)\n"
                "  facing(mug, axis=-x, tol=20)  facing(mug, bowl)  yaw(box, deg=-30, mod=90, tol=3)  parallel(fork, knife)\n"
                "  row(a, b, c, axis=-y, gap=0.02)  align(a, b, axis=x, edge=max)  evenly_spaced(a, b, c, axis=y)\n"
                "  grid(a, b, c, d, cols=2, dx=0.1, dy=-0.1)  cluster(a, b, r=0.1)  symmetric(fork, knife, plate, axis=y)\n"
                "  between(cup, plate, bottle, axis=y, gap=0.05)  centered(cup, saucer)  offset(fork, plate, dy=0.15, frame=local)\n"
                "  attach(lid, pot)  within(apple, table.top, side=left, frac=0.5)  within(apple, x0=0.3, y1=0.2)\n"
                "  keep_out(*, oven, side=near, depth=0.3)  max_move(*, d=0.03)  max_turn(*, deg=10)\n"
                "  stack(b1, b2, b3)  balanced_on(b2, b1, margin=0.01)  left_of(mug, bowl, gap=0.1, soft=0.5)\n")
        prog = parse_program(text)
        self.assertEqual(len(prog.statements), 25)
        self.assertTrue(prog.statements[-1].soft)
        self.assertFalse(prog.statements[0].soft)

    def test_v1_errors_are_syntax_errors(self):
        bad = ["row(a)", "row(*, b)", "row(a, a)", "row(a, b, axis=z)", "grid(a, b)", "grid(a, b, cols=1.5)", "cluster(a, b)",
               "yaw(a)", "yaw(a, deg=0, mod=45)", "within(a)", "within(a, t, frac=0.5)", "within(a, t, side=left, frac=1.5)",
               "within(a, t, side=left, frac=0.5, depth=0.1)", "keep_out(a, x0=0.5, x1=0.4)", "symmetric(a, a, b)",
               "left_of(a, b, soft=2)", "reach(a, r=0.3, min=0.5)", "facing(a, b, x=0.1)", "max_turn(a, deg=-1)",
               "offset(a, b, frame=b)", "approach(a, toward=up)", "within(a, t, side=middle)", "at(a)", "at(a, z=1)", "beside(a, b)", "beside(a, b, side=up)", "beside(a, b, side=left, gap=0.05, max=0.02)"]
        for s in bad:
            with self.assertRaises(SyntaxError, msg=s):
                parse_program("program\n  " + s)

    def test_json_round_trip(self):
        sc = table_scene(box("mug", (0.08, 0.08, 0.1), (0.5, 0.1, 0.05)), box("bowl", (0.15, 0.15, 0.06), (0.6, -0.1, 0.03)),
                         box("cup", (0.07, 0.07, 0.08), (0.45, -0.2, 0.04)))
        obj = {"statements": [
            {"op": "row", "members": ["mug", "bowl", "cup"], "axis": "-y", "gap": 0.02, "reason": "a row"},
            {"op": "reach", "a": "*", "r": 0.8},
            {"op": "symmetric", "a": "mug", "b": "cup", "c": "bowl"},
            {"op": "within", "a": "mug", "b": "table.top", "side": "left", "frac": 0.3},
            {"op": "within", "a": "cup", "b": "table.top"},
            {"op": "keep_out", "a": "*", "x0": 0.4, "x1": 0.5},
            {"op": "grid", "members": ["mug", "bowl", "cup"], "cols": 3},
            {"op": "left_of", "a": "mug", "b": "bowl", "soft": 0.5}]}
        dsl, notes = validate_program_json(obj, sc)
        self.assertIn("row(mug, bowl, cup, gap=0.02, axis=-y)", dsl)
        self.assertIn("symmetric(mug, cup, bowl)", dsl)
        self.assertIn("within(mug, table.top, side=left, frac=0.3)", dsl)
        self.assertIn("within(cup, table.top, inset=0.02)", dsl)          # a whole region keeps the recorded default inset
        self.assertIn("grid(mug, bowl, cup, cols=3)", dsl)
        self.assertIn("left_of(mug, bowl, gap=0.05, soft=0.5)", dsl)
        prog = parse_program(dsl)
        compile_program(prog, copy.deepcopy(sc))

    def test_json_errors_name_the_statement(self):
        sc = table_scene(box("mug", (0.08, 0.08, 0.1), (0.5, 0.1, 0.05)), box("bowl", (0.15, 0.15, 0.06), (0.6, -0.1, 0.03)))
        cases = [{"op": "row", "members": ["mug"]}, {"op": "row", "members": ["mug", "ghost"]}, {"op": "row", "a": "mug"},
                 {"op": "symmetric", "a": "mug", "b": "bowl"}, {"op": "grid", "members": ["mug", "bowl"]},
                 {"op": "centered", "a": "mug", "b": "bowl", "c": "mug"}, {"op": "yaw", "a": "mug", "deg": True},
                 {"op": "within", "a": "mug", "b": "table.top", "frac": 0.5}, {"op": "row", "members": ["mug", "bowl"], "axis": "z"},
                 {"op": "row", "members": ["mug", "*"]}, {"op": "facing", "a": "mug", "b": "bowl", "x": 0.2}]
        for st in cases:
            with self.assertRaises(ProgramError, msg=st) as cm:
                validate_program_json({"statements": [st]}, sc)
            self.assertIn("statement 0", str(cm.exception))

    def test_api_schema_constrains_members(self):
        sc = table_scene(box("mug", (0.08, 0.08, 0.1), (0.5, 0.1, 0.05)))
        sch = api_schema(sc)
        items = sch["properties"]["statements"]["items"]["properties"]
        self.assertEqual(set(items["members"]["items"]["enum"]), {"table", "mug"})
        self.assertNotIn("minimum", json.dumps(sch))
        text = prompt_for("put the mug on the left", sc)
        for word in ("row(members, axis, gap)", "keep_out", "soft (0 to 1]", "near (-x"):
            self.assertIn(word.split("(")[0], text)


class Rows(unittest.TestCase):
    """Every linearised row predicts the change of its own residual under a small step."""

    def _residuals(self, gen, sc, step):
        sc2 = copy.deepcopy(sc)
        free = [k for k, b in enumerate(sc2.bodies) if not b.fixed]
        for i, k in enumerate(free):
            b = sc2.bodies[k]
            b.center = b.center + np.array([step[3 * i], step[3 * i + 1], 0.0])
            b.rotation = rotz(step[3 * i + 2]) @ b.rotation
        return gen(state_of(sc2)), free

    def test_linearisations_match_finite_differences(self):
        texts = ["facing(a, b, axis=+x)", "facing(a, axis=-y)", "offset(a, b, dx=0.1, dy=0.05, frame=local)", "attach(a, b)",
                 "cluster(a, b, c, r=0.05)", "reach(a, r=0.3, min=0.2)", "balanced_on(a, b)", "parallel(a, b)", "yaw(a, deg=40)"]
        rng = np.random.default_rng(0)
        for text in texts:
            sc = table_scene(box("a", (0.08, 0.05, 0.1), (0.5, 0.1, 0.05), yaw=0.4), box("b", (0.1, 0.1, 0.1), (0.7, -0.15, 0.05), yaw=-0.2),
                             box("c", (0.05, 0.05, 0.05), (0.4, -0.2, 0.025)))
            sc["a"].meta["com"] = np.array([0.02, -0.01, 0.0])
            prog = parse_program("program\n  " + text)
            compile_program(prog, sc)                                  # records the statement's reference poses
            st = prog.statements[0]
            ctx = V1.Ctx(sc, {b.name: k for k, b in enumerate(sc.bodies)}, {}, lambda tok, exclude=(): [b.name for b in sc.free()] if tok == "*" else [tok])
            gen = V1.compile_statement(st, ctx)
            rows0 = gen(state_of(sc))
            free = [k for k, b in enumerate(sc.bodies) if not b.fixed]
            col = {k: 3 * i for i, k in enumerate(free)}
            step = rng.normal(0.0, 1.0, 3 * len(free)) * np.array([1e-4, 1e-4, 1e-3] * len(free))
            rows1, _ = self._residuals(gen, sc, step)
            self.assertEqual(len(rows0), len(rows1), text)
            for (co, lo0, hi0), (_, lo1, hi1) in zip(rows0, rows1):
                pred = sum(v * step[col[k] + {"x": 0, "y": 1, "yaw": 2}[c]] for (k, c), v in co.items())
                finite = [b for b in (lo0, hi0) if math.isfinite(b)]
                after = [b for b in (lo1, hi1) if math.isfinite(b)]
                for b0, b1 in zip(finite, after):
                    # the row bounds are (target - current residual): a step changes them by minus its effect
                    self.assertAlmostEqual(b0 - b1, pred, delta=2e-6 + 0.02 * abs(pred), msg=text)

    def test_grid_rows_are_exact_and_free_pitch_is_shared(self):
        bodies = [box(f"g{i}", (0.04, 0.04, 0.04), (0.4 + 0.1 * (i // 3), -0.2 + 0.1 * (i % 3), 0.02)) for i in range(6)]
        sc = table_scene(*bodies)
        for text in ("grid(g0, g1, g2, g3, g4, g5, cols=3)", "grid(g0, g1, g2, g3, g4, g5, cols=3, dx=0.1, dy=0.1)"):
            prog = parse_program("program\n  " + text)
            spec = compile_program(prog, sc)
            rows = spec.rows_fn(state_of(sc))
            self.assertTrue(rows)
            self.assertTrue(all(lo <= 1e-9 <= hi + 2e-9 for _, lo, hi in rows), text)     # a regular grid satisfies its rows
            self.assertTrue(all(p[1] for p in check_predicates(prog, sc)))
        sc["g4"].center = sc["g4"].center + np.array([0.0, 0.03, 0.0])
        prog = parse_program("program\n  grid(g0, g1, g2, g3, g4, g5, cols=3)")
        self.assertFalse(check_predicates(prog, sc)[0][1])

    def test_keep_out_takes_the_nearest_side(self):
        sc = table_scene(box("a", (0.1, 0.1, 0.1), (0.52, 0.04, 0.05)))
        prog = parse_program("program\n  keep_out(a, x0=0.4, x1=0.6, y0=-0.1, y1=0.1)")
        rows = compile_program(prog, sc).rows_fn(state_of(sc))
        (co, lo, hi), = rows
        self.assertEqual(list(co), [(1, "y")])                        # out through the left side, 0.11 m away
        self.assertAlmostEqual(lo, 0.1 - (0.04 - 0.05), places=9)

    def test_regions(self):
        lo, hi = (0.2, -0.5), (0.9, 0.5)
        self.assertEqual(V1.region_rect(lo, hi, {"side": "left", "frac": 0.25}), (0.2, 0.9, 0.25, 0.5))
        self.assertEqual(V1.region_rect(lo, hi, {"side": "near", "depth": 0.1}), (0.1, 0.2, -0.5, 0.5))
        x0, x1, y0, y1 = V1.region_rect(lo, hi, {"side": "center", "frac": 0.5})
        self.assertAlmostEqual(x1 - x0, 0.35); self.assertAlmostEqual(y1 - y0, 0.5)
        self.assertEqual(V1.region_rect(lo, hi, {"x0": 0.5, "y1": 0.0}), (0.5, 0.9, -0.5, 0.0))
        with self.assertRaises(V1.StatementError):
            V1.region_rect(lo, hi, {"x0": 0.95})


class Repair(unittest.TestCase):
    def test_arrangements_are_reached_from_a_heap(self):
        rng = np.random.default_rng(1)
        objs = []
        for i in range(6):
            e = rng.uniform(0.04, 0.08, 3); e[2] = rng.uniform(0.05, 0.1)
            objs.append(box(f"o{i}", e, (0.55 + rng.normal(0, 0.05), rng.normal(0, 0.05), e[2] / 2), yaw=rng.uniform(-0.5, 0.5)))
        for extra in ("row(o0, o1, o2, o3, axis=+y, gap=0.02)", "grid(o0, o1, o2, o3, o4, o5, cols=3)", "cluster(o0, o1, o2, r=0.08)",
                      "symmetric(o0, o1, o2, axis=y)  min_distance(o0, o1, r=0.2)", "between(o0, o1, o2, axis=y, gap=0.06)",
                      "keep_out(*, table, x0=0.45, x1=0.65, y0=-0.12, y1=0.12)", "approach(o0, length=0.25, width=0.10)",
                      "facing(*, axis=+x, tol=10)", "yaw(*, deg=0, mod=90, tol=2)", "at(o0, x=0.45, y=0.2, tol=0.005)",
                      "beside(o0, o1, side=left, gap=0.02, max=0.05)  beside(o2, o1, side=near, gap=0.03)", "reach(*, r=0.6, min=0.45)", "clearance(o0, gap=0.04)",
                      "align(o0, o1, o2, axis=x, edge=max)  evenly_spaced(o3, o4, o5, axis=y)"):
            sc = table_scene(*copy.deepcopy(objs))
            res, preds, rep = repaired(sc, HEAD + "  " + extra + "\n")
            self.assertEqual(rep.pen_pairs, 0, extra)
            failed = [n for n, p in preds.items() if not p[1]]
            self.assertEqual(failed, [], extra)

    def test_stacks_and_balance(self):
        sc = table_scene(box("b1", (0.2, 0.2, 0.08), (0.5, -0.2, 0.04)), box("b2", (0.15, 0.15, 0.07), (0.6, 0.1, 0.035)),
                         box("b3", (0.1, 0.1, 0.06), (0.45, 0.25, 0.03)))
        res, preds, rep = repaired(sc, "program\n  no_penetration(*)\n  upright(*)\n  on_support(*, table)\n  stack(b1, b2, b3)\n")
        self.assertEqual(rep.pen_pairs, 0)
        self.assertTrue(all(p[1] for p in preds.values()), [n for n, p in preds.items() if not p[1]])
        self.assertIn("stack(b3,b2):on", preds)
        self.assertNotIn("on_support(b2,table)", preds)               # the wildcard yields to the stack
        self.assertGreater(sc["b3"].world_aabb()[0][2], sc["b2"].world_aabb()[1][2] - 1e-3)
        self.assertEqual(effective_supports(parse_program("program\n  stack(b1, b2)\n  on_support(*, table)"), sc), {"b1": "table", "b2": "b1", "b3": "table"})
        sc = table_scene(box("base", (0.10, 0.10, 0.08), (0.5, 0.0, 0.04)), box("top", (0.25, 0.08, 0.04), (0.65, 0.2, 0.02)))
        sc["top"].meta["com"] = np.array([0.06, 0.0, 0.0])           # a heavy end: the balance point is off the reference centre
        res, preds, rep = repaired(sc, "program\n  no_penetration(*)\n  upright(*)\n  on_support(base, table)\n  on_support(top, base)\n  balanced_on(top, base)\n")
        self.assertTrue(preds["balanced_on(top,base)"][1])
        com = sc["top"].center[:2] + (sc["top"].rotation @ sc["top"].meta["com"])[:2]
        r = sc["base"].rotation[:2, :2].T @ (com - sc["base"].center[:2])          # in the base's own frame
        self.assertLess(float(np.abs(r).max()), 0.05 - 0.01 + 2e-3)

    def test_centered_and_attach(self):
        sc = table_scene(box("saucer", (0.14, 0.14, 0.02), (0.5, 0.0, 0.01)), box("cup", (0.08, 0.08, 0.09), (0.62, 0.1, 0.045)))
        res, preds, rep = repaired(sc, "program\n  no_penetration(*)\n  upright(*)\n  on_support(saucer, table)\n  on_support(cup, saucer)\n  centered(cup, saucer, tol=0.005)\n")
        self.assertTrue(preds["centered(cup,saucer)"][1] and preds["on_support(cup,saucer)"][1])
        sc = table_scene(box("pot", (0.15, 0.15, 0.1), (0.5, 0.0, 0.05)), box("lid", (0.06, 0.06, 0.02), (0.5, 0.15, 0.01)),
                         box("wall", (0.05, 0.05, 0.05), (0.7, 0.0, 0.025), fixed=True))
        before = sc["lid"].center[:2] - sc["pot"].center[:2]
        res, preds, rep = repaired(sc, "program\n  no_penetration(*)\n  upright(*)\n  on_support(*, table)\n  attach(lid, pot)\n  left_of(pot, wall, gap=0.25)\n")
        self.assertTrue(all(p[1] for p in preds.values()))
        self.assertGreater(sc["pot"].center[1], 0.2)                       # the pot moved left ...
        self.assertTrue(np.allclose(sc["lid"].center[:2] - sc["pot"].center[:2], before, atol=0.012))   # ... and the lid with it

    def test_preference_gives_way_to_a_requirement(self):
        sc = table_scene(box("a", (0.06, 0.06, 0.06), (0.5, 0.0, 0.03)), box("b", (0.06, 0.06, 0.06), (0.5, 0.2, 0.03)))
        res, preds, rep = repaired(sc, HEAD + "  left_of(a, b, gap=0.1)\n  left_of(b, a, gap=0.1, soft=0.3)\n")
        self.assertTrue(preds["left_of(a,b,gap=0.1)"][1])
        self.assertFalse(preds["left_of(b,a,gap=0.1)"][1])
        self.assertTrue(preds["left_of(b,a,gap=0.1)"].soft and not preds["left_of(a,b,gap=0.1)"].soft)
        sc = table_scene(box("a", (0.08, 0.05, 0.06), (0.5, 0.0, 0.03), yaw=0.3))
        res, preds, rep = repaired(sc, HEAD + "  max_turn(a, deg=3)\n  yaw(a, deg=45, tol=2, soft=0.5)\n")
        self.assertTrue(preds["max_turn(a,deg=3)"][1])

    def test_program_the_contacts_contradict_leaves_no_penetration(self):
        # both bodies on the table yet centred on each other: impossible; the contacts win
        sc = table_scene(box("a", (0.08, 0.08, 0.08), (0.5, 0.0, 0.04)), box("b", (0.08, 0.08, 0.08), (0.6, 0.0, 0.04)))
        res, preds, rep = repaired(sc, HEAD + "  centered(a, b, tol=0.005)\n")
        self.assertEqual(rep.pen_pairs, 0)
        self.assertFalse(preds["centered(a,b)"][1])

    def test_max_move_is_measured_from_the_start(self):
        sc = table_scene(box("a", (0.06, 0.06, 0.06), (0.5, 0.0, 0.03)), box("b", (0.06, 0.06, 0.06), (0.53, 0.0, 0.03)))
        res, preds, rep = repaired(sc, HEAD + "  max_move(a, d=0.01)\n")
        self.assertEqual(rep.pen_pairs, 0)
        self.assertTrue(preds["max_move(a,d=0.01)"][1])
        self.assertLess(float(np.linalg.norm(sc["a"].center[:2] - [0.5, 0.0])), 0.01 + 2e-3)
        self.assertGreater(float(np.linalg.norm(sc["b"].center[:2] - [0.53, 0.0])), 0.02)    # b made the room
        with self.assertRaises(CompileError):
            check_predicates(parse_program("program\n  max_move(a, d=0.01)"), sc)          # no start pose recorded

    def test_projection_is_reported_and_optional(self):
        objs = [box(f"o{i}", (0.05, 0.05, 0.05), (0.5 + 0.01 * i, 0.01 * i, 0.025)) for i in range(4)]
        sc = table_scene(*copy.deepcopy(objs))
        res, preds, _ = repaired(sc, HEAD + "  row(o0, o1, o2, o3, axis=-y, gap=0.02)\n")
        self.assertTrue(any("solved alone" in n for n in res.notes))
        sc = table_scene(*copy.deepcopy(objs))
        res, preds, _ = repaired(sc, HEAD + "  row(o0, o1, o2, o3, axis=-y, gap=0.02)\n", project_iters=0)
        self.assertFalse(any("solved alone" in n for n in res.notes))


class Headings(unittest.TestCase):
    def test_parallel_is_about_long_sides(self):
        # a knife authored along its local y and a board along its local x: parallel when the long sides are
        sc = table_scene(box("knife", (0.03, 0.25, 0.01), (0.45, -0.25, 0.005)), box("board", (0.38, 0.22, 0.02), (0.55, 0.1, 0.01)))
        res, preds, rep = repaired(sc, HEAD + "  parallel(knife, board, tol=2)\n  yaw(board, deg=90, axis=long, tol=2)\n")
        self.assertTrue(preds["parallel(knife,board)"][1] and preds["yaw(board,deg=90)"][1])
        long_k = math.atan2(*(sc["knife"].rotation @ [0.0, 1.0, 0.0])[1::-1])
        long_b = math.atan2(*(sc["board"].rotation @ [1.0, 0.0, 0.0])[1::-1])
        self.assertLess(abs(((long_k - long_b) + math.pi / 2) % math.pi - math.pi / 2), math.radians(2.5))
        self.assertLess(abs((long_b - math.pi / 2 + math.pi / 2) % math.pi - math.pi / 2), math.radians(2.5))

    def test_at_is_a_hard_position(self):
        sc = table_scene(box("plate", (0.17, 0.17, 0.02), (0.6, 0.2, 0.01)), box("cup", (0.08, 0.08, 0.1), (0.5, 0.0, 0.05)))
        res, preds, rep = repaired(sc, HEAD + "  at(plate, x=0.5, y=0.0, tol=0.005)\n")
        self.assertTrue(preds["at(plate)"][1])
        self.assertLess(float(np.abs(sc["plate"].center[:2] - [0.5, 0.0]).max()), 0.005 + 2e-3)
        self.assertEqual(rep.pen_pairs, 0)                               # the cup made room


class Ordering(unittest.TestCase):
    def test_row_sorts_its_members(self):
        objs = [box("tall", (0.06, 0.06, 0.15), (0.5, -0.1, 0.075)), box("short", (0.06, 0.06, 0.06), (0.5, 0.0, 0.03)),
                box("mid", (0.06, 0.06, 0.10), (0.5, 0.1, 0.05))]
        sc = table_scene(*objs)
        res, preds, rep = repaired(sc, HEAD + "  row(tall, short, mid, axis=-y, gap=0.02, order=height, spacing=even, tol=0.005)\n")
        self.assertIn("row(tall,short,mid):even", preds)
        self.assertTrue(all(p[1] for p in preds.values()), [n for n, p in preds.items() if not p[1]])
        ys = {n: float(sc[n].center[1]) for n in ("short", "mid", "tall")}
        self.assertGreater(ys["short"], ys["mid"]); self.assertGreater(ys["mid"], ys["tall"])   # shortest at the left (+y)
        sc["tall"].meta["physics"] = {"mass": 0.1}; sc["short"].meta["physics"] = {"mass": 2.0}; sc["mid"].meta["physics"] = {"mass": 1.0}
        from simready.dsl.statements import sorted_members
        st = parse_program("program\n  row(tall, short, mid, order=-mass)").statements[0]
        self.assertEqual(sorted_members(st, sc), ["short", "mid", "tall"])


class EvenSpacing(unittest.TestCase):
    def test_every_spacing_within_tol_of_the_mean(self):
        # spacings 0.10, 0.12, 0.10, 0.12: second differences of 0.02 but every spacing 0.01 from the mean
        objs = [box(f"c{i}", (0.05, 0.05, 0.1), (0.5, y, 0.05)) for i, y in enumerate((0.0, 0.10, 0.22, 0.32, 0.44))]
        sc = table_scene(*objs)
        names = ", ".join(b.name for b in objs)
        ok = {tol: check_predicates(parse_program(f"program\n  evenly_spaced({names}, axis=y, tol={tol})"), sc)[0][1] for tol in (0.005, 0.0105)}
        self.assertEqual(ok, {0.005: False, 0.0105: True})
        sc = table_scene(*copy.deepcopy(objs))
        res, preds, rep = repaired(sc, HEAD + f"  row({names}, axis=+y, gap=0.02, spacing=even, tol=0.002)\n")
        ys = np.diff([float(sc[b.name].center[1]) for b in objs])
        self.assertLess(float(np.abs(ys - ys.mean()).max()), 0.002 + 2e-3)


class Seating(unittest.TestCase):
    def test_a_ridge_between_the_corners_is_cleared(self):
        # a carton whose top has a 1.5 mm ridge across its middle, between a box's corner rays
        base = trimesh.creation.box(extents=(0.3, 0.3, 0.1))
        ridge = trimesh.creation.box(extents=(0.004, 0.3, 0.0015)); ridge.apply_translation((0.0, 0.0, 0.05075))
        m = trimesh.util.concatenate([base, ridge])
        carton = Body.from_mesh("carton", m.vertices, m.faces, center=np.array([0.5, 0.0, 0.05075]), fixed=False)
        sc = table_scene(carton, box("top", (0.2, 0.2, 0.1), (0.5, 0.0, 0.16)))
        from simready.cli import repair_scene
        prog = parse_program("program\n  no_penetration(*)\n  upright(*)\n  on_support(carton, table)\n  stack(carton, top)\n")
        out = repair_scene(sc, prog, tail=10)
        self.assertEqual(out["after"].pen_pairs, 0)
        self.assertGreaterEqual(float(sc["top"].world_aabb()[0][2]), float(sc["carton"].world_aabb()[1][2]) - 1e-4)
        self.assertTrue(out["ok"], [p[0] for p in out["hard"] if not p[1]])


class Disjunctions(unittest.TestCase):
    def test_keep_out_side_leaves_room_in_the_bodys_region(self):
        # a 12 cm wide body held in the band |y| < 0.2 starts in a keep-out square |y| < 0.1: leaving
        # sideways cannot fit it in the band, so it has to leave along x
        sc = table_scene(box("mug", (0.12, 0.12, 0.1), (0.42, 0.01, 0.05)))
        res, preds, rep = repaired(sc, HEAD + "  within(mug, y0=-0.2, y1=0.2)\n  keep_out(mug, x0=0.3, x1=0.5, y0=-0.1, y1=0.1)\n")
        self.assertTrue(preds["within(mug,region)"][1] and preds["keep_out(mug,region)"][1], [n for n, p in preds.items() if not p[1]])

    def test_stray_vertices_do_not_lift_a_body(self):
        m = trimesh.creation.box(extents=(0.4, 0.4, 0.05))
        v = np.vstack([m.vertices, [[0.0, 0.0, 0.5]]])                   # a vertex no triangle uses, high above the top
        slab = Body.from_mesh("slab", v, m.faces, center=np.array([0.5, 0.0, 0.0]), fixed=True, tags={"support", "fixture"})
        cube = box("cube", (0.1, 0.1, 0.1), (0.5, 0.0, 0.3))
        sc = Scene([slab, cube])
        from simready.dsl.compile import support_height_for
        top = float(slab.world_vertices()[np.unique(slab.faces)][:, 2].max())
        self.assertAlmostEqual(support_height_for(sc, cube, slab), top, places=6)    # the top face, not the stray vertex above it


class Reports(unittest.TestCase):
    def test_preferences_are_reported_apart(self):
        try:
            import pxr  # noqa: F401
        except ImportError:
            self.skipTest("usd-core not installed")
        from simready.cli import main
        with tempfile.TemporaryDirectory() as d:
            layout = os.path.join(d, "layout.json"); prog = os.path.join(d, "p.json"); out = os.path.join(d, "out.json")
            cube = os.path.join(d, "cube.usda")
            m = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
            pts = ", ".join(f"({x:.4f}, {y:.4f}, {z + 0.03:.4f})" for x, y, z in m.vertices)
            with open(cube, "w") as f:
                f.write('#usda 1.0\n(\n    defaultPrim = "Cube"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Xform "Cube" {\n'
                        f'    def Mesh "geo" {{\n        point3f[] points = [{pts}]\n'
                        f'        int[] faceVertexCounts = [{", ".join(["3"] * len(m.faces))}]\n'
                        f'        int[] faceVertexIndices = [{", ".join(str(int(i)) for i in m.faces.reshape(-1))}]\n    }}\n}}\n')
            with open(layout, "w") as f:
                json.dump({"objects": [{"name": "a", "usd_path": cube, "x": 0.4, "y": 0.0, "z": 0.0},
                                       {"name": "b", "usd_path": cube, "x": 0.4, "y": 0.2, "z": 0.0}]}, f)
            with open(prog, "w") as f:
                json.dump({"statements": [{"op": "on_support", "a": "*", "b": "table"}, {"op": "upright", "a": "*"},
                                          {"op": "left_of", "a": "a", "b": "b", "gap": 0.1},
                                          {"op": "left_of", "a": "b", "b": "a", "gap": 0.1, "soft": 0.3}]}, f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                main(["repair", layout, prog, "--out", out])
            report = json.loads(buf.getvalue())
            self.assertEqual(cm.exception.code, 0)                       # the failed preference does not fail the repair
            self.assertTrue(report["ok"])
            self.assertEqual(report["failed_preferences"], ["left_of(b,a,gap=0.1)"])
            self.assertEqual(report["failed_predicates"], [])


if __name__ == "__main__":
    unittest.main()
