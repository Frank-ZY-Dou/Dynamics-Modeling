"""Regression tests for the S4R solver contracts.

Run from the Penetration_Solving directory:

    python -m unittest discover tests

The CPU tests need python-fcl and OSQP (requirements.txt). The GPU tests
in test_gpu_contracts.py need warp-lang and a CUDA device and skip
otherwise.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "s4r"))

from mesh_collision import (  # noqa: E402
    MeshObject, evaluate_mesh_object_scene, evaluate_world_collision_meshes,
)
from s4r_qp import (  # noqa: E402
    PrebuiltFCLOracle, _optimize_rotations_fcl, _optimize_rotations_trimesh,
    qp_status_ok, separate_coincident_centroids, solve_s4r_qp,
)


def box_object(center, extent=0.2, vertex_scale=1.0, rotation=None):
    """Cube of edge ``extent`` whose model vertices are stored at
    ``vertex_scale`` times their size with a compensating normalize_factor,
    so the world geometry is independent of vertex_scale."""
    m = trimesh.creation.box(extents=[extent] * 3)
    v = np.asarray(m.vertices, dtype=np.float64) * vertex_scale
    f = np.asarray(m.faces, dtype=np.int32)
    return MeshObject(name="box", center=np.asarray(center, dtype=np.float64),
                      rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64),
                      collision_verts_model=v, collision_faces=f,
                      visual_verts_model=v.copy(), visual_faces=f.copy(),
                      normalize_factor=1.0 / vertex_scale, inv_mass=1.0)


def mesh_object(mesh, center, rotation=None):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    return MeshObject(name="mesh", center=np.asarray(center, dtype=np.float64),
                      rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64),
                      collision_verts_model=v, collision_faces=f,
                      visual_verts_model=v.copy(), visual_faces=f.copy(),
                      normalize_factor=1.0, inv_mass=1.0)


def solve(objects, **kwargs):
    kw = dict(contact_backend="fcl_prebuilt", verbose=False, revalidate_interval=1)
    kw.update(kwargs)
    return solve_s4r_qp(objects, **kw)


def world_vertices(obj, result, k):
    R = result["final_rotations"][k]
    return obj.normalize_factor * (obj.collision_verts_model @ R.T) + result["final_centers"][k]


class OutcomeContract(unittest.TestCase):
    def setUp(self):
        os.environ["S4R_DISABLE_PENALTY_CLEANUP"] = "1"

    def tearDown(self):
        os.environ.pop("S4R_DISABLE_PENALTY_CLEANUP", None)

    def test_partial_scale_is_reported_not_claimed(self):
        objects = [box_object([0, 0, 0]), box_object([0.1, 0, 0])]
        r = solve(objects, ds_max=0.05, max_steps=1)
        self.assertLess(r["scale"], 1.0 - 1e-6)
        self.assertFalse(r["continuation_complete"])
        self.assertFalse(r["native_converged"])
        self.assertEqual(r["status"], "max_steps")
        # The score is taken at full scale, where the two boxes still overlap.
        self.assertEqual(r["evaluated_at_scale"], 1.0)
        self.assertEqual(r["pen"], 1)

    def test_full_run_converges(self):
        objects = [box_object([0, 0, 0]), box_object([0.1, 0, 0])]
        r = solve(objects, ds_max=0.05)
        self.assertTrue(r["continuation_complete"])
        self.assertTrue(r["native_converged"])
        self.assertEqual(r["status"], "converged")
        self.assertEqual(r["pen"], 0)
        self.assertTrue(r["poses_finite"])

    def test_status_follows_the_evaluator_not_the_tail(self):
        objects = [box_object([0, 0, 0]), box_object([0.1, 0, 0])]
        os.environ["S4R_MAX_TAIL_ITERS"] = "0"
        try:
            r = solve(objects, ds_max=0.05)
        finally:
            os.environ.pop("S4R_MAX_TAIL_ITERS", None)
        self.assertEqual(r["tail_stop_reason"], "tail_disabled")
        self.assertFalse(r["native_converged"])
        if r["pen"] == 0:
            self.assertEqual(r["status"], "converged")
        else:
            self.assertEqual(r["status"], "residual_penetration")

    def test_input_contracts(self):
        objects = [box_object([0, 0, 0]), box_object([0.1, 0, 0])]
        with self.assertRaises(ValueError):
            solve(objects, ds_max=0.0)
        with self.assertRaises(ValueError):
            solve(objects, max_steps=0)
        bad = [box_object([0, 0, 0]), box_object([float("nan"), 0, 0])]
        with self.assertRaises(ValueError):
            solve(bad)
        with self.assertRaises(ValueError):
            solve(objects, use_dual=True, box_bounds=(np.zeros(3), np.ones(3)))
        with self.assertRaises(ValueError):
            solve(objects, use_dual=True, enable_rotation=True)
        with self.assertRaises(ValueError):
            solve(objects, joints=[(0, 1, np.zeros(3), np.zeros(3))])

    def test_empty_and_single_body(self):
        r = solve([])
        self.assertEqual(r["status"], "converged")
        self.assertEqual(r["pen"], 0)
        r = solve([box_object([0.3, 0.2, 0.1])])
        self.assertEqual(r["status"], "converged")
        np.testing.assert_allclose(r["final_centers"][0], [0.3, 0.2, 0.1])


class HardConstraints(unittest.TestCase):
    def setUp(self):
        os.environ["S4R_DISABLE_PENALTY_CLEANUP"] = "1"

    def tearDown(self):
        os.environ.pop("S4R_DISABLE_PENALTY_CLEANUP", None)

    def test_single_body_wall_without_pair_contacts(self):
        objects = [box_object([0.95, 0.5, 0.5])]
        r = solve(objects, box_bounds=(np.zeros(3), np.ones(3)))
        v = world_vertices(objects[0], r, 0)
        self.assertTrue(np.all(v >= -1e-7) and np.all(v <= 1 + 1e-7),
                        "pair-free continuation must still enforce the container walls")
        self.assertEqual(r["status"], "converged")
        self.assertLessEqual(r["wall_violation"], 1e-7)

    def test_walls_hold_with_pair_contacts(self):
        objects = [box_object([0.15, 0.5, 0.5]), box_object([0.17, 0.5, 0.5]), box_object([0.9, 0.5, 0.5])]
        r = solve(objects, box_bounds=(np.zeros(3), np.ones(3)))
        for k, o in enumerate(objects):
            v = world_vertices(o, r, k)
            self.assertTrue(np.all(v >= -1e-7) and np.all(v <= 1 + 1e-7))
        self.assertEqual(r["pen"], 0)
        self.assertEqual(r["status"], "converged")

    def test_infeasible_container_is_reported(self):
        # Two 0.2 cubes cannot both fit inside a 0.3 box: the contact rows
        # and the wall rows become jointly infeasible before full scale.
        objects = [box_object([0.1, 0.15, 0.15]), box_object([0.2, 0.15, 0.15])]
        r = solve(objects, box_bounds=(np.zeros(3), np.full(3, 0.3)))
        self.assertFalse(r["continuation_complete"])
        self.assertIn(r["status"], ("qp_failure", "container_infeasible"))
        self.assertFalse(r["native_converged"])
        # Scored at full scale: the boxes still overlap and the wall
        # violation of the delivered (full-size) poses is reported.
        self.assertEqual(r["pen"], 1)
        self.assertGreater(r["wall_violation"], 0.0)

    def test_single_body_infeasible_container(self):
        # A 0.2 cube cannot fit in a 0.1 box: reported, never relaxed.
        r = solve([box_object([0.05, 0.05, 0.05])], box_bounds=(np.zeros(3), np.full(3, 0.1)))
        self.assertTrue(r["container_infeasible"])
        self.assertEqual(r["status"], "container_infeasible")

    def test_joint_anchors_hold(self):
        # Two links joined at a point between them (a 0.1 gap keeps the
        # links themselves apart), both overlapping a third box.
        a = box_object([0.0, 0.0, 0.0], extent=0.2)
        b = box_object([0.3, 0.0, 0.0], extent=0.2)
        c = box_object([0.15, 0.12, 0.0], extent=0.2)
        joints = [(0, 1, np.array([0.15, 0.0, 0.0]), np.array([-0.15, 0.0, 0.0]))]
        r = solve([a, b, c], enable_rotation=True, joints=joints, ds_max=0.05)
        self.assertTrue(r["continuation_complete"])
        self.assertLessEqual(r["joint_residual"], 1e-6)
        self.assertNotEqual(r["status"], "joint_violation")


class OracleContracts(unittest.TestCase):
    def contacts(self, objects, s, ds):
        nfs = [o.normalize_factor for o in objects]
        mv = [o.collision_verts_model for o in objects]
        mf = [o.collision_faces for o in objects]
        oracle = PrebuiltFCLOracle(nfs, mv, mf, d_hat=0.02)
        centers = np.array([o.center for o in objects])
        rots = [o.rotation for o in objects]
        return oracle.find_contacts(s, ds, centers, rots)

    def test_margin_covers_next_scale_overlap(self):
        # Half-edge 0.1 cubes 0.105 apart: at s=0.5 the gap is 0.005 and the
        # next step overlaps by 0.005, so the pair must be a candidate.
        objects = [box_object([0, 0, 0]), box_object([0.105, 0, 0])]
        c = self.contacts(objects, 0.5, 0.05)
        self.assertEqual([(x[0], x[1]) for x in c], [(0, 1)])
        self.assertAlmostEqual(c[0][2], 0.005, places=6)

    def test_broadphase_is_representation_invariant(self):
        base = [box_object([0, 0, 0]), box_object([0.105, 0, 0]), box_object([0.0, 0.31, 0])]
        scaled = [box_object([0, 0, 0], vertex_scale=10.0),
                  box_object([0.105, 0, 0], vertex_scale=10.0),
                  box_object([0.0, 0.31, 0], vertex_scale=10.0)]
        for s, ds in ((0.5, 0.05), (0.9, 0.05), (1.0, 0.0)):
            c1 = self.contacts(base, s, ds)
            c2 = self.contacts(scaled, s, ds)
            self.assertEqual([(x[0], x[1]) for x in c1], [(x[0], x[1]) for x in c2])
            for a, b in zip(c1, c2):
                self.assertAlmostEqual(a[2], b[2], places=9)
                np.testing.assert_allclose(a[3], b[3], atol=1e-9)

    def test_nested_body_is_a_penetration(self):
        big = mesh_object(trimesh.creation.box(extents=[1, 1, 1]), [0, 0, 0])
        small = mesh_object(trimesh.creation.box(extents=[0.2] * 3), [0.1, 0.05, 0.0])
        c = self.contacts([big, small], 1.0, 0.0)
        self.assertEqual(len(c), 1)
        self.assertLess(c[0][2], 0.0)
        stats = evaluate_mesh_object_scene([big, small])
        self.assertEqual(stats.pen_pairs, 1)
        self.assertGreater(stats.max_penetration, 0.3)

    def test_piece_of_a_body_inside_another_body_is_a_penetration(self):
        # A body made of two closed pieces one unit apart, placed so that one
        # piece sits inside the big cube (0.1 from its +x face) and the other
        # far outside. The bodies' AABBs do not nest and the surfaces are
        # apart, so only a piece-wise containment check can see it.
        big = mesh_object(trimesh.creation.box(extents=[1, 1, 1]), [0, 0, 0])
        two = trimesh.util.concatenate([
            trimesh.creation.box(extents=[0.2] * 3).apply_translation([-1.0, 0, 0]),
            trimesh.creation.box(extents=[0.2] * 3).apply_translation([1.0, 0, 0])])
        pieces = mesh_object(two, [1.3, 0, 0])
        c = self.contacts([big, pieces], 1.0, 0.0)
        self.assertEqual([(x[0], x[1]) for x in c], [(0, 1)])
        self.assertAlmostEqual(c[0][2], -0.3, places=6)   # gap 0.1 to the +x face plus the piece's width 0.2
        self.assertGreater(c[0][3][0], 0.99)              # exit through the nearest face: +x
        stats = evaluate_mesh_object_scene([big, pieces])
        self.assertEqual(stats.pen_pairs, 1)
        self.assertAlmostEqual(stats.max_penetration, 0.3, places=6)
        # the far piece alone is not a contact
        outside = mesh_object(two, [2.0, 0, 0])
        self.assertEqual(self.contacts([big, outside], 1.0, 0.0), [])
        self.assertEqual(evaluate_mesh_object_scene([big, outside]).pen_pairs, 0)
        res = solve([big, pieces], d_hat=0.02, ds_max=0.05, max_steps=200, adaptive_ds=True)
        self.assertEqual(res["status"], "converged")

    def test_cavity_normal_points_out_of_the_wall(self):
        ring = trimesh.creation.annulus(r_min=1.0, r_max=2.0, height=0.4, sections=32)
        # Separated box inside the hole: the witness direction for the box
        # is toward the hole centre (-x), opposite to the centroid line.
        objects = [mesh_object(ring, [0, 0, 0]),
                   mesh_object(trimesh.creation.box(extents=[0.1] * 3), [0.85, 0, 0])]
        nfs = [1.0, 1.0]
        oracle = PrebuiltFCLOracle(nfs, [o.collision_verts_model for o in objects],
                                   [o.collision_faces for o in objects], d_hat=0.2)
        c = oracle.find_contacts(1.0, 0.0, np.array([o.center for o in objects]), [np.eye(3)] * 2)
        self.assertEqual(len(c), 1)
        self.assertLess(c[0][3][0], -0.9)
        # Penetrating box through the inner wall: same direction.
        objects[1] = mesh_object(trimesh.creation.box(extents=[0.4, 0.4, 0.2]), [0.85, 0, 0])
        oracle = PrebuiltFCLOracle(nfs, [o.collision_verts_model for o in objects],
                                   [o.collision_faces for o in objects], d_hat=0.02)
        c = oracle.find_contacts(1.0, 0.0, np.array([o.center for o in objects]), [np.eye(3)] * 2)
        self.assertEqual(len(c), 1)
        self.assertLess(c[0][2], 0.0)
        self.assertLess(c[0][3][0], -0.9)
        # The evaluator does not count the separated box in the hole.
        objects[1] = mesh_object(trimesh.creation.box(extents=[0.1] * 3), [0.85, 0, 0])
        self.assertEqual(evaluate_mesh_object_scene(objects).pen_pairs, 0)

    def test_incremental_detection_matches_full(self):
        rng = np.random.default_rng(7)
        objects = [box_object(rng.uniform(-0.2, 0.2, 3), extent=float(rng.uniform(0.08, 0.2)),
                              rotation=trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3])
                   for _ in range(25)]
        nfs = [o.normalize_factor for o in objects]
        mv = [o.collision_verts_model for o in objects]
        mf = [o.collision_faces for o in objects]
        inc = PrebuiltFCLOracle(nfs, mv, mf, d_hat=0.02)
        full = PrebuiltFCLOracle(nfs, mv, mf, d_hat=0.02)
        centers = np.array([o.center for o in objects])
        rots = [o.rotation for o in objects]

        def same(a, b):
            self.assertEqual([(x[0], x[1]) for x in a], [(x[0], x[1]) for x in b])
            for x, y in zip(a, b):
                self.assertEqual(x[2], y[2])
                np.testing.assert_array_equal(x[3], y[3])

        same(inc.find_contacts(1.0, 0.0, centers, rots, incremental=True),
             full.find_contacts(1.0, 0.0, centers, rots))
        for _ in range(6):
            moved = rng.random(25) < 0.3
            centers[moved] += rng.uniform(-0.02, 0.02, (int(moved.sum()), 3))
            same(inc.find_contacts(1.0, 0.0, centers, rots, incremental=True),
                 full.find_contacts(1.0, 0.0, centers, rots))
        # A rotation change invalidates the cache.
        rots[3] = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        same(inc.find_contacts(1.0, 0.0, centers, rots, incremental=True),
             full.find_contacts(1.0, 0.0, centers, rots))

    def test_inward_winding_is_normalised(self):
        m = trimesh.creation.box(extents=[0.2] * 3)
        flipped = trimesh.Trimesh(vertices=m.vertices, faces=np.asarray(m.faces)[:, ::-1], process=False)
        a = [box_object([0, 0, 0]), box_object([0.15, 0.02, 0.0])]
        b = [mesh_object(flipped, [0, 0, 0]), mesh_object(flipped, [0.15, 0.02, 0.0])]
        ca = self.contacts(a, 1.0, 0.0)
        cb = self.contacts(b, 1.0, 0.0)
        self.assertEqual(len(ca), 1)
        self.assertEqual(len(cb), 1)
        self.assertAlmostEqual(ca[0][2], cb[0][2], places=9)
        np.testing.assert_allclose(ca[0][3], cb[0][3], atol=1e-9)


class StartingScale(unittest.TestCase):
    def test_audit_rows_describe_the_current_state(self):
        # with a long refresh interval, the audit counts on cached steps must be those of
        # the state the step starts from, not of the last refresh
        objects = [box_object([0, 0, 0]), box_object([0.12, 0, 0]), box_object([0.24, 0.02, 0]), box_object([0.05, 0.16, 0])]
        res = solve(objects, revalidate_interval=2, audit=True, adaptive_ds=False, ds_max=0.05)
        rows = res["audit_log"]
        self.assertGreater(len(rows), 2)
        for a, b in zip(rows, rows[1:]):
            if abs(a["scale_after"] - b["scale_before"]) < 1e-12:
                self.assertEqual(a["evaluator_pen_after"], b["evaluator_pen_before"])

    def test_attraction_weights_are_per_body(self):
        # two separated cubes, a target 0.1 to the +x of each, weight 0 for the first: it stays
        objects = [box_object([0, 0, 0]), box_object([1.0, 0, 0])]
        targets = np.array([[0.1, 0, 0], [1.1, 0, 0]])
        res = solve(objects, target_centers=targets, attraction_alpha=np.array([0.0, 1.0]), max_steps=1, ds_max=0.05)
        moved = res["final_centers"] - np.array([o.center for o in objects])
        self.assertAlmostEqual(float(np.linalg.norm(moved[0])), 0.0, places=9)
        self.assertGreater(float(moved[1][0]), 0.0)

    def test_three_coincident_centroids(self):
        centers = np.zeros((3, 3))
        radii = np.full(3, math.sqrt(3) * 0.1)
        passes, remaining = separate_coincident_centroids(centers, radii, 0.02, 0.01)
        self.assertEqual(remaining, 0)
        target = 0.02 + 0.01 * (radii[0] + radii[1]) + 1e-6
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertGreaterEqual(np.linalg.norm(centers[i] - centers[j]), target)

    def test_large_coincident_clusters_converge(self):
        radii_r = math.sqrt(3) * 0.1
        for n in (3, 6, 10, 12, 15, 26, 40):
            centers = np.zeros((n, 3))
            passes, remaining = separate_coincident_centroids(centers, np.full(n, radii_r), 0.02, 0.01)
            self.assertEqual(remaining, 0, f"{n} coincident bodies not separated")
            self.assertLess(passes, 100)
            target = 0.02 + 0.01 * 2 * radii_r + 1e-6
            d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            d[np.arange(n), np.arange(n)] = np.inf
            self.assertGreaterEqual(d.min(), target)

    def test_solver_handles_coincident_boxes(self):
        os.environ["S4R_DISABLE_PENALTY_CLEANUP"] = "1"
        try:
            objects = [box_object([0, 0, 0]), box_object([0, 0, 0]), box_object([0, 0, 0])]
            r = solve(objects)
        finally:
            os.environ.pop("S4R_DISABLE_PENALTY_CLEANUP", None)
        self.assertTrue(r["continuation_complete"])
        self.assertEqual(r["pen"], 0)


class OSQPStatus(unittest.TestCase):
    def test_status_text(self):
        def res(text):
            return SimpleNamespace(info=SimpleNamespace(status=text))
        self.assertTrue(qp_status_ok(res("solved")))
        self.assertTrue(qp_status_ok(res("solved inaccurate")))
        self.assertTrue(qp_status_ok(res("solved_inaccurate")))
        self.assertFalse(qp_status_ok(res("maximum iterations reached")))
        self.assertFalse(qp_status_ok(res("primal infeasible")))
        self.assertFalse(qp_status_ok(res("primal infeasible inaccurate")))
        self.assertFalse(qp_status_ok(res("dual infeasible")))


class RotationHelper(unittest.TestCase):
    def scene(self):
        a = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
        b = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        centers = np.array([[0.0, 0.0, 0.0], [1.14, 0.35, 0.0]])
        rotations = [np.eye(3), np.eye(3)]
        verts = [np.asarray(m.vertices).copy() for m in (a, b)]
        faces = [np.asarray(m.faces).copy() for m in (a, b)]
        return centers, rotations, verts, faces

    @staticmethod
    def gap(centers, rotations, verts, faces):
        x, y = [trimesh.Trimesh(vertices=verts[i] @ rotations[i].T + centers[i],
                                faces=faces[i], process=False) for i in range(2)]
        return min(trimesh.proximity.closest_point_naive(x, y.vertices)[1].min(),
                   trimesh.proximity.closest_point_naive(y, x.vertices)[1].min())

    def test_trimesh_helper_opens_the_gap(self):
        centers, rotations, verts, faces = self.scene()
        before = self.gap(centers, rotations, verts, faces)
        moved = _optimize_rotations_trimesh(centers, rotations, [1.0, 1.0], verts, faces, 2, 0.1, 1.0, 1)
        self.assertGreater(moved, 0.0)
        self.assertGreater(self.gap(centers, rotations, verts, faces), before)

    def test_fcl_helper_opens_the_gap(self):
        centers, rotations, verts, faces = self.scene()
        before = self.gap(centers, rotations, verts, faces)
        moved = _optimize_rotations_fcl(centers, rotations, [1.0, 1.0], verts, faces, 2, 0.1, 1.0, 1)
        self.assertGreater(moved, 0.0)
        self.assertGreater(self.gap(centers, rotations, verts, faces), before)


class Evaluator(unittest.TestCase):
    def test_touching_outside_is_not_penetration(self):
        big = trimesh.creation.box(extents=[1, 1, 1])
        apart = trimesh.creation.box(extents=[0.2] * 3)
        apart.apply_translation([0.7, 0, 0])
        self.assertEqual(evaluate_world_collision_meshes([big, apart]).pen_pairs, 0)

    def test_crossing_bars(self):
        a = trimesh.creation.box(extents=[2, 0.02, 0.02])
        b = trimesh.creation.box(extents=[0.02, 2, 0.03])
        b.apply_translation([0.137, 0.113, 0.002])
        self.assertEqual(evaluate_world_collision_meshes([a, b]).pen_pairs, 1)


if __name__ == "__main__":
    unittest.main()
