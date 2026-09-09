"""GPU regression tests (Warp oracles, dual APGD, GPU driver).

Skipped when warp-lang or a CUDA device is unavailable. Run from the
Penetration_Solving directory:

    python -m unittest tests.test_gpu_contracts -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "s4r"))
sys.path.insert(0, str(ROOT / "s4r_gpu"))

try:
    import warp as wp  # noqa: F401
    wp.init()
    _HAS_CUDA = wp.get_cuda_device_count() > 0
except Exception:  # noqa: BLE001
    _HAS_CUDA = False

if _HAS_CUDA:
    from warp_pair_contact_v3 import S4RWarpContactOracleV3
    from oracle_v4 import S4RWarpContactOracleV4
    from dual_apgd import DualAPGD
    from s4r_gpu_native import solve_s4r_gpu

from mesh_collision import MeshObject  # noqa: E402
from s4r_qp import PrebuiltFCLOracle  # noqa: E402


class _Shim:
    def __init__(self, nf, v, f):
        self.normalize_factor = nf
        self.collision_verts_model = np.asarray(v, dtype=np.float64)
        self.collision_faces = np.asarray(f, dtype=np.int32)


def shim(mesh, nf=1.0):
    return _Shim(nf, mesh.vertices, mesh.faces)


def mesh_object(mesh, center):
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    return MeshObject(name="mesh", center=np.asarray(center, dtype=np.float64), rotation=np.eye(3),
                      collision_verts_model=v, collision_faces=f, visual_verts_model=v.copy(),
                      visual_faces=f.copy(), normalize_factor=1.0, inv_mass=1.0)


@unittest.skipUnless(_HAS_CUDA, "needs warp-lang and a CUDA device")
class WarpOracle(unittest.TestCase):
    def test_crossing_bars_are_detected(self):
        # Two thin bars that cross without any vertex or edge sample of one
        # lying inside the other: the crossing pass must still report them.
        for dz, extents_b in ((0.0, [0.02, 2, 0.02]), (0.002, [0.02, 2, 0.03])):
            a = trimesh.creation.box(extents=[2, 0.02, 0.02])
            b = trimesh.creation.box(extents=extents_b)
            oracle = S4RWarpContactOracleV3([shim(a), shim(b)], d_hat=0.02)
            C = np.array([[0, 0, 0], [0.137, 0.113, dz]], dtype=np.float64)
            c = oracle.find_contacts(1.0, 0.0, C, [np.eye(3), np.eye(3)])
            self.assertEqual(len(c), 1)
            self.assertLess(c[0][2], 0.0)

    def test_cavity_normal(self):
        ring = trimesh.creation.annulus(r_min=1.0, r_max=2.0, height=0.4, sections=32)
        box = trimesh.creation.box(extents=[0.1] * 3)
        oracle = S4RWarpContactOracleV3([shim(ring), shim(box)], d_hat=0.2)
        C = np.array([[0, 0, 0], [0.85, 0, 0]], dtype=np.float64)
        c = oracle.find_contacts(1.0, 0.0, C, [np.eye(3), np.eye(3)])
        self.assertEqual(len(c), 1)
        self.assertGreater(c[0][2], 0.0)
        self.assertLess(c[0][3][0], -0.9, "witness direction must point into the hole, not along the centroid line")
        box2 = trimesh.creation.box(extents=[0.4, 0.4, 0.2])
        oracle = S4RWarpContactOracleV3([shim(ring), shim(box2)], d_hat=0.02)
        c = oracle.find_contacts(1.0, 0.0, C, [np.eye(3), np.eye(3)])
        self.assertEqual(len(c), 1)
        self.assertLess(c[0][2], 0.0)
        self.assertLess(c[0][3][0], -0.9)

    def test_nested_body(self):
        big = trimesh.creation.box(extents=[1, 1, 1])
        small = trimesh.creation.box(extents=[0.2] * 3)
        oracle = S4RWarpContactOracleV3([shim(big), shim(small)], d_hat=0.02)
        C = np.array([[0, 0, 0], [0.1, 0.05, 0.0]], dtype=np.float64)
        c = oracle.find_contacts(1.0, 0.0, C, [np.eye(3), np.eye(3)])
        self.assertEqual(len(c), 1)
        self.assertLess(c[0][2], -0.3)

    def test_v3_v4_and_fcl_agree_on_candidates(self):
        rng = np.random.default_rng(3)
        meshes = [trimesh.creation.box(extents=rng.uniform(0.08, 0.2, 3)) for _ in range(12)]
        shims = [shim(m) for m in meshes]
        C = rng.uniform(-0.15, 0.15, (12, 3))
        R = [np.eye(3)] * 12
        o3 = S4RWarpContactOracleV3(shims, d_hat=0.02)
        o4 = S4RWarpContactOracleV4(shims, d_hat=0.02)
        fcl = PrebuiltFCLOracle([1.0] * 12, [np.asarray(m.vertices) for m in meshes],
                                [np.asarray(m.faces, np.int32) for m in meshes], 0.02)
        for s, ds in ((0.5, 0.05), (1.0, 0.0)):
            c3 = o3.find_contacts(s, ds, C, R)
            c4 = o4.find_contacts(s, ds, C, R)
            cf = fcl.find_contacts(s, ds, C, R)
            k3 = [(c[0], c[1]) for c in c3]
            self.assertEqual(k3, [(c[0], c[1]) for c in c4])
            for a, b in zip(c3, c4):
                self.assertAlmostEqual(a[2], b[2], places=6)
                np.testing.assert_allclose(a[3], b[3], atol=1e-6)
            self.assertEqual(set(k3), set((c[0], c[1]) for c in cf))
            pen3 = {(c[0], c[1]) for c in c3 if c[2] < 0}
            penf = {(c[0], c[1]) for c in cf if c[2] < 0}
            self.assertEqual(pen3, penf)

    def test_representation_invariance(self):
        a = trimesh.creation.box(extents=[0.2] * 3)
        base = [shim(a), shim(a)]
        scaled = [_Shim(0.1, np.asarray(a.vertices) * 10.0, a.faces), _Shim(0.1, np.asarray(a.vertices) * 10.0, a.faces)]
        C = np.array([[0, 0, 0], [0.105, 0, 0]], dtype=np.float64)
        c1 = S4RWarpContactOracleV3(base, d_hat=0.02).find_contacts(0.5, 0.05, C, [np.eye(3)] * 2)
        c2 = S4RWarpContactOracleV3(scaled, d_hat=0.02).find_contacts(0.5, 0.05, C, [np.eye(3)] * 2)
        self.assertEqual(len(c1), 1)
        self.assertEqual(len(c2), 1)
        self.assertAlmostEqual(c1[0][2], c2[0][2], places=5)


@unittest.skipUnless(_HAS_CUDA, "needs warp-lang and a CUDA device")
class Lipschitz(unittest.TestCase):
    def test_certified_bound_dominates_spectral_norm(self):
        rng = np.random.default_rng(0)
        apgd = DualAPGD(capacity_K=256, capacity_N=64)
        for _ in range(50):
            N = int(rng.integers(4, 20))
            K = int(rng.integers(3, 40))
            ci = rng.integers(0, N, K).astype(np.int32)
            cj = ((ci + rng.integers(1, N, K)) % N).astype(np.int32)
            n = rng.normal(size=(K, 3))
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            n = n.astype(np.float32)
            A = np.zeros((K, 3 * N))
            for k in range(K):
                A[k, 3 * ci[k]:3 * ci[k] + 3] = -n[k].astype(np.float64)
                A[k, 3 * cj[k]:3 * cj[k] + 3] = n[k].astype(np.float64)
            lam_max = float(np.linalg.eigvalsh(A @ A.T)[-1])
            L = apgd._certified_L(ci, cj, n, K, N)
            self.assertGreaterEqual(L * (1 + 1e-9), lam_max)
            self.assertLessEqual(L, apgd._gershgorin_L(ci, cj, K, N) + 1e-9)


@unittest.skipUnless(_HAS_CUDA, "needs warp-lang and a CUDA device")
class Driver(unittest.TestCase):
    def test_partial_scale_is_reported(self):
        objects = [mesh_object(trimesh.creation.box(extents=[0.2] * 3), [0, 0, 0]),
                   mesh_object(trimesh.creation.box(extents=[0.2] * 3), [0.1, 0, 0])]
        r = solve_s4r_gpu(objects, ds_max=0.05, max_steps=1, verbose=False)
        self.assertFalse(r["continuation_complete"])
        self.assertEqual(r["status"], "max_steps")
        self.assertFalse(r["native_converged"])
        self.assertEqual(r["pen"], 1)

    def test_two_boxes_converge_with_exact_check(self):
        objects = [mesh_object(trimesh.creation.box(extents=[0.2] * 3), [0, 0, 0]),
                   mesh_object(trimesh.creation.box(extents=[0.2] * 3), [0.1, 0, 0])]
        r = solve_s4r_gpu(objects, ds_max=0.05, verbose=False)
        self.assertTrue(r["continuation_complete"])
        self.assertEqual(r["status"], "converged")
        self.assertTrue(r["fcl_verified"])
        self.assertEqual(r["pen"], 0)
        self.assertTrue(np.all(np.isfinite(r["final_centers"])))

    def test_coincident_centroids(self):
        objects = [mesh_object(trimesh.creation.box(extents=[0.2] * 3), [0, 0, 0]) for _ in range(3)]
        r = solve_s4r_gpu(objects, ds_max=0.05, verbose=False)
        self.assertTrue(r["continuation_complete"])
        self.assertEqual(r["pen"], 0)


if __name__ == "__main__":
    unittest.main()
