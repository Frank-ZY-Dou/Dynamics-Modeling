"""Tests for the Scenario 3D Collection assets on synthetic GLBs (no network): the fetch turns glTF's
+y up to +z up, scales the mesh to the listed dimension, puts the origin at the centre of the footprint
on the lowest point and checks the sha256; a body gets the manifest's physics, class, handle side and
container tag, and lies down when it cannot stand; layouts name assets and a repair never downloads.
Also the repair's guards these assets exposed: a decimated copy that loses part of a body is not used
(MeshProxy), and the last seat keeps a body on the level of a fixture the repair held it at.
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPO = "test/collection"


def write_asset(root: Path, name: str, mesh: trimesh.Trimesh) -> str:
    """The GLB an asset's fetch would download, already in place; returns its sha256."""
    d = root / REPO / name
    d.mkdir(parents=True, exist_ok=True)
    mesh.export(str(d / "source.glb"))
    return hashlib.sha256((d / "source.glb").read_bytes()).hexdigest()


def manifest(assets: dict) -> dict:
    return {"repo": REPO, "revision": "0" * 40, "license": "CC-BY-4.0", "credit": "Test", "up": "+y", "assets": assets}


def entry(sha: str, **kw) -> dict:
    e = {"id": "test:asset", "path": "assets/x/output.glb", "sha256": sha, "label": "box", "mass_kg": 0.3, "material": "ceramic",
         "static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.2}
    e.update(kw)
    return e


class ScenarioAssets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"SIMREADY_ASSETS_DIR": str(self.root)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def use(self, assets: dict):
        from simready.io import scenario_assets as SA
        return mock.patch.object(SA, "manifest", return_value=manifest(assets))

    def test_fetch_turns_y_up_scales_and_sets_the_origin(self):
        from simready.io import scenario_assets as SA
        box = trimesh.creation.box(extents=(2.0, 3.0, 1.0))                     # glTF frame: 3 units along +y (up)
        box.apply_translation((5.0, 7.0, -1.0))
        sha = write_asset(self.root, "tall", box)
        with self.use({"tall": entry(sha, height_m=0.15)}):
            d = SA.fetch("tall")
            m = trimesh.load(str(d / "mesh.ply"), process=False, force="mesh")
            rec = json.loads((d / SA.RECORD).read_text())
        np.testing.assert_allclose(np.ptp(m.vertices, axis=0), (0.10, 0.05, 0.15), atol=1e-9)   # x, -z, y
        lo, hi = m.vertices.min(0), m.vertices.max(0)
        np.testing.assert_allclose([lo[0] + hi[0], lo[1] + hi[1], lo[2]], 0.0, atol=1e-9)     # footprint centre, lowest point
        self.assertAlmostEqual(rec["scale"], 0.05)
        self.assertIsNone(rec["decimation_deviation"])
        self.assertEqual((rec["revision"], rec["sha256"]), ("0" * 40, sha))

    def test_fetch_refuses_a_file_that_differs_from_the_manifest(self):
        from simready.io import scenario_assets as SA
        from simready.io.nvidia_assets import AssetError
        sha = write_asset(self.root, "tall", trimesh.creation.box(extents=(1, 1, 1)))
        with self.use({"tall": entry("f" * 64, height_m=0.1)}), mock.patch.object(SA, "_get", return_value=(b"other bytes", {})):
            with self.assertRaises(AssetError):
                SA.fetch("tall")
        self.assertNotEqual(sha, "f" * 64)

    def test_body_carries_physics_handle_and_container(self):
        from simready.io import scenario_assets as SA
        sha = write_asset(self.root, "mug", trimesh.creation.box(extents=(1.2, 1.0, 0.9)))
        with self.use({"mug": entry(sha, label="coffee mug", height_m=0.1, handle="+x", container=True)}):
            SA.fetch("mug")
            b = SA.body_from_scenario("cup", "mug", (0.4, 0.1, 0.0), yaw_deg=90.0)
        self.assertEqual(b.meta["physics"]["mass"], 0.3)
        self.assertEqual(b.meta["physics"]["static_friction"], 0.6)
        self.assertEqual((b.meta["label"], b.meta["grasp_axis"]), ("coffee mug", "+x"))
        self.assertIn("container", b.tags)
        lo, hi = b.world_aabb()
        self.assertAlmostEqual(float(lo[2]), 0.0, places=9)                    # the origin is the lowest point
        np.testing.assert_allclose(0.5 * (lo + hi)[:2], (0.4, 0.1), atol=1e-9)
        self.assertEqual(b.meta["scenario"]["credit"], "Test")

    def test_an_asset_that_cannot_stand_lies_down(self):
        from simready.io import scenario_assets as SA
        sha = write_asset(self.root, "plate", trimesh.creation.box(extents=(1.0, 2.0, 0.05)))   # standing on its edge
        with self.use({"plate": entry(sha, length_m=0.2)}):
            SA.fetch("plate")
            b = SA.body_from_scenario("plate", "plate", (0.5, 0.0, 0.0))
        self.assertIn("rest_rotation", b.meta)
        self.assertLess(float(np.ptp(b.verts[:, 2])), 0.01)                     # lying flat: 5 mm thick
        self.assertAlmostEqual(float(b.world_aabb()[0][2]), 0.0, places=9)

    def test_layout_names_assets_and_a_repair_never_downloads(self):
        from simready.cli import load_any
        from simready.io import scenario_assets as SA
        sha = write_asset(self.root, "mug", trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
        layout = self.root / "layout.json"
        layout.write_text(json.dumps({"objects": [{"name": "mug", "scenario": "mug", "x": 0.3, "y": 0.0, "z": 0.0}]}))
        with self.use({"mug": entry(sha, height_m=0.1)}):
            SA.fetch("mug")
            sc = load_any(str(layout))
            self.assertAlmostEqual(sc["mug"].meta["physics"]["mass"], 0.3)
            (self.root / REPO / "mug" / SA.RECORD).unlink()
            with self.assertRaises(FileNotFoundError):
                load_any(str(layout))


class ProxyAndSeat(unittest.TestCase):
    def test_a_copy_that_loses_part_of_the_body_is_not_used(self):
        from simready.scene.model import Body, MeshProxy, Scene
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.05)          # 5120 faces
        b = Body.from_mesh("ball", sphere.vertices, sphere.faces, center=np.zeros(3))
        real = trimesh.Trimesh.simplify_quadric_decimation

        def lossy(self, face_count=None, **kw):
            m = real(self, face_count=face_count, **kw)
            if face_count <= 2000:                                                # the lower half is gone
                keep = m.vertices[m.faces][:, :, 2].min(axis=1) > 0.0
                return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
            return m
        with mock.patch.object(trimesh.Trimesh, "simplify_quadric_decimation", lossy):
            proxy = MeshProxy(Scene([b]), faces=2000)
            with proxy:
                self.assertLess(len(b.faces), 5120)
                self.assertAlmostEqual(float(b.verts[:, 2].min()), -0.05, delta=0.003)
            self.assertEqual(len(proxy.notes), 1)
            self.assertIn("one of 4000", proxy.notes[0])

        def broken(self, face_count=None, **kw):
            m = real(self, face_count=face_count, **kw)
            keep = m.vertices[m.faces][:, :, 2].min(axis=1) > 0.0
            return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
        with mock.patch.object(trimesh.Trimesh, "simplify_quadric_decimation", broken):
            proxy = MeshProxy(Scene([b]), faces=2000)
            with proxy:
                self.assertEqual(len(b.faces), 5120)                              # no copy keeps the body: its own mesh
            self.assertIn("ran on its 5120 faces", proxy.notes[0])

    def test_the_last_seat_keeps_the_level_of_the_fixture(self):
        from simready.scene.model import Body, Scene
        from simready.repair.upright_s4r import RepairSpec, reseat_on_supports
        top = trimesh.creation.box(extents=(0.6, 0.6, 0.02)); top.apply_translation((0.0, 0.0, -0.01))
        shelf = trimesh.creation.box(extents=(0.6, 0.6, 0.02)); shelf.apply_translation((0.0, 0.0, -0.51))
        table = trimesh.util.concatenate([top, shelf])                           # a top at z = 0 and a shelf under it
        t = Body.from_mesh("table", table.vertices, table.faces, fixed=True, tags={"support"})
        box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        b = Body.from_mesh("box", box.vertices, box.faces, center=np.array([0.0, 0.0, -0.03]))   # sank under the top
        spec = RepairSpec(supports={"box": 0.0}, support_of={"box": "table"})
        reseat_on_supports(Scene([t, b]), spec)
        self.assertAlmostEqual(float(b.world_aabb()[0][2]), 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
