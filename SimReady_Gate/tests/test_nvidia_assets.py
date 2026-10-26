"""Tests for the SimReady package reader and fetcher on synthetic packages (no network): physics,
semantics and the grasp hint read from the UsdPhysics schemas, the resting frame of an asset
authored on its edge, the bill-of-materials check, layouts that name packages, and the settle test
using the asset's friction and mass.
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


def write_package(root: Path, name: str, extents, com=None, grasp=None, mass=0.5, friction=0.4, label="mug"):
    """A minimal SimReady package: root USD with a mesh, MassAPI, a physics material, semantic
    labels and a grasp prim, plus the packaging JSON, the bill of materials and root_usds.json."""
    pkg = root / "nvidia" / "simready-assets" / name
    (pkg / ".metadata").mkdir(parents=True, exist_ok=True)
    (pkg / "simready_usd").mkdir(parents=True, exist_ok=True)
    m = trimesh.creation.box(extents=extents)
    v = np.asarray(m.vertices) + np.array([0.0, 0.0, extents[2] / 2])        # origin at the bottom, as SimReady props
    pts = ", ".join(f"({x:.5f}, {y:.5f}, {z:.5f})" for x, y, z in v)
    idx = ", ".join(str(int(i)) for i in m.faces.reshape(-1))
    com = com if com is not None else (0.0, 0.0, extents[2] / 2)
    g = ""
    if grasp is not None:
        g = (f'    def Scope "Grasp" {{\n        def Xform "grasp_identifier" {{\n            custom float3 intersection_location = ({grasp[0]}, {grasp[1]}, {grasp[2]})\n'
             f'            custom float3 intersection_normal = (-1, 0, 0)\n            uniform token purpose = "guide"\n        }}\n    }}\n')
    text = f'''#usda 1.0
(
    customLayerData = {{
        string asset_name = "{name}"
        string asset_type = "prop"
        string profile = "Prop-Robotics-Neutral"
    }}
    defaultPrim = "RootNode"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RootNode" (
    prepend apiSchemas = ["SemanticsLabelsAPI:wikidata_class"]
)
{{
    custom string omni:simready:documentation = "a test asset"
    token[] semantics:labels:wikidata_class = ["{label}"]
    def Scope "Geometry" {{
        def Xform "body_01_obj_00" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {{
            rel material:binding:physics = </RootNode/PhysicsMaterials/mat>
            def Mesh "body_01_mesh_00" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMassAPI"]
            )
            {{
                point3f[] points = [{pts}]
                int[] faceVertexCounts = [{", ".join(["3"] * len(m.faces))}]
                int[] faceVertexIndices = [{idx}]
                float physics:mass = {mass}
                point3f physics:centerOfMass = ({com[0]}, {com[1]}, {com[2]})
            }}
        }}
    }}
    def Scope "PhysicsMaterials" {{
        def Material "mat" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:density = 1000
            float physics:staticFriction = {friction}
            float physics:dynamicFriction = {friction * 0.8}
            float physics:restitution = 0.1
        }}
    }}
{g}}}
'''
    usd = pkg / "simready_usd" / f"sm_{name}_01.usda"
    usd.write_text(text)
    bom = {"format_version": "1.0", "items": [{"relative_path": f"simready_usd/sm_{name}_01.usda", "size": usd.stat().st_size,
                                               "hash": {"sha256": hashlib.sha256(usd.read_bytes()).hexdigest()}}]}
    (pkg / ".metadata" / "com.nvidia.simready.packaging.bom.json").write_text(json.dumps(bom))
    (pkg / ".metadata" / "com.nvidia.simready.root_usds.json").write_text(json.dumps({"entries": [f"simready_usd/sm_{name}_01.usda"]}))
    (pkg / "com.nvidia.simready.packaging.json").write_text(json.dumps({"package_id": f"test.{name}", "license": "CC-BY-4.0"}))
    (pkg / ".simready_fetch.json").write_text(json.dumps({"repo": "nvidia/simready-assets", "package": name, "revision": "0" * 40,
                                                          "files": {}, "bom_checked": True, "license": "CC-BY-4.0"}))
    return pkg


class Packages(unittest.TestCase):
    def setUp(self):
        try:
            import pxr  # noqa: F401
        except ImportError:
            self.skipTest("usd-core not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"SIMREADY_ASSETS_DIR": str(self.root)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_read_package_physics_semantics_grasp(self):
        from simready.io.nvidia_assets import read_package, body_from_package
        pkg = write_package(self.root, "mug_a01", (0.1, 0.08, 0.1), com=(-0.01, 0.0, 0.04), grasp=(-0.05, 0.0, 0.05))
        info = read_package(pkg)
        self.assertAlmostEqual(info["mass"], 0.5, places=5)
        self.assertAlmostEqual(info["static_friction"], 0.4, places=5)
        self.assertEqual(info["labels"]["wikidata_class"], ["mug"])
        self.assertEqual(info["rigid_bodies"], 1)
        self.assertEqual(info["license"], "CC-BY-4.0")
        b = body_from_package("cup", pkg, (0.5, 0.1, 0.0), yaw_deg=90.0)
        self.assertEqual(b.meta["label"], "mug")
        self.assertEqual(b.meta["grasp_axis"], "-x")
        self.assertTrue(np.allclose(b.meta["com"], [-0.01, 0.0, -0.01], atol=1e-6))     # relative to the box centre
        self.assertIn("container", b.tags)
        self.assertTrue(np.allclose(b.world_aabb()[0][2], 0.0, atol=1e-6))             # the origin is the asset's bottom
        self.assertTrue(np.allclose(b.center[:2], [0.5, 0.1], atol=1e-6))

    def test_root_usd_stays_in_the_package(self):
        from simready.io.nvidia_assets import root_usd
        pkg = write_package(self.root, "mug_b01", (0.1, 0.08, 0.1))
        outside = self.root / "outside.usda"; outside.write_text("#usda 1.0\n")
        (pkg / ".metadata" / "com.nvidia.simready.root_usds.json").write_text(json.dumps({"entries": ["../../../outside.usda"]}))
        self.assertEqual(root_usd(pkg).name, "sm_mug_b01_01.usda")       # the listed entry escapes: the package's own root

    def test_edge_standing_asset_rests_flat(self):
        from simready.io.nvidia_assets import body_from_package, tipping_deg
        pkg = write_package(self.root, "board_a01", (0.015, 0.22, 0.38), label="cutting board")
        b = body_from_package("board", pkg, (0.5, 0.0, 0.0))
        ext = np.ptp(b.world_vertices(), axis=0)
        self.assertLess(ext[2], 0.02)                                   # lying on its face
        self.assertIn("rest_rotation", b.meta)
        self.assertGreater(tipping_deg(b.verts, b.meta.get("com")), 12.0)
        pkg2 = write_package(self.root, "box_a01", (0.2, 0.2, 0.1), label="box")
        self.assertNotIn("rest_rotation", body_from_package("box", pkg2, (0.5, 0.0, 0.0)).meta)   # a stable pose is kept

    def test_fetch_checks_the_bill_of_materials(self):
        from simready.io import nvidia_assets as NA
        good = b"#usda 1.0\n"
        files = [{"path": "pkg_a01/simready_usd/sm_pkg_a01_01.usda", "size": len(good), "sha256": None},
                 {"path": "pkg_a01/.metadata/com.nvidia.simready.packaging.bom.json", "size": 1, "sha256": None},
                 {"path": "pkg_a01/simready_usd/Textures/t_alb.png", "size": 1, "sha256": None}]
        bom = json.dumps({"items": [{"relative_path": "simready_usd/sm_pkg_a01_01.usda", "hash": {"sha256": hashlib.sha256(good).hexdigest()}}]}).encode()
        served = {"sm_pkg_a01_01.usda": good, "com.nvidia.simready.packaging.bom.json": bom}
        requested = []

        def fake_get(url, **kw):
            requested.append(url)
            return served[url.rsplit("/", 1)[1]], {}
        with mock.patch.object(NA, "resolve_revision", lambda repo, rev: "a" * 40), \
                mock.patch.object(NA, "list_files", lambda *a: files), mock.patch.object(NA, "_get", fake_get):
            d = NA.fetch("pkg_a01")
            rec = json.loads((d / NA.RECORD).read_text())
            self.assertEqual(rec["revision"], "a" * 40)
            self.assertTrue(rec["bom_checked"])
            self.assertFalse(any("Textures" in u for u in requested))     # textures are not downloaded by default
            n = len(requested)
            NA.fetch("pkg_a01")
            self.assertEqual(len(requested), n)                          # the same commit is not fetched twice
            served["sm_pkg_a01_01.usda"] = b"#usda 1.0\n# tampered\n"
            with self.assertRaises(NA.AssetError):
                NA.fetch("pkg_a01", force=True)
            served["sm_pkg_a01_01.usda"] = good                          # a bill of materials the package does not vouch for
            files.append({"path": "pkg_a01/com.nvidia.simready.packaging.json", "size": 1, "sha256": None})
            served["com.nvidia.simready.packaging.json"] = json.dumps({"metadata": [{"name": "com.nvidia.simready.packaging.bom.json",
                                                                                      "hash": {"sha256": "0" * 64}}]}).encode()
            with self.assertRaises(NA.AssetError):
                NA.fetch("pkg_a01", force=True)
        with self.assertRaises(NA.AssetError):
            NA.fetch("../escape")

    def test_layout_with_packages_and_settle_physics(self):
        from simready.cli import load_any
        from simready.io.paths import portable
        write_package(self.root, "mug_a01", (0.08, 0.08, 0.1), friction=0.3, mass=0.3)
        write_package(self.root, "slab_table_a01", (0.7, 1.0, 0.05), label="table", mass=20.0)
        layout = self.root / "layout.json"
        layout.write_text(json.dumps({"table": {"package": "slab_table_a01", "position": [0.55, 0.0, -0.05]},
                                      "objects": [{"name": "mug", "package": "mug_a01", "x": 0.5, "y": 0.0, "z": 0.0}]}))
        sc = load_any(str(layout))
        self.assertTrue(sc["table"].fixed and "support" in sc["table"].tags)
        self.assertAlmostEqual(sc["mug"].meta["physics"]["static_friction"], 0.3, places=5)
        self.assertTrue(portable(sc["mug"].source).startswith("<SIMREADY_ASSETS_DIR>/"))
        from simready.gates.settle_mujoco import settle_and_measure, body_physics
        mu, rho = body_physics(sc["mug"], [(sc["mug"].verts, sc["mug"].faces)])
        self.assertAlmostEqual(mu, 0.3, places=5)
        self.assertAlmostEqual(rho * 0.08 * 0.08 * 0.1, 0.3, places=3)    # the density that gives the asset's mass
        rep = settle_and_measure(sc, seconds=0.1, support_tops={"table": 0.0})
        self.assertEqual(rep.asset_physics, ["table", "mug"])
        (self.root / "nvidia" / "simready-assets" / "mug_a01" / ".simready_fetch.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_any(str(layout))                                       # a repair never downloads

    def test_layout_fixtures_and_a_resting_side_it_chooses(self):
        from simready.cli import load_any
        write_package(self.root, "book_a01", (0.15, 0.21, 0.03), label="book", mass=0.4)     # authored lying flat, its height along y
        write_package(self.root, "slab_table_a01", (0.7, 1.0, 0.05), label="table", mass=20.0)
        write_package(self.root, "storage_cabinet_a01", (0.3, 0.5, 0.8), label="cabinet", mass=30.0)
        layout = self.root / "layout.json"
        doc = {"table": {"package": "slab_table_a01", "position": [0.55, 0.0, -0.05]},
               "fixtures": [{"name": "cabinet", "package": "storage_cabinet_a01", "position": [1.2, 0.0, 0.0]}],
               "objects": [{"name": "book", "package": "book_a01", "x": 0.5, "y": 0.0, "z": 0.0, "up": "+y", "front": "-x"}]}
        layout.write_text(json.dumps(doc))
        sc = load_any(str(layout))
        self.assertTrue(sc["cabinet"].fixed and {"support", "fixture"} <= sc["cabinet"].tags)
        self.assertNotIn("container", sc["cabinet"].tags)                    # 'cabinet' holds no word 'bin'
        self.assertAlmostEqual(float(np.ptp(sc["book"].world_vertices()[:, 2])), 0.21, places=4)   # standing on its edge
        self.assertEqual(sc["book"].meta["front_axis"], "-x")
        doc["objects"][0]["front"] = "+y"                                    # the side that ends up on top
        layout.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            load_any(str(layout))


class Words(unittest.TestCase):
    def test_tags_match_whole_words(self):
        from simready.io.nvidia_assets import has_word
        self.assertFalse(has_word("cabinet_file_d02 cabinet", "bin"))
        self.assertFalse(has_word("handTool_pliers_combination_a01 pliers", "bin"))
        self.assertTrue(has_word("storage_bin_a01 bin", "bin"))
        self.assertTrue(has_word("crate_box_open_a01", "box_open"))
        self.assertTrue(has_word("kitchenware_bowl_q01 bowl", "bowl"))


if __name__ == "__main__":
    unittest.main()
