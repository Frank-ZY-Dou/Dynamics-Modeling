"""Export a scene for a physics engine: meshes, a manifest, an MJCF file and a USD stage.

Every body's mesh is written in its own frame (the AABB-centred reference frame the solver
works in) as an OBJ file, and the manifest carries the world pose of every body. The MJCF
file and the USD stage are generated from the same data, so MuJoCo, Genesis (which reads the
manifest or the MJCF) and Isaac Sim (which reads the USD stage) simulate the same scene:

    out/
      manifest.json      bodies, poses (metres, Z up), mesh files, fixed flags, ground height
      meshes/<name>.obj  one mesh per body in the body frame (plus CoACD pieces with --decompose)
      scene.xml          MJCF: static bodies for fixtures, free joints for the rest, a floor plane
      scene.usda         USD: UsdPhysics rigid bodies and colliders, a physics scene, a ground slab

Fixed bodies keep their full mesh as a collider; MuJoCo replaces it by its convex hull, whose top
face is the plate of a table, and Isaac Sim keeps the triangle mesh for static colliders. A flat
fixed sheet (a ground plane) has no volume to hull, so it is marked `flat` in the manifest and
the ground of the export stands in for it. Free bodies are convex hulls unless `decompose`
writes CoACD pieces for them.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from ..scene.model import Scene


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def _quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()   # xyzw
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def prim_name(name: str) -> str:
    """A USD/MJCF-safe identifier for a body name."""
    s = re.sub(r"\W", "_", name)
    return s if s and not s[0].isdigit() else "b_" + s


def ground_height(scene: Scene) -> float:
    fixed = [b for b in scene.bodies if b.fixed]
    if fixed:
        return min(float(b.world_aabb()[0][2]) for b in fixed) - 0.005
    return min(float(b.world_aabb()[0][2]) for b in scene.bodies) - 1.0


def export_scene(scene: Scene, out_dir, decompose: bool = False, max_hull_verts: int = 256,
                 density: float = 500.0, friction=(1.0, 0.005, 0.0001), timestep: float = 2e-3) -> dict:
    out = Path(out_dir); (out / "meshes").mkdir(parents=True, exist_ok=True)
    names = [prim_name(b.name) for b in scene.bodies]
    if len(set(names)) != len(names):
        raise ValueError(f"body names collide after sanitising: {names}")
    z_floor = ground_height(scene)
    bodies = []
    for b, pn in zip(scene.bodies, names):
        flat = bool(b.fixed and ("ground" in b.tags or int(np.linalg.matrix_rank(b.verts - b.verts.mean(0), tol=1e-6)) < 3))
        mesh_rel = f"meshes/{pn}.obj"
        _write_obj(out / mesh_rel, b.verts, b.faces)
        pieces = []
        if decompose and not b.fixed:
            from ..gates.settle_mujoco import coacd_pieces_info
            parts, _ = coacd_pieces_info(b)
            for k, (v, f) in enumerate(parts):
                rel = f"meshes/{pn}_p{k}.obj"
                _write_obj(out / rel, v, f)
                pieces.append(rel)
        bodies.append({"name": b.name, "prim": pn, "fixed": bool(b.fixed), "flat": flat, "mesh": mesh_rel, "collision_pieces": pieces,
                       "position": [float(x) for x in b.center], "quaternion_wxyz": _quat_wxyz(b.rotation),
                       "tags": sorted(b.tags), "source": b.source})
    manifest = {"units": "m", "up": "z", "ground_z": z_floor, "timestep": timestep, "density": density,
                "friction": list(friction), "max_hull_verts": max_hull_verts,
                "files": {"mjcf": "scene.xml", "usd": "scene.usda"}, "bodies": bodies}
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    write_mjcf(manifest, out / "scene.xml")
    manifest["usd_written"] = write_usd(scene, manifest, out / "scene.usda")
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def write_mjcf(manifest: dict, path) -> None:
    fr = " ".join(f"{x:g}" for x in manifest["friction"])
    assets, bodies = [], []
    for b in manifest["bodies"]:
        if b.get("flat"):
            continue                       # the floor plane stands in for a flat sheet
        files = b["collision_pieces"] or [b["mesh"]]
        geoms = []
        for k, rel in enumerate(files):
            mname = f"{b['prim']}_{k}"
            assets.append(f'<mesh name="{mname}" file="{os.path.basename(rel)}" inertia="convex" maxhullvert="{manifest["max_hull_verts"]}"/>')
            geoms.append(f'<geom type="mesh" mesh="{mname}" condim="3" friction="{fr}" density="{manifest["density"]:g}"/>')
        w, x, y, z = b["quaternion_wxyz"]
        pos = " ".join(f"{v:.6f}" for v in b["position"])
        joint = "" if b["fixed"] else "<freejoint/>"
        bodies.append(f'<body name="{b["prim"]}" pos="{pos}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">{joint}{"".join(geoms)}</body>')
    xml = (f'<mujoco model="simready_scene">\n<compiler angle="radian" meshdir="meshes" boundmass="0.001" boundinertia="1e-7"/>\n'
           f'<option timestep="{manifest["timestep"]:g}" gravity="0 0 -9.81"/>\n<asset>\n' + "\n".join(assets) + "\n</asset>\n<worldbody>\n"
           f'<geom name="floor" type="plane" size="0 0 1" pos="0 0 {manifest["ground_z"]:.6f}" condim="3" friction="{fr}"/>\n'
           + "\n".join(bodies) + "\n</worldbody>\n</mujoco>\n")
    with open(path, "w") as f:
        f.write(xml)


def write_usd(scene: Scene, manifest: dict, path) -> bool:
    """A self-contained stage with UsdPhysics schemas; returns False when pxr is not installed."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt
    except ImportError:
        return False
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0)); ps.CreateGravityMagnitudeAttr(9.81)

    def mesh_prim(parent_path, verts, faces):
        m = UsdGeom.Mesh.Define(stage, parent_path + "/mesh")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in verts]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(faces).reshape(-1)]))
        m.CreateSubdivisionSchemeAttr("none")
        return m

    # ground: a 40 m slab whose top is the manifest's ground height
    g = UsdGeom.Xform.Define(stage, "/World/ground")
    g.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, manifest["ground_z"] - 0.05))
    gv = np.array([[sx * 20.0, sy * 20.0, sz * 0.05] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    gf = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    gm = mesh_prim("/World/ground", gv, gf)
    UsdPhysics.CollisionAPI.Apply(gm.GetPrim())
    for body, b in zip(scene.bodies, manifest["bodies"]):
        if b.get("flat"):
            continue                       # the ground slab stands in for a flat sheet
        xf = UsdGeom.Xform.Define(stage, f"/World/{b['prim']}")
        xf.AddTranslateOp().Set(Gf.Vec3d(*b["position"]))
        w, x, y, z = b["quaternion_wxyz"]
        xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
        m = mesh_prim(f"/World/{b['prim']}", body.verts, body.faces)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
        mc.CreateApproximationAttr("none" if b["fixed"] else "convexDecomposition")
        if not b["fixed"]:
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateDensityAttr(float(manifest["density"]))
        xf.GetPrim().SetCustomDataByKey("simready:name", b["name"])
        xf.GetPrim().SetCustomDataByKey("simready:fixed", bool(b["fixed"]))
    stage.GetRootLayer().Save()
    return True
