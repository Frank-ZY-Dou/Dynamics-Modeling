"""Export a scene for a physics engine: meshes, a manifest, an MJCF file and a USD stage.

Every body's mesh is written in its own frame (the AABB-centred reference frame the solver
works in) as an OBJ file, and the manifest carries the world pose of every body. The MJCF
file and the USD stage are generated from the same data, so MuJoCo, Genesis (which reads the
manifest or the MJCF) and Isaac Sim (which reads the USD stage) simulate the same scene:

    out/
      manifest.json                bodies, poses (metres, Z up), mesh files, fixed flags, ground height
      meshes/0007_cup.obj          one mesh per body in the body frame, named by the body's index
      meshes/0007_cup/p0000.obj    CoACD pieces of that body (with --decompose)
      scene.xml                    MJCF: static bodies for fixtures, free joints for the rest, a floor plane
      scene.usda                   USD: UsdPhysics rigid bodies and colliders, a physics scene, a ground slab

File names carry the body's index, so two bodies whose names differ only by a suffix can never
share a file; every output path is checked for uniqueness before anything is written.

Fixed bodies keep their full mesh as a collider; MuJoCo replaces it by its convex hull, whose top
face is the plate of a table, and Isaac Sim keeps the triangle mesh for static colliders. A fixed
body whose mesh is a single plane is handled by what it is: a horizontal sheet at the ground
height, or one tagged `ground`, is the ground and is marked `flat` (the export's ground stands in
for it); any other sheet (a wall, a shelf board) is written with a small thickness so that every
engine keeps it as a solid collider, and the manifest records that thickness. Free bodies are
convex hulls unless `decompose` writes CoACD pieces for them.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from ..scene.model import Scene
from .paths import portable

SHEET_THICKNESS = 0.01     # metres given to a planar fixed body that is not the ground


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


def plane_normal(verts: np.ndarray):
    """Unit normal of a planar vertex set (rank 2), or None when the set spans a volume."""
    c = verts - verts.mean(0)
    if np.linalg.matrix_rank(c, tol=1e-6) >= 3:
        return None
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)


def thicken(verts: np.ndarray, faces: np.ndarray, normal: np.ndarray, thickness: float):
    """A solid slab from a planar mesh: the sheet offset by half the thickness to each side, its
    boundary edges closed by side walls."""
    n = np.asarray(normal, dtype=np.float64) * (0.5 * thickness)
    top = verts + n; bot = verts - n
    V = np.vstack([top, bot]); k = len(verts)
    F = [np.asarray(faces, dtype=np.int32), np.asarray(faces, dtype=np.int32)[:, ::-1] + k]
    edges = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges[key] = edges.get(key, 0) + 1
    boundary = [e for e, count in edges.items() if count == 1]
    sides = [[a, b, b + k] for a, b in boundary] + [[a, b + k, a + k] for a, b in boundary]
    if sides:
        F.append(np.array(sides, dtype=np.int32))
    return V, np.vstack(F).astype(np.int32)


def export_scene(scene: Scene, out_dir, decompose: bool = False, max_hull_verts: int = 256,
                 density: float = 500.0, friction=(1.0, 0.005, 0.0001), timestep: float = 2e-3) -> dict:
    out = Path(out_dir); (out / "meshes").mkdir(parents=True, exist_ok=True)
    names = [prim_name(b.name) for b in scene.bodies]
    if len(set(names)) != len(names):
        raise ValueError(f"body names collide after sanitising: {names}")
    z_floor = ground_height(scene)
    planned = {}                      # relative path -> (verts, faces); every path checked before writing
    bodies = []
    for k, (b, pn) in enumerate(zip(scene.bodies, names)):
        stem = f"{k:04d}_{pn}"
        normal = plane_normal(b.verts) if b.fixed else None
        horizontal = normal is not None and abs(float(normal[2])) > 0.99
        at_ground = horizontal and float(b.world_aabb()[0][2]) <= z_floor + 0.01
        flat = bool(b.fixed and ("ground" in b.tags or at_ground))
        verts, faces, thickness = b.verts, b.faces, 0.0
        if normal is not None and not flat:
            verts, faces = thicken(b.verts, b.faces, normal, SHEET_THICKNESS)
            thickness = SHEET_THICKNESS
        mesh_rel = f"meshes/{stem}.obj"
        planned[mesh_rel] = (verts, faces)
        pieces, note = [], ""
        if decompose and not b.fixed:
            from ..gates.settle_mujoco import coacd_pieces_info
            parts, used_hull = coacd_pieces_info(b)
            for j, (v, f) in enumerate(parts):
                rel = f"meshes/{stem}/p{j:04d}.obj"
                planned[rel] = (v, f)
                pieces.append(rel)
            note = "CoACD failed; one convex hull" if used_hull else f"{len(parts)} CoACD pieces"
        entry = {"name": b.name, "prim": pn, "fixed": bool(b.fixed), "flat": flat, "mesh": mesh_rel, "collision_pieces": pieces,
                 "position": [float(x) for x in b.center], "quaternion_wxyz": _quat_wxyz(b.rotation),
                 "tags": sorted(b.tags), "source": portable(b.source)}
        if thickness:
            entry["thickened_m"] = thickness
        if note:
            entry["collision_note"] = note
        bodies.append(entry)
    if len(planned) != len({os.path.normpath(p) for p in planned}):
        raise ValueError("export paths collide")
    for rel, (v, f) in planned.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        _write_obj(out / rel, v, f)
    manifest = {"units": "m", "up": "z", "ground_z": z_floor, "timestep": timestep, "density": density,
                "friction": list(friction), "max_hull_verts": max_hull_verts,
                "files": {"mjcf": "scene.xml", "usd": "scene.usda"}, "bodies": bodies}
    write_mjcf(manifest, out / "scene.xml")
    manifest["usd_written"] = write_usd(manifest, out / "scene.usda", planned)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def write_mjcf(manifest: dict, path) -> None:
    fr = " ".join(f"{x:g}" for x in manifest["friction"])
    assets, bodies = [], []
    for b in manifest["bodies"]:
        if b.get("flat"):
            continue                       # the floor plane stands in for the ground sheet
        files = b["collision_pieces"] or [b["mesh"]]
        geoms = []
        for k, rel in enumerate(files):
            mname = f"{b['prim']}_{k}"
            assets.append(f'<mesh name="{mname}" file="{os.path.relpath(rel, "meshes")}" inertia="convex" maxhullvert="{manifest["max_hull_verts"]}"/>')
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


def write_usd(manifest: dict, path, meshes: dict) -> bool:
    """A self-contained stage with UsdPhysics schemas; returns False when pxr is not installed.
    `meshes` maps the manifest's mesh paths to the (verts, faces) that were written."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt
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
    for b in manifest["bodies"]:
        if b.get("flat"):
            continue                       # the ground slab stands in for the ground sheet
        xf = UsdGeom.Xform.Define(stage, f"/World/{b['prim']}")
        xf.AddTranslateOp().Set(Gf.Vec3d(*b["position"]))
        w, x, y, z = b["quaternion_wxyz"]
        xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
        verts, faces = meshes[b["mesh"]]
        m = mesh_prim(f"/World/{b['prim']}", verts, faces)
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
