"""MJCF import for RoboCasa / robosuite objects: the geometry MuJoCo actually simulates.

An object `model.xml` is compiled with MuJoCo itself (`mujoco.MjModel.from_xml_path`), so
every compiler rule is honoured: `<default>` classes (contype / conaffinity / group),
mesh `scale`, `refpos` / `refquat`, euler conventions, nested bodies, mesh re-centring.
Collision geometry = geoms with contype | conaffinity != 0; visual geometry = the rest
(robosuite keeps visuals in group 1 with contype = conaffinity = 0). RoboCasa's
`reg_bbox` region box (contype = conaffinity = 0) is therefore never collision geometry.

All geometry is returned in the object's root-body frame, the frame robosuite's placement
samplers position (`pos`, `quat` of the root body).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from ..scene.model import Body, Scene

_MODEL_CACHE: dict = {}


def _quat_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sanitized_xml(xml_path) -> str:
    """The object XML as MuJoCo can compile it standalone: mesh/texture files made absolute, and
    textures whose files do not exist (RoboCasa ships absolute paths from the authors' machines)
    dropped together with the material attribute that referenced them."""
    xml_path = Path(xml_path).resolve()
    tree = ET.parse(xml_path); root = tree.getroot(); base = xml_path.parent
    comp = root.find("compiler")
    meshdir = Path(comp.get("meshdir", "")) if comp is not None else Path("")
    texdir = Path(comp.get("texturedir", "")) if comp is not None else Path("")
    if comp is not None:
        comp.attrib.pop("meshdir", None); comp.attrib.pop("texturedir", None)
    for m in root.iter("mesh"):
        f = m.get("file")
        if f:
            m.set("file", str(f if Path(f).is_absolute() else (base / meshdir / f).resolve()))
    dropped = set()
    for asset in root.iter("asset"):
        for t in list(asset.findall("texture")):
            f = t.get("file")
            if f is None:
                continue
            fp = Path(f) if Path(f).is_absolute() else (base / texdir / f)
            if fp.exists():
                t.set("file", str(fp.resolve()))
            else:
                dropped.add(t.get("name")); asset.remove(t)
    for mat in root.iter("material"):
        if mat.get("texture") in dropped:
            mat.attrib.pop("texture")
    return ET.tostring(root, encoding="unicode")


def compile_object(xml_path):
    """(model, data, root_body_id) for an object XML, compiled by MuJoCo and forwarded once."""
    import mujoco
    xml_path = str(Path(xml_path).resolve())
    if xml_path in _MODEL_CACHE:
        return _MODEL_CACHE[xml_path]
    model = mujoco.MjModel.from_xml_string(sanitized_xml(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    if not roots:
        raise ValueError(f"no body in {xml_path}")
    _MODEL_CACHE[xml_path] = (model, data, roots[0])
    return _MODEL_CACHE[xml_path]


def _geom_mesh(model, g):
    """(verts, faces, texcoords|None, face_texcoords|None) of geom g in the geom frame."""
    import mujoco
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64)
        f = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int32)
        tc = ftc = None
        try:
            ta, tn = model.mesh_texcoordadr[mid], model.mesh_texcoordnum[mid]
            if tn > 0:
                tc = np.asarray(model.mesh_texcoord[ta:ta + tn], dtype=np.float64)
                ftc = np.asarray(model.mesh_facetexcoord[fa:fa + fn], dtype=np.int32)
        except AttributeError:
            pass
        return v, f, tc, ftc
    sz = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        m = trimesh.creation.box(extents=(2 * sz[0], 2 * sz[1], 2 * sz[2]))
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        m = trimesh.creation.icosphere(subdivisions=2, radius=float(sz[0]))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        m = trimesh.creation.cylinder(radius=float(sz[0]), height=2 * float(sz[1]))
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # MuJoCo: size = (radius, half-length of the cylindrical part); the hemispherical caps
        # add a radius at each end, so the full length is 2 (h + r). trimesh's capsule takes the
        # cylindrical height and is centred at the origin, like the MuJoCo geom frame.
        m = trimesh.creation.capsule(radius=float(sz[0]), height=2 * float(sz[1]))
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0); m.apply_scale(sz[:3])
    else:
        return None
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32), None, None


def object_geoms(xml_path, collision: bool = True):
    """Per-geom [(name, verts, faces, texcoords, face_texcoords)] in the root-body frame:
    collision geoms (contype | conaffinity != 0) or visual geoms (the others)."""
    model, data, root = compile_object(xml_path)
    R0 = data.xmat[root].reshape(3, 3); t0 = data.xpos[root]
    import mujoco
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        is_col = (int(model.geom_contype[g]) | int(model.geom_conaffinity[g])) != 0
        if is_col != collision:
            continue
        got = _geom_mesh(model, g)
        if got is None:
            continue
        v, f, tc, ftc = got
        Rg = data.geom_xmat[g].reshape(3, 3); tg = data.geom_xpos[g]
        vw = (R0.T @ ((Rg @ v.T).T + tg - t0).T).T          # geom -> world -> root frame
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        out.append((name, vw, f, tc, ftc))
    return out


def load_mjcf_object(xml_path, collision_group: int = 0, prefer_visual_if_empty: bool = True):
    """(verts, faces) of the object's collision geometry (MuJoCo's own), root-body frame.
    `collision_group` is kept for API compatibility: 0 = collision, anything else = visual."""
    geoms = object_geoms(xml_path, collision=(collision_group == 0))
    if not geoms and prefer_visual_if_empty:
        geoms = object_geoms(xml_path, collision=False)
    if not geoms:
        raise ValueError(f"no geometry in {xml_path}")
    V, F, off = [], [], 0
    for _, v, f, _, _ in geoms:
        V.append(v); F.append(f + off); off += len(v)
    return np.vstack(V), np.vstack(F)


def load_mjcf_pieces(xml_path, collision_group: int = 0):
    """Per-geom collision pieces [(verts, faces), ...] in the root-body frame."""
    return [(v, f) for _, v, f, _, _ in object_geoms(xml_path, collision=True)]


def region_bbox(xml_path):
    """RoboCasa's `reg_bbox` (centre, half_size) in the root-body frame, or None."""
    root = ET.parse(xml_path).getroot()
    for g in root.iter("geom"):
        if g.get("name") == "reg_bbox":
            pos = np.array([float(x) for x in g.get("pos", "0 0 0").split()])
            half = np.array([float(x) for x in g.get("size").split()])
            return pos, half
    return None


def body_from_mjcf_object(name, xml_path, pos, quat_wxyz=(1, 0, 0, 0), fixed=False, tags=None):
    verts, faces = load_mjcf_object(xml_path)
    R = _quat_to_R(quat_wxyz)
    lo, hi = verts.min(0), verts.max(0)
    c_model = 0.5 * (lo + hi)
    t = np.asarray(pos, dtype=np.float64)
    return Body(name=name, verts=verts - c_model, faces=faces, center=t + R @ c_model, rotation=R,
                fixed=fixed, tags=set(tags or ()), source=str(xml_path), meta={"c_model": c_model})


def mjcf_object_stats(xml_path):
    """G0 admission facts from the compiled model: collision / visual geom counts, whether the
    merged collision mesh is watertight, extents, and the reg_bbox versus collision extents."""
    col = object_geoms(xml_path, collision=True); vis = object_geoms(xml_path, collision=False)
    v, f = load_mjcf_object(xml_path)
    m = trimesh.Trimesh(v, f, process=False)
    rb = region_bbox(xml_path)
    ext = v.max(0) - v.min(0)
    return {"collision_geoms": len(col), "visual_geoms": len(vis), "watertight": bool(m.is_watertight),
            "verts": int(len(v)), "faces": int(len(f)), "extent": [float(x) for x in ext],
            "reg_bbox_extent": [float(2 * x) for x in rb[1]] if rb is not None else None,
            "collision_below_bbox_mm": float(1000 * ((rb[0][2] - rb[1][2]) - v[:, 2].min())) if rb is not None else None}


def load_mjcf_scene(xml_path):  # pragma: no cover - reserved for robosuite arenas
    raise NotImplementedError("scene-level MJCF import is not wired yet; use body_from_mjcf_object per object")
