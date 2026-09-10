"""USD import/export for RoboLab-style scenes (usd-core only, no Isaac).

A scene .usda holds a `world` Xform whose children are fixtures (payloads
under assets/fixtures) and objects (payloads under assets/objects), each
with translate/orient/scale xformOps. Every Mesh prim below a child is
gathered in world space, re-expressed in the child's rotation frame and
AABB-centred to form one Body.
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np

from .asset_rest import rest_rotation

from ..scene.model import Body, Scene


def _active_geom_prims(root):
    """Geometry prims below `root`: active prims only (Usd.PrimDefaultPredicate), instance
    proxies included, and purpose inherited (guide/proxy geometry is excluded)."""
    from pxr import Usd, UsdGeom
    out = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not _is_geom(p):
            continue
        if UsdGeom.Imageable(p).ComputePurpose() not in ("default", "render"):
            continue
        out.append(p)
    return out


def _triangulate(counts, indices):
    # returns an (m, 3) int array; polygons with fewer than 3 vertices are skipped
    tris, k = [], 0
    for c in counts:
        idx = indices[k:k + c]; k += c
        for t in range(1, c - 1):
            tris.append((idx[0], idx[t], idx[t + 1]))
    return np.asarray(tris, dtype=np.int32)


def _matrix_np(m):
    """pxr Gf.Matrix4d (row-vector convention) -> (R, t) acting on column vectors."""
    a = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    R = a[:3, :3].T
    t = a[3, :3]
    return R, t


def _gprim_mesh(prim):
    """Tessellate a UsdGeom gprim (Cube/Cylinder/Sphere/Capsule/Cone) in its local frame."""
    import trimesh
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        counts = list(m.GetFaceVertexCountsAttr().Get() or [])
        idx = list(m.GetFaceVertexIndicesAttr().Get() or [])
        if pts.size == 0 or not counts:
            return None
        return pts, _triangulate(counts, idx)
    if prim.IsA(UsdGeom.Cube):
        sz = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        t = trimesh.creation.box(extents=(sz, sz, sz))
        return np.asarray(t.vertices, dtype=np.float64), np.asarray(t.faces, dtype=np.int32)
    axis_map = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
    if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cone):
        g = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else (UsdGeom.Capsule(prim) if prim.IsA(UsdGeom.Capsule) else UsdGeom.Cone(prim))
        r = float(g.GetRadiusAttr().Get() or 1.0); h = float(g.GetHeightAttr().Get() or 2.0)
        ax = str(g.GetAxisAttr().Get() or "Z")
        if prim.IsA(UsdGeom.Capsule):
            t = trimesh.creation.capsule(height=h, radius=r)
        elif prim.IsA(UsdGeom.Cone):
            t = trimesh.creation.cone(radius=r, height=h); t.apply_translation([0, 0, -h / 2])
        else:
            t = trimesh.creation.cylinder(radius=r, height=h)
        if ax != "Z":
            from scipy.spatial.transform import Rotation as Rot
            Rax = Rot.align_vectors([axis_map[ax]], [(0, 0, 1)])[0].as_matrix()
            t.apply_transform(np.block([[Rax, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
        return np.asarray(t.vertices, dtype=np.float64), np.asarray(t.faces, dtype=np.int32)
    if prim.IsA(UsdGeom.Sphere):
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        t = trimesh.creation.icosphere(subdivisions=2, radius=r)
        return np.asarray(t.vertices, dtype=np.float64), np.asarray(t.faces, dtype=np.int32)
    return None


def _is_geom(prim):
    from pxr import UsdGeom
    return any(prim.IsA(T) for T in (UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Capsule, UsdGeom.Cone, UsdGeom.Sphere))


def _rotation_from_affine(R: np.ndarray) -> np.ndarray:
    """Polar-decompose an affine 3x3 (rotation * scale) into a proper rotation."""
    u, _, vt = np.linalg.svd(R)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def child_source(child, scene_path):
    """Provenance of a scene child: (source path, payload targets). A payload path is relative
    to the layer that declares it, the scene file here."""
    src = ""
    for arc in child.GetPrimStack():
        src = arc.layer.identifier
    payload_targets = []
    for spec in child.GetPrimStack():
        pl = getattr(spec, "payloadList", None)
        for pp in (pl.GetAddedOrExplicitItems() if pl else []):
            payload_targets.append(pp.assetPath)
    srcpath = payload_targets[0] if payload_targets else src
    if srcpath and not os.path.isabs(srcpath) and not srcpath.startswith(("omniverse:", "http")):
        srcpath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(str(scene_path))), srcpath))
    return srcpath, payload_targets


def _stage_frame(stage, path):
    """(metersPerUnit, upAxis) as authored, None for each the stage does not author. Every stage
    read here works in metres, Z up: one that says otherwise is refused rather than read as if it
    were, one that says nothing is taken as metres, Z up (RoboLab authors both). GetMetadata would
    return the schema fallback, centimetres and Y up, for a stage that authors nothing, so only
    authored values are judged."""
    mpu = stage.GetMetadata("metersPerUnit") if stage.HasAuthoredMetadata("metersPerUnit") else None
    if mpu is not None and abs(float(mpu) - 1.0) > 1e-9:
        raise ValueError(f"{path}: stage metersPerUnit={mpu}; only stages authored in metres are supported")
    up_axis = stage.GetMetadata("upAxis") if stage.HasAuthoredMetadata("upAxis") else None
    if up_axis is not None and str(up_axis) != "Z":
        raise ValueError(f"{path}: stage upAxis={up_axis}; only Z-up stages are supported")
    return mpu, up_axis


def load_scene_usda(path, fixture_key: str = "fixtures", world_prim: str | None = None,
                    support_names=("table", "franka_table", "island", "counter")) -> Scene:
    from pxr import Usd, UsdGeom
    path = str(path)
    try:
        stage = Usd.Stage.Open(path)
    except Exception as e:  # pxr reports a missing or unreadable file as an exception
        raise ValueError(f"cannot open USD stage {path}: {e}") from e
    if stage is None:
        raise ValueError(f"cannot open USD stage {path}")
    mpu, up_axis = _stage_frame(stage, path)
    root = stage.GetPrimAtPath(world_prim) if world_prim else stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot().GetChildren()[0]
    cache = UsdGeom.XformCache()
    bodies = []
    dropped = []
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xformable) and not child.GetChildren():
            continue
        meshes = _active_geom_prims(child)
        if not meshes:
            if child.HasAuthoredPayloads() or child.HasAuthoredReferences():
                dropped.append(child.GetName())          # a referenced asset that did not resolve
            continue
        # world-frame vertex soup for this child
        V, F, off = [], [], 0
        for mp in meshes:
            got = _gprim_mesh(mp)
            if got is None or len(got[1]) == 0:
                continue
            pts, tri = got
            Rm, tm = _matrix_np(cache.GetLocalToWorldTransform(mp))
            V.append((Rm @ pts.T).T + tm)
            F.append(tri + off)
            off += len(pts)
        if not V:
            continue
        Vw = np.vstack(V); Fw = np.vstack(F)
        Rc_aff, tc = _matrix_np(cache.GetLocalToWorldTransform(child))
        Rc = _rotation_from_affine(Rc_aff)
        local = (Rc.T @ (Vw - tc).T).T
        srcpath, payload_targets = child_source(child, path)
        # An asset whose authored pose cannot stand is re-expressed in its resting frame
        # (simready.io.asset_rest): the vertices turn once, the pose rotation absorbs the
        # inverse, the world geometry is unchanged.
        R_rest = rest_rotation(srcpath or child.GetName()) if payload_targets else rest_rotation(child.GetName())
        if R_rest is not None:
            local = local @ R_rest.T
            Rc = Rc @ R_rest.T
        lo, hi = local.min(0), local.max(0)
        c_model = 0.5 * (lo + hi)
        center = tc + Rc @ c_model
        lname = child.GetName().lower()
        inline = not payload_targets                      # authored in the scene layer itself
        is_fixture = (f"/{fixture_key}/" in srcpath.replace("\\", "/")) or any(k in lname for k in support_names) \
            or (inline and any(k in lname for k in ("ground", "plane", "wall", "floor")))
        tags = set()
        if is_fixture:
            tags.add("fixture")
            if any(k in lname for k in support_names):
                tags.add("support")
            if any(k in lname for k in ("ground", "floor")):
                tags.add("ground")
        meta = {"prim": str(child.GetPath()), "n_mesh_prims": len(meshes),
                "c_model": c_model.copy(), "center_loaded": center.copy()}
        if R_rest is not None:
            meta["rest_rotation"] = R_rest.copy()
        bodies.append(Body(name=child.GetName(), verts=local - c_model, faces=Fw, center=center,
                           rotation=Rc, fixed=is_fixture, tags=tags, source=srcpath, meta=meta))
    if dropped:
        import warnings
        warnings.warn(f"{path}: {len(dropped)} referenced children without geometry (unresolved payloads?): {dropped}")
    layers = sorted({os.path.abspath(lay.realPath) for lay in stage.GetUsedLayers()
                     if getattr(lay, "realPath", "") and os.path.isfile(lay.realPath)
                     and os.path.abspath(lay.realPath) != os.path.abspath(path)})
    return Scene(bodies, meta={"path": path, "root": str(root.GetPath()), "dropped": dropped, "layers": layers,
                               "meters_per_unit": None if mpu is None else float(mpu), "up_axis": None if up_axis is None else str(up_axis)})


def _quat_wxyz(R: np.ndarray):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def write_scene_poses(scene: Scene, src_path, out_path, world_prim: str | None = None):
    """Write the free bodies' poses back into a copy of the scene layer.

    The Body's reference centre is c = t + R c_model, so the prim translate is
    t = c - R c_model; c_model is recovered from the mesh at load time and
    stored on the body as meta['c_model'] when available, else re-derived.
    """
    from pxr import Usd, UsdGeom, Gf, Sdf
    stage = Usd.Stage.Open(str(src_path))
    root = stage.GetPrimAtPath(world_prim) if world_prim else stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot().GetChildren()[0]
    cache = UsdGeom.XformCache()
    for b in scene.free():
        prim = root.GetChild(b.name)
        if not prim or not prim.IsValid():
            continue
        R_old_aff, t_old = _matrix_np(cache.GetLocalToWorldTransform(prim))
        R_old = _rotation_from_affine(R_old_aff)
        if "c_model" in b.meta:
            c_model = np.asarray(b.meta["c_model"], dtype=np.float64)
        elif "center_loaded" in b.meta:
            c_model = R_old.T @ (np.asarray(b.meta["center_loaded"]) - t_old)
        else:
            raise ValueError(f"{b.name}: no c_model / center_loaded in meta; load the scene with load_scene_usda")
        # the child's authored scale is baked into its mesh (local = R_old^T (world - t_old)), so the
        # new world pose is T(t_new) R_new S_old with S_old = R_old^T R_aff; a body kept in its
        # resting frame (meta rest_rotation) turns back into the authored frame here
        S_old = R_old.T @ R_old_aff
        R_authored = b.rotation @ np.asarray(b.meta.get("rest_rotation", np.eye(3)), dtype=np.float64)
        t_new_w = b.center - b.rotation @ c_model
        # express in the parent's frame (a prim that resets the xform stack ignores its parent)
        xf = UsdGeom.Xformable(prim)
        reset = bool(xf.GetResetXformStack())
        if reset:
            P_aff, P_t = np.eye(3), np.zeros(3)
        else:
            P_aff, P_t = _matrix_np(cache.GetLocalToWorldTransform(prim.GetParent()))
        P_inv = np.linalg.inv(P_aff)
        t_loc = P_inv @ (t_new_w - P_t)
        M_loc = P_inv @ (R_authored @ S_old)
        R_loc = _rotation_from_affine(M_loc)
        S_loc = R_loc.T @ M_loc
        scale = np.diag(S_loc)
        if np.abs(S_loc - np.diag(scale)).max() > 1e-9:
            # the local map has shear (a scale authored after the rotation, or a non-uniform parent
            # scale under a rotation): translate, orient and scale cannot express it; one matrix op does
            xf.ClearXformOpOrder()
            M4 = np.eye(4)
            M4[:3, :3] = M_loc.T                 # Gf matrices act on row vectors
            M4[3, :3] = t_loc
            xf.AddTransformOp().Set(Gf.Matrix4d(*[float(x) for x in M4.ravel()]))
            xf.SetResetXformStack(reset)
            continue
        names = [op.GetOpName() for op in xf.GetOrderedXformOps()]
        canonical = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        # the authored ops are reused only when they compose as translate, orient, scale in that
        # order, which is what the pose is decomposed into
        simple = all(n in canonical for n in names) and names == [c for c in canonical if c in names]
        if not simple:
            # rotateXYZ / transform / pivot stacks or another op order: replace by translate, orient, scale
            xf.ClearXformOpOrder()
            tr = xf.AddTranslateOp(); orient = xf.AddOrientOp(); sc = xf.AddScaleOp()
            sc.Set(Gf.Vec3f(*[float(x) for x in scale]))
            xf.SetResetXformStack(reset)
        else:
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            tr = ops.get("xformOp:translate") or xf.AddTranslateOp()
            orient = ops.get("xformOp:orient") or xf.AddOrientOp()
            if "xformOp:scale" in ops:
                ops["xformOp:scale"].Set(Gf.Vec3f(*[float(x) for x in scale]) if ops["xformOp:scale"].GetAttr().GetTypeName() == Sdf.ValueTypeNames.Float3 else Gf.Vec3d(*[float(x) for x in scale]))
        if tr.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Float3:
            tr.Set(Gf.Vec3f(*[float(x) for x in t_loc]))
        else:
            tr.Set(Gf.Vec3d(*[float(x) for x in t_loc]))
        w, x, y, z = _quat_wxyz(R_loc)
        if orient.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Quatd:
            orient.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
        else:
            orient.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    src_dir = os.path.dirname(os.path.abspath(str(src_path))); out_dir = os.path.dirname(os.path.abspath(str(out_path)))
    if src_dir != out_dir:
        # the scene refers to its assets by relative paths: re-anchor them so the copy still resolves
        for prim in stage.Traverse():
            for kind in ("payloads", "references"):
                lst = prim.GetPayloads() if kind == "payloads" else prim.GetReferences()
                items = []
                for spec in prim.GetPrimStack():
                    field = spec.payloadList if kind == "payloads" else spec.referenceList
                    for it in field.GetAddedOrExplicitItems():
                        if it.assetPath and not os.path.isabs(it.assetPath) and not it.assetPath.startswith(("http", "omniverse")):
                            items.append((os.path.normpath(os.path.join(os.path.dirname(spec.layer.realPath or src_dir), it.assetPath)), it.primPath))
                if items:
                    if kind == "payloads":
                        lst.ClearPayloads()
                        for ap_, pp in items:
                            lst.AddPayload(Sdf.Payload(ap_, pp) if pp else Sdf.Payload(ap_))
                    else:
                        lst.ClearReferences()
                        for ap_, pp in items:
                            lst.AddReference(Sdf.Reference(ap_, pp) if pp else Sdf.Reference(ap_))
    stage.GetRootLayer().Export(str(out_path))
    return out_path


def load_object_usd(path):
    """Mesh (verts, faces) of an object USD in its own default-prim frame, plus
    the AABB centre offset. Used to instantiate catalog objects at solver poses."""
    from pxr import Usd, UsdGeom
    import os
    if not os.path.exists(str(path)):
        raise FileNotFoundError(f"object USD not found: {path}")
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as e:  # pxr reports an unreadable file as an exception
        raise ValueError(f"cannot open object USD {path}: {e}") from e
    if stage is None:
        raise ValueError(f"cannot open object USD {path}")
    _stage_frame(stage, path)
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    V, F, off = [], [], 0
    for mp in _active_geom_prims(root):
        got = _gprim_mesh(mp)
        if got is None or len(got[1]) == 0:
            continue
        pts, tri = got
        Rm, tm = _matrix_np(cache.GetLocalToWorldTransform(mp))
        V.append((Rm @ pts.T).T + tm)
        F.append(tri + off)
        off += len(pts)
    if not V:
        raise ValueError(f"no Mesh prims in {path}")
    return np.vstack(V), np.vstack(F)


def body_from_object_usd(name, path, position, yaw_deg=0.0, quat_wxyz=None, fixed=False, tags=None):
    """Instantiate an object USD at a RoboLab solver pose (translate + yaw or quat)."""
    import math
    verts, faces = load_object_usd(path)
    R_rest = rest_rotation(path)
    if R_rest is not None:                      # resting frame, see simready.io.asset_rest
        verts = verts @ R_rest.T
    if quat_wxyz is not None and yaw_deg not in (0, 0.0, None):
        raise ValueError("give either yaw_deg or quat_wxyz, not both")
    if quat_wxyz is not None:
        w, x, y, z = quat_wxyz
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    else:
        t = math.radians(yaw_deg)
        R = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
    lo, hi = verts.min(0), verts.max(0)
    c_model = 0.5 * (lo + hi)
    t = np.asarray(position, dtype=np.float64)
    meta = {"prim_translate": t.copy(), "c_model": c_model}
    if R_rest is not None:
        meta["rest_rotation"] = R_rest.copy()
    b = Body(name=name, verts=verts - c_model, faces=faces, center=t + R @ c_model, rotation=R,
             fixed=fixed, tags=set(tags or ()), source=str(path), meta=meta)
    return b
