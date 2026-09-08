"""Export a RoboLab object/fixture USD to OBJ (+ a materials JSON) for rendering.

The geometry is exactly what the solver sees: the same prim traversal and local-to-world
composition as `simready.io.usd_io.load_object_usd`, so the render mesh and the collision
mesh are the same vertices in the same (native default-prim) frame. UVs (`primvars:st`) and
per-face material assignment (GeomSubsets) are carried into the OBJ; OmniPBR/SimPBR MDL
shader inputs (diffuse texture or constant colour, normal map, roughness, metallic, ORM)
are written to `<name>.materials.json` because MTL cannot express them.

Usage: python viz/usd_to_obj.py <file.usd> <out_dir> [--name NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simready  # noqa: E402,F401
from simready.io.usd_io import _gprim_mesh, _is_geom, _matrix_np  # noqa: E402

ROBOLAB_MATERIALS = os.path.join(os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "RoboLab")), "assets", "materials")

TEX_INPUTS = {
    "inputs:diffuse_texture": "diffuse_texture",
    "inputs:normalmap_texture": "normal_texture",
    "inputs:reflectionroughness_texture": "roughness_texture",
    "inputs:metallic_texture": "metallic_texture",
    "inputs:ORM_texture": "orm_texture",
    "inputs:opacity_texture": "opacity_texture",
    "inputs:emissive_mask_texture": "emissive_texture",
}
SCALAR_INPUTS = {
    "inputs:diffuse_color_constant": "diffuse_color",
    "inputs:diffuse_tint": "diffuse_tint",
    "inputs:reflection_roughness_constant": "roughness",
    "inputs:metallic_constant": "metallic",
    "inputs:specular_level": "specular",
    "inputs:enable_ORM_texture": "enable_orm",
    "inputs:opacity_constant": "opacity",
    "inputs:normalmap_scale": "normal_scale",  # not in OmniPBR, harmless
    "inputs:bump_factor": "normal_strength",
    "inputs:texture_scale": "texture_scale",
    "inputs:texture_translate": "texture_translate",
    "inputs:texture_rotate": "texture_rotate",
}


def _asset_path(value, layer_dir):
    if value is None:
        return None
    p = getattr(value, "resolvedPath", "") or ""
    if not p:
        raw = getattr(value, "path", "") or str(value)
        if raw.startswith("@") and raw.endswith("@"):
            raw = raw[1:-1]
        p = raw if os.path.isabs(raw) else os.path.normpath(os.path.join(layer_dir, raw))
    return p if p and os.path.exists(p) else None


def _shader_of(material):
    """The MDL/preview shader prim under a Material: the first Shader child (Omni assets
    keep one shader per material); returns None for materials without a shader."""
    from pxr import UsdShade
    for child in material.GetPrim().GetChildren():
        if child.GetTypeName() == "Shader":
            return UsdShade.Shader(child)
    return None


def _material_record(material, layer_dir, cache):
    key = material.GetPath().pathString
    if key in cache:
        return key
    rec = {"name": key.strip("/").replace("/", "_"), "prim": key}
    sh = _shader_of(material)
    if sh is not None:
        prim = sh.GetPrim()
        src = prim.GetAttribute("info:mdl:sourceAsset")
        rec["mdl"] = str(src.Get()) if src and src.Get() else None
        for a in prim.GetAttributes():
            n = a.GetName()
            if n in TEX_INPUTS:
                rec[TEX_INPUTS[n]] = _asset_path(a.Get(), layer_dir)
            elif n in SCALAR_INPUTS:
                v = a.Get()
                if v is None:
                    continue
                try:
                    rec[SCALAR_INPUTS[n]] = [float(x) for x in v]
                except TypeError:
                    rec[SCALAR_INPUTS[n]] = float(v) if not isinstance(v, bool) else bool(v)
    if sh is not None:
        _fill_gltf(rec, material, sh.GetPrim(), layer_dir)
    if sh is not None and not rec.get("diffuse_texture") and not rec.get("diffuse_color"):
        _fill_from_mdl(rec, sh.GetPrim(), layer_dir)
    if not rec.get("diffuse_texture") and not rec.get("diffuse_color"):
        _fill_by_name(rec, material.GetPath().pathString)
    cache[key] = rec
    return key


def _fill_gltf(rec, material, shader_prim, layer_dir):
    """glTF-converted assets: a `gltf_material` shader (base_color_factor, roughness_factor,
    metallic_factor) plus sibling `gltf_texture_lookup` shaders whose `inputs:texture` carry the
    maps (names: baseColorTex, normalTex, metallicRoughnessTex, ...)."""
    sub = shader_prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
    if not sub or sub.Get() != "gltf_material":
        return
    for name, out in (("base_color_factor", "diffuse_color"), ("roughness_factor", "roughness"), ("metallic_factor", "metallic")):
        a = shader_prim.GetAttribute("inputs:" + name)
        v = a.Get() if a else None
        if v is not None:
            try:
                rec[out] = [float(x) for x in v]
            except TypeError:
                rec[out] = float(v)
    for child in material.GetPrim().GetChildren():
        if child.GetTypeName() != "Shader" or child == shader_prim:
            continue
        tex = child.GetAttribute("inputs:texture")
        path = _asset_path(tex.Get(), layer_dir) if tex else None
        if not path:
            continue
        n = child.GetName().lower()
        if "basecolor" in n or "base_color" in n or "diffuse" in n or "albedo" in n:
            rec["diffuse_texture"] = path
        elif "normal" in n:
            rec["normal_texture"] = path
        elif "metallicroughness" in n or "orm" in n:
            rec["orm_texture"] = path; rec["enable_orm"] = True
        elif "emissive" in n:
            rec["emissive_texture"] = path


_NAME_COLOURS = [("black", (0.03, 0.03, 0.03)), ("white", (0.9, 0.9, 0.9)), ("red", (0.7, 0.1, 0.08)), ("green", (0.15, 0.5, 0.2)),
                 ("blue", (0.15, 0.3, 0.7)), ("yellow", (0.85, 0.75, 0.15)), ("orange", (0.9, 0.5, 0.1)), ("brown", (0.4, 0.25, 0.12)),
                 ("oak", (0.55, 0.38, 0.22)), ("wood", (0.5, 0.35, 0.2)), ("grey", (0.5, 0.5, 0.5)), ("gray", (0.5, 0.5, 0.5)),
                 ("steel", (0.6, 0.6, 0.62)), ("metal", (0.55, 0.55, 0.57)), ("chrome", (0.8, 0.8, 0.82))]


def _fill_by_name(rec, path):
    """Library materials without readable inputs (vMaterials carpaint, ...): a colour from the name."""
    n = path.lower()
    for key, col in _NAME_COLOURS:
        if key in n:
            rec["diffuse_color"] = list(col)
            break
    else:
        rec["diffuse_color"] = [0.6, 0.6, 0.6]
    if "matte" in n:
        rec["roughness"] = 0.85
    if "metal" in n or "steel" in n or "chrome" in n:
        rec["metallic"] = 0.8; rec["roughness"] = rec.get("roughness", 0.35)
    if "gloss" in n or "paint" in n:
        rec["roughness"] = rec.get("roughness", 0.25)
    rec["guessed"] = True


_MDL_TEX = {"diffuse_texture": "diffuse_texture", "normalmap_texture": "normal_texture", "ORM_texture": "orm_texture",
            "reflectionroughness_texture": "roughness_texture", "metallic_texture": "metallic_texture"}


def _fill_from_mdl(rec, shader_prim, layer_dir):
    """Omniverse library materials (Base/Wood/Oak.mdl, ...) keep their textures inside the .mdl
    text: parse `texture_2d("./Oak/Oak_BaseColor.png")` style defaults and colour constants."""
    import re
    src = shader_prim.GetAttribute("info:mdl:sourceAsset")
    path = _asset_path(src.Get(), layer_dir) if src else None
    if not path and src and src.Get():
        # Omniverse base materials referenced by URL: RoboLab ships the same library locally
        raw = str(getattr(src.Get(), "path", src.Get())).strip("@")
        if "/Materials/Base/" in raw:
            cand = os.path.join(ROBOLAB_MATERIALS, "Base", raw.split("/Materials/Base/", 1)[1])
            if os.path.exists(cand):
                path = cand
    if not path or not path.endswith(".mdl") or not os.path.exists(path):
        return
    txt = open(path, errors="ignore").read()
    mdl_dir = os.path.dirname(path)
    for key, out in _MDL_TEX.items():
        m = re.search(key + r"\s*:\s*texture_2d\(\s*\"([^\"]+)\"", txt)
        if m:
            tp = os.path.normpath(os.path.join(mdl_dir, m.group(1)))
            if os.path.exists(tp):
                rec[out] = tp
                if out == "orm_texture":
                    rec["enable_orm"] = True
    m = re.search(r"diffuse_color_constant\s*:\s*color\(([^)]+)\)", txt)
    if m and not rec.get("diffuse_color"):
        try:
            vals = [float(x) for x in m.group(1).replace("f", "").split(",")]
            rec["diffuse_color"] = vals[:3] if len(vals) >= 3 else [vals[0]] * 3
        except ValueError:
            pass
    for key, out in (("reflection_roughness_constant", "roughness"), ("metallic_constant", "metallic")):
        m = re.search(key + r"\s*:\s*([0-9.]+)f?", txt)
        if m and out not in rec:
            rec[out] = float(m.group(1))
    rec["mdl_file"] = path


def _gather(root, xcache, usd_path, to_local=None):
    """Collect triangles, uvs and per-face materials below `root`, in world coordinates of its
    stage or, with to_local=(Rc, tc), in the frame Rc^T (x - tc) (the scene child's frame with
    its scale baked in, exactly as simready.io.usd_io.load_scene_usda builds body meshes)."""
    from pxr import Usd, UsdGeom, UsdShade
    V, VT, F, FT, FM = [], [], [], [], []      # vertices, uvs, tri vertex idx, tri uv idx (or -1), tri material key
    mats = {}
    v_off = vt_off = 0
    n_prims = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not _is_geom(prim) or UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide":
            continue
        layer_dir = os.path.dirname(prim.GetPrimStack()[0].layer.realPath or usd_path) if prim.GetPrimStack() else os.path.dirname(usd_path)
        Rm, tm = _matrix_np(xcache.GetLocalToWorldTransform(prim))
        mesh_mat = None
        try:
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if bound and bound.GetPrim().IsValid():
                mesh_mat = _material_record(bound, layer_dir, mats)
        except Exception:  # noqa: BLE001
            mesh_mat = None
        if prim.GetTypeName() == "Mesh":
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get() or [], dtype=np.float64)
            counts = list(m.GetFaceVertexCountsAttr().Get() or [])
            idx = list(m.GetFaceVertexIndicesAttr().Get() or [])
            if len(pts) == 0 or not counts:
                continue
            # uv
            uv_vals, uv_mode = None, None
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if pv and pv.HasValue():
                try:
                    flat = np.asarray(pv.ComputeFlattened(), dtype=np.float64)
                    interp = pv.GetInterpolation()
                    if interp == "faceVarying" and len(flat) == len(idx):
                        uv_vals, uv_mode = flat, "fv"
                    elif interp == "vertex" and len(flat) == len(pts):
                        uv_vals, uv_mode = flat, "vtx"
                except Exception:  # noqa: BLE001
                    uv_vals = None
            # per-face material from GeomSubsets
            face_mat = {}
            for sub in UsdGeom.Subset.GetAllGeomSubsets(m):
                if sub.GetFamilyNameAttr().Get() not in (None, "", "materialBind"):
                    continue
                try:
                    b = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
                    if not (b and b.GetPrim().IsValid()):
                        continue
                    k = _material_record(b, layer_dir, mats)
                except Exception:  # noqa: BLE001
                    continue
                for fi in (sub.GetIndicesAttr().Get() or []):
                    face_mat[int(fi)] = k
            world = (Rm @ pts.T).T + tm
            if to_local is not None:
                Rc, tc = to_local; world = (Rc.T @ (world - tc).T).T
            V.append(world)
            if uv_vals is not None:
                VT.append(uv_vals)
            k0 = 0
            for fi, c in enumerate(counts):
                poly = idx[k0:k0 + c]
                for j in range(1, c - 1):
                    tri = (poly[0], poly[j], poly[j + 1])
                    F.append([t + v_off for t in tri])
                    if uv_mode == "fv":
                        FT.append([k0 + 0 + vt_off, k0 + j + vt_off, k0 + j + 1 + vt_off])
                    elif uv_mode == "vtx":
                        FT.append([t + vt_off for t in tri])
                    else:
                        FT.append([-1, -1, -1])
                    FM.append(face_mat.get(fi, mesh_mat))
                k0 += c
            v_off += len(pts)
            vt_off += len(uv_vals) if uv_vals is not None else 0
        else:
            got = _gprim_mesh(prim)
            if got is None:
                continue
            pts, tri = got
            world = (Rm @ pts.T).T + tm
            if to_local is not None:
                Rc, tc = to_local; world = (Rc.T @ (world - tc).T).T
            V.append(world)
            for t in tri:
                F.append([int(x) + v_off for x in t]); FT.append([-1, -1, -1]); FM.append(mesh_mat)
            v_off += len(pts)
        n_prims += 1
    if not V:
        return None
    V = np.vstack(V); VT = np.vstack(VT) if VT else np.zeros((0, 2))
    return V, VT, F, FT, FM, mats, n_prims


def _write(out_dir, name, usd_path, gathered):
    V, VT, F, FT, FM, mats, n_prims = gathered
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / f"{name}.obj"
    with open(obj_path, "w") as f:
        f.write(f"# exported from {usd_path}\nmtllib {name}.mtl\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in VT:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        # group faces by material so `usemtl` switches are few
        order = sorted(range(len(F)), key=lambda i: (FM[i] or ""))
        cur = object()
        for i in order:
            mk = FM[i]
            if mk != cur:
                f.write(f"usemtl {mats[mk]['name'] if mk in mats else 'default'}\n"); cur = mk
            a, b, c = F[i]; ta, tb, tc = FT[i]
            if ta >= 0:
                f.write(f"f {a+1}/{ta+1} {b+1}/{tb+1} {c+1}/{tc+1}\n")
            else:
                f.write(f"f {a+1} {b+1} {c+1}\n")
    with open(out_dir / f"{name}.mtl", "w") as f:
        f.write("newmtl default\nKd 0.6 0.6 0.6\n")
        for rec in mats.values():
            f.write(f"newmtl {rec['name']}\n")
            col = rec.get("diffuse_color") or [0.6, 0.6, 0.6]
            f.write(f"Kd {col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n")
            if rec.get("diffuse_texture"):
                f.write(f"map_Kd {rec['diffuse_texture']}\n")
    info = {"usd": usd_path, "name": name, "n_prims": n_prims, "n_verts": int(len(V)), "n_tris": int(len(F)),
            "n_uv": int(len(VT)), "aabb": [V.min(0).tolist(), V.max(0).tolist()],
            "materials": {rec["name"]: rec for rec in mats.values()}}
    json.dump(info, open(out_dir / f"{name}.materials.json", "w"), indent=1)
    return info


def export(usd_path, out_dir, name=None):
    """One object/fixture USD in its own stage frame (matches load_object_usd)."""
    from pxr import Usd, UsdGeom
    usd_path = str(usd_path)
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    g = _gather(root, UsdGeom.XformCache(), usd_path)
    if g is None:
        raise ValueError(f"no geometry in {usd_path}")
    return _write(out_dir, name or root.GetName(), usd_path, g)


def export_scene(usda_path, out_dir, world_prim=None):
    """Every child of the scene's world prim as its own OBJ in the child's local frame with the
    child's scale baked in; returns {child_name: info} with `c_model` (AABB centre of the local
    mesh), `center` and `rotation` matching load_scene_usda's Body for that child."""
    from pxr import Usd, UsdGeom
    from simready.io.usd_io import _rotation_from_affine
    usda_path = str(usda_path)
    stage = Usd.Stage.Open(usda_path)
    root = stage.GetPrimAtPath(world_prim) if world_prim else stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot().GetChildren()[0]
    xcache = UsdGeom.XformCache()
    out = {}
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xformable) and not child.GetChildren():
            continue
        Rc_aff, tc = _matrix_np(xcache.GetLocalToWorldTransform(child))
        Rc = _rotation_from_affine(Rc_aff)
        g = _gather(child, xcache, usda_path, to_local=(Rc, tc))
        if g is None:
            continue
        info = _write(out_dir, child.GetName(), usda_path, g)
        lo, hi = np.asarray(info["aabb"])
        c_model = 0.5 * (lo + hi)
        info.update({"c_model": c_model.tolist(), "center": (tc + Rc @ c_model).tolist(), "rotation": Rc.tolist(), "prim": str(child.GetPath())})
        json.dump(info, open(Path(out_dir) / f"{child.GetName()}.materials.json", "w"), indent=1)
        out[child.GetName()] = info
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("usd"); ap.add_argument("out_dir"); ap.add_argument("--name"); ap.add_argument("--scene", action="store_true")
    a = ap.parse_args()
    if a.scene:
        res = export_scene(a.usd, a.out_dir)
        for k, info in res.items():
            print(f"{k:22s} tris={info['n_tris']:7d} uv={info['n_uv']:7d} mats={len(info['materials'])} tex={sum(1 for m in info['materials'].values() if m.get('diffuse_texture'))} aabb={np.round(info['aabb'], 3).tolist()}")
    else:
        info = export(a.usd, a.out_dir, a.name)
        print(json.dumps({k: v for k, v in info.items() if k != "materials"}), "\nmaterials:", len(info["materials"]),
              [(m["name"], bool(m.get("diffuse_texture")), m.get("diffuse_color")) for m in info["materials"].values()][:6])
