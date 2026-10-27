"""Export a RoboCasa MJCF object's VISUAL geometry to OBJ (+ materials JSON) for rendering.

Geometry, texture coordinates and the geom poses come from the MuJoCo-compiled model
(simready.io.mjcf_io.object_geoms), so the render mesh sits in exactly the root-body frame the
solver uses for the collision geometry; the body's `c_model` (AABB centre of the COLLISION mesh)
must be supplied by the caller so both share one reference frame. Materials come from the compiled model and are
mapped as MuJoCo's own physically based renderer draws them (_record); texture files from the XML's <texture>
elements (relative to the XML directory). A texture not on disk, or a material feature that mapping does not
cover, is refused.

Usage: python viz/mjcf_to_obj.py <model.xml> <out_dir> --name NAME
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simready  # noqa: E402,F401
from simready.io.mjcf_io import object_geoms  # noqa: E402


# MuJoCo's own physically based renderer (Filament: model_renderables.cc and phong_*.mat, google-deepmind/mujoco)
# draws a material without PBR textures in the specular-glossiness model: base colour = the geom's rgba when it is not
# the default (0.5, 0.5, 0.5, 1), else the material's, times the material's RGB texture; specular colour (F0) =
# SPECULAR_MULT x specular; glossiness = SHININESS_MULT x shininess, roughness = 1 - glossiness; no metalness. A geom
# without a material takes specular and shininess 0.5. The F0 is rendered as the IOR that gives it.
SPECULAR_MULT, SHININESS_MULT = 0.2, 0.1
TEXTURE_ROLES = ("RGB", "RGBA", "NORMAL", "ORM", "METALLIC", "ROUGHNESS", "OCCLUSION", "OPACITY", "EMISSIVE")


def _texture_files(xml_path):
    """texture name -> (its file, relative to the XML directory; the colour space the XML declares, or None), from the
    XML's <texture> elements."""
    root = ET.parse(xml_path).getroot(); base = Path(xml_path).parent
    out = {}
    for t in root.iter("texture"):
        f = t.get("file")
        if f:
            out[t.get("name")] = (str((Path(f) if Path(f).is_absolute() else base / f).resolve()), t.get("colorspace"))
    return out


# how a colour texture's values are decoded: as the XML declares it, and when it declares nothing as sRGB, the encoding
# of 8-bit colour images (MuJoCo 3.17 compiles an undeclared texture as linear; the record keeps both)
DECLARED_COLORSPACE = {"sRGB": "sRGB", "linear": "raw", None: "sRGB"}


def _record(model, g, tex_files, xml_path):
    """The material record of visual geom g, from the compiled model, as MuJoCo's own renderer draws it; what that
    mapping does not cover is refused."""
    import math
    import mujoco
    matid, rgba = int(model.geom_matid[g]), [float(x) for x in model.geom_rgba[g]]
    texid = -1
    if matid >= 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid) or f"material{matid}"
        if rgba == [0.5, 0.5, 0.5, 1.0]:
            rgba = [float(x) for x in model.mat_rgba[matid]]
        else:
            name = f"{name}_geom{g}"
        spec, shin = float(model.mat_specular[matid]), float(model.mat_shininess[matid])
        if model.mat_metallic[matid] >= 0 or model.mat_roughness[matid] >= 0:
            raise ValueError(f"{xml_path}: material {name} sets metallic or roughness (MuJoCo's PBR path), not carried into the export")
        if model.mat_reflectance[matid] > 0 or model.mat_emission[matid] > 0:
            raise ValueError(f"{xml_path}: material {name} reflects ({model.mat_reflectance[matid]}) or emits ({model.mat_emission[matid]}), "
                             "not carried into the export")
        tex = {r: int(model.mat_texid[matid, int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_" + r))]) for r in TEXTURE_ROLES}
        other = [r for r, t in tex.items() if t >= 0 and r != "RGB"]
        if other:
            raise ValueError(f"{xml_path}: material {name} has {other} textures, not carried into the export")
        texid = tex["RGB"]
        repeat = [float(x) for x in model.mat_texrepeat[matid]]
        if texid >= 0 and model.mat_texuniform[matid]:
            raise ValueError(f"{xml_path}: material {name} repeats its texture per metre (texuniform), not carried into the export")
    else:
        name, spec, shin, repeat = f"geom{g}", 0.5, 0.5, [1.0, 1.0]
    if rgba[3] < 1.0:
        raise ValueError(f"{xml_path}: geom {g} is transparent (alpha {rgba[3]}), not carried into the export")
    f0 = SPECULAR_MULT * spec
    # the record keys of viz/blender_render.py; the texture coordinates are uv * texrepeat (model_renderables.cc)
    rec = {"name": name, "diffuse_tint": [1.0, 1.0, 1.0], "albedo_brightness": 1.0, "albedo_add": 0.0, "albedo_desaturation": 0.0,
           "roughness": 1.0 - SHININESS_MULT * shin, "roughness_source": "constant", "metallic": 0.0, "metallic_source": "constant",
           "texture_combine": "replace", "ao_source": "one", "ao_to_diffuse": 0.0, "specular": 0.5,
           "ior": (1 + math.sqrt(f0)) / (1 - math.sqrt(f0)), "texture_scale": [1.0, 1.0], "texture_translate": [0.0, 0.0],
           "mujoco": {"rgba": rgba, "specular": spec, "shininess": shin}}
    if texid >= 0:
        if int(model.tex_type[texid]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
            raise ValueError(f"{xml_path}: material {name} has a texture that is not 2D, not carried into the export")
        tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, texid)
        if tname not in tex_files:
            raise ValueError(f"{xml_path}: texture {tname} of material {name} has no file in the XML")
        tfile, declared = tex_files[tname]
        if declared not in DECLARED_COLORSPACE:
            raise ValueError(f"{xml_path}: texture {tname} declares the colour space {declared!r}, not carried into the export")
        compiled = {int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB): "sRGB", int(mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR): "linear",
                    int(mujoco.mjtColorSpace.mjCOLORSPACE_AUTO): "auto"}[int(model.tex_colorspace[texid])]
        rec["mujoco"]["texture_colorspace"] = {"declared": declared, "compiled": compiled}
        if not os.path.exists(tfile):
            rec["missing"] = [tfile]
        else:
            rec["diffuse_texture"], rec["diffuse_tint"] = tfile, rgba[:3]
            rec["diffuse_texture_colorspace"] = DECLARED_COLORSPACE[declared]
        rec["texture_scale"] = repeat
    else:
        rec["diffuse_color"] = rgba[:3]
    return rec


def export_object(xml_path, out_dir, name, c_model=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    geoms = object_geoms(xml_path, collision=False)
    import mujoco
    from simready.io.mjcf_io import compile_object
    model, data, root = compile_object(xml_path)
    index = {}
    for g in range(model.ngeom):                    # object_geoms' names: the geom's own, else geom<g>
        key = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        if key in index:
            raise ValueError(f"{xml_path}: two geoms are named {key}")
        index[key] = g
    tex_files = _texture_files(xml_path)
    V, VT, F, FT, FM = [], [], [], [], []
    recs = {}
    v_off = vt_off = 0
    for gname, v, f, tc, ftc in geoms:
        if gname == "reg_bbox":
            continue
        rec = _record(model, index[gname], tex_files, xml_path)
        if rec["name"] in recs and recs[rec["name"]] != rec:
            raise ValueError(f"{xml_path}: two records named {rec['name']}")
        mname = rec["name"]; recs[mname] = rec
        V.append(v)
        has_uv = tc is not None and ftc is not None and len(tc) > 0
        if has_uv:
            VT.append(tc)
        if rec.get("diffuse_texture") and not has_uv:
            raise ValueError(f"{xml_path}: geom {gname} has a textured material and no texture coordinates")
        for k in range(len(f)):
            F.append([int(x) + v_off for x in f[k]])
            FT.append([int(x) + vt_off for x in ftc[k]] if has_uv else [-1, -1, -1])
            FM.append(mname)
        v_off += len(v); vt_off += len(tc) if has_uv else 0
    if not V:
        raise ValueError(f"no visual geometry in {xml_path}")
    V = np.vstack(V); VT = np.vstack(VT) if VT else np.zeros((0, 2))
    with open(out_dir / f"{name}.obj", "w") as fh:
        fh.write(f"# visual geometry of {xml_path} (MuJoCo-compiled, root-body frame)\nmtllib {name}.mtl\n")
        for p in V:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in VT:                     # MuJoCo keeps an OBJ's texture coordinates as (u, 1 - v); the file takes (u, v)
            fh.write(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}\n")
        cur = None
        for tri, tt, mk in zip(F, FT, FM):
            if mk != cur:
                fh.write(f"usemtl {mk}\n"); cur = mk
            if tt[0] >= 0:
                fh.write(f"f {tri[0]+1}/{tt[0]+1} {tri[1]+1}/{tt[1]+1} {tri[2]+1}/{tt[2]+1}\n")
            else:
                fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    with open(out_dir / f"{name}.mtl", "w") as fh:
        for mname, rec in recs.items():
            col = rec.get("diffuse_color") or [1.0, 1.0, 1.0]
            fh.write(f"newmtl {mname}\nKd {col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n")
            if rec.get("diffuse_texture"):
                fh.write(f"map_Kd {rec['diffuse_texture']}\n")
    info = {"xml": str(xml_path), "name": name, "n_verts": int(len(V)), "n_tris": int(len(F)), "n_uv": int(len(VT)),
            "aabb": [V.min(0).tolist(), V.max(0).tolist()], "materials": recs,
            "c_model": list(map(float, c_model)) if c_model is not None else None}
    json.dump(info, open(out_dir / f"{name}.materials.json", "w"), indent=1)
    missing = {n: r["missing"] for n, r in recs.items() if r.get("missing")}
    if missing:
        raise RuntimeError(f"{name} ({xml_path}): textures not on disk: {missing}")
    return info


def export_box(out_dir, name, extents, center, color=(0.85, 0.85, 0.85), roughness=0.6):
    """A plain box (counter, region marker) as OBJ in a frame centred on `center` (c_model = 0)."""
    import trimesh
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    m = trimesh.creation.box(extents=extents)
    with open(out_dir / f"{name}.obj", "w") as fh:
        fh.write(f"mtllib {name}.mtl\n")
        for p in m.vertices:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        fh.write(f"usemtl {name}\n")
        for t in m.faces:
            fh.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    with open(out_dir / f"{name}.mtl", "w") as fh:
        fh.write(f"newmtl {name}\nKd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
    info = {"name": name, "n_verts": 8, "n_tris": 12, "n_uv": 0, "aabb": [m.bounds[0].tolist(), m.bounds[1].tolist()],
            "materials": {name: {"name": name, "diffuse_color": list(color), "diffuse_tint": [1.0, 1.0, 1.0], "albedo_brightness": 1.0,
                                 "albedo_add": 0.0, "albedo_desaturation": 0.0, "roughness": roughness, "roughness_source": "constant",
                                 "metallic": 0.0, "metallic_source": "constant", "texture_combine": "replace", "ao_source": "one",
                                 "ao_to_diffuse": 0.0, "specular": 0.5, "ior": 1.5, "texture_scale": [1.0, 1.0],
                                 "texture_translate": [0.0, 0.0]}},
            "c_model": [0.0, 0.0, 0.0]}
    json.dump(info, open(out_dir / f"{name}.materials.json", "w"), indent=1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("xml"); ap.add_argument("out_dir"); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    info = export_object(a.xml, a.out_dir, a.name)
    print({k: v for k, v in info.items() if k != "materials"}, [(m, bool(r.get("diffuse_texture"))) for m, r in info["materials"].items()])
