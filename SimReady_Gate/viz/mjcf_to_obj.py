"""Export a RoboCasa MJCF object's VISUAL geometry to OBJ (+ materials JSON) for rendering.

Geometry, texture coordinates and the geom poses come from the MuJoCo-compiled model
(simready.io.mjcf_io.object_geoms), so the render mesh sits in exactly the root-body frame the
solver uses for the collision geometry; the body's `c_model` (AABB centre of the COLLISION mesh)
must be supplied by the caller so both share one reference frame. Textures are read from the
XML's <material>/<texture> elements (files relative to the XML directory).

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


def _materials(xml_path):
    """geom name -> material record (diffuse texture path or rgba colour)."""
    root = ET.parse(xml_path).getroot(); base = Path(xml_path).parent
    tex = {}
    for t in root.iter("texture"):
        f = t.get("file")
        if f:
            fp = Path(f) if Path(f).is_absolute() else base / f
            if fp.exists():
                tex[t.get("name")] = str(fp.resolve())
    mats = {}
    for m in root.iter("material"):
        rec = {"name": m.get("name")}
        if m.get("texture") in tex:
            rec["diffuse_texture"] = tex[m.get("texture")]
        if m.get("rgba"):
            rec["diffuse_color"] = [float(x) for x in m.get("rgba").split()][:3]
        if m.get("shininess"):
            rec["roughness"] = max(0.05, 1.0 - float(m.get("shininess")))
        mats[m.get("name")] = rec
    by_geom, by_mesh = {}, {}
    for g in root.iter("geom"):
        if g.get("material") in mats:
            if g.get("name"):
                by_geom[g.get("name")] = mats[g.get("material")]
            if g.get("mesh"):
                by_mesh[g.get("mesh")] = mats[g.get("material")]
    return by_geom, by_mesh, mats


def export_object(xml_path, out_dir, name, c_model=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    geoms = object_geoms(xml_path, collision=False)
    by_geom, by_mesh, mats = _materials(xml_path)
    # MuJoCo's geom -> mesh names, to map unnamed geoms to their XML material
    import mujoco
    from simready.io.mjcf_io import compile_object
    model, data, root = compile_object(xml_path)
    mesh_of_geom = {}
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_of_geom[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
    V, VT, F, FT, FM = [], [], [], [], []
    recs = {}
    v_off = vt_off = 0
    for gname, v, f, tc, ftc in geoms:
        if gname == "reg_bbox":
            continue
        rec = by_geom.get(gname) or by_mesh.get(mesh_of_geom.get(gname)) or {"name": "default", "diffuse_color": [0.6, 0.6, 0.6]}
        mname = rec.get("name", "default"); recs[mname] = rec
        V.append(v)
        has_uv = tc is not None and ftc is not None and len(tc) > 0
        if has_uv:
            VT.append(tc)
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
        for t in VT:
            fh.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
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
            col = rec.get("diffuse_color") or [0.6, 0.6, 0.6]
            fh.write(f"newmtl {mname}\nKd {col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n")
            if rec.get("diffuse_texture"):
                fh.write(f"map_Kd {rec['diffuse_texture']}\n")
    info = {"xml": str(xml_path), "name": name, "n_verts": int(len(V)), "n_tris": int(len(F)), "n_uv": int(len(VT)),
            "aabb": [V.min(0).tolist(), V.max(0).tolist()], "materials": recs,
            "c_model": list(map(float, c_model)) if c_model is not None else None}
    json.dump(info, open(out_dir / f"{name}.materials.json", "w"), indent=1)
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
            "materials": {name: {"name": name, "diffuse_color": list(color), "roughness": roughness}}, "c_model": [0.0, 0.0, 0.0]}
    json.dump(info, open(out_dir / f"{name}.materials.json", "w"), indent=1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("xml"); ap.add_argument("out_dir"); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    info = export_object(a.xml, a.out_dir, a.name)
    print({k: v for k, v in info.items() if k != "materials"}, [(m, bool(r.get("diffuse_texture"))) for m, r in info["materials"].items()])
