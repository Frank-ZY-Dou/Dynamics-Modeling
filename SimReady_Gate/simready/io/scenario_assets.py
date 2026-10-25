"""Scenario 3D Collection assets: AI-generated objects of the Hugging Face dataset
scenario-labs/3d-collection (CC BY 4.0, credit "Scenario"), fetched, normalized and read as bodies.

The collection's meshes carry no physics and no real size (most generators normalize a mesh to about
two units), so `simready/data/scenario_assets.json` lists the objects the gate uses: for each, its GLB
in the dataset at a pinned commit with its sha256, its semantic class, one real dimension, its mass,
material, friction and restitution (estimates for the object's kind, not measurements) and, for an
object with a handle, the side it is on, and whether other objects can be put inside it.

`fetch` downloads an asset's GLB at that commit, checks it against the sha256, turns glTF's +y up to
+z up, scales it to the listed dimension, puts the origin at the centre of its footprint on its lowest
point, decimates it to at most MAX_FACES faces (measuring how far the decimated surface lies from the
original) and keeps `mesh.ply` with a record of all that (`.scenario_fetch.json`) under
SIMREADY_ASSETS_DIR. `body_from_scenario` makes a Body from it at a layout pose, with the physics in
meta as `body_from_package` does for SimReady packages. A repair never downloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
from functools import lru_cache
from pathlib import Path

import numpy as np

from .nvidia_assets import AssetError, HF, _get, assets_root, rest_rotation_for, _axis_word

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "scenario_assets.json"
RECORD = ".scenario_fetch.json"
MAX_FACES = 40_000          # a decimated copy above this: the full-mesh checks stay as fast as for the SimReady props
DEVIATION_SAMPLES = 20_000  # points of the original surface whose distance to the decimated one is measured
UP_TO_Z = {"+y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), "+z": np.eye(3)}


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def entry(name: str) -> dict:
    assets = manifest()["assets"]
    if name not in assets:
        raise AssetError(f"no Scenario asset '{name}' in {MANIFEST.name}; the assets are: {', '.join(sorted(assets))}")
    return assets[name]


def asset_dir(name: str) -> Path:
    return assets_root() / manifest()["repo"] / name


def _dimension(e: dict) -> tuple[str, float]:
    dims = [k for k in ("height_m", "width_m", "length_m") if k in e]
    if len(dims) != 1:
        raise AssetError(f"a Scenario asset needs exactly one of height_m, width_m, length_m; got {dims}")
    return dims[0][:-2], float(e[dims[0]])


def _extent(verts: np.ndarray, dim: str) -> float:
    ext = np.ptp(verts, axis=0)
    return float({"height": ext[2], "width": max(ext[0], ext[1]), "length": ext.max()}[dim])


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _glb_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of every mesh of a GLB scene, in the scene's frame (node transforms applied)."""
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    V, F, off = [], [], 0
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        if not hasattr(g, "faces") or not len(g.faces):
            continue
        V.append(trimesh.transform_points(np.asarray(g.vertices, dtype=np.float64), T))
        F.append(np.asarray(g.faces, dtype=np.int64) + off)
        off += len(g.vertices)
    if not V:
        raise AssetError(f"{path}: no triangle mesh in the GLB")
    return np.vstack(V), np.vstack(F)


def fetch(name: str, force: bool = False) -> Path:
    """Download an asset's GLB at the manifest's commit, check its sha256 and write the normalized,
    decimated mesh; returns the asset directory. An asset already fetched with the same file is not
    processed again."""
    import trimesh
    e, m = entry(name), manifest()
    dest = asset_dir(name)
    rec_path = dest / RECORD
    if rec_path.exists() and not force:
        rec = json.loads(rec_path.read_text())
        if rec.get("sha256") == e["sha256"] and rec.get("revision") == m["revision"] and (dest / "mesh.ply").exists():
            return dest
    dest.mkdir(parents=True, exist_ok=True)
    src = dest / "source.glb"
    if not (src.exists() and _sha256(src) == e["sha256"]):
        body, _ = _get(f"{HF}/datasets/{m['repo']}/resolve/{m['revision']}/{urllib.parse.quote(e['path'])}", timeout=900)
        tmp = src.with_name(src.name + ".part")
        tmp.write_bytes(body)
        if _sha256(tmp) != e["sha256"]:
            tmp.unlink()
            raise AssetError(f"{name}: {e['path']} does not match the sha256 of the manifest")
        os.replace(tmp, src)
    V, F = _glb_mesh(src)
    up = m.get("up", "+y")
    if up not in UP_TO_Z:
        raise AssetError(f"unsupported up axis '{up}' in {MANIFEST.name}")
    mesh = trimesh.Trimesh(V @ UP_TO_Z[up].T, F, process=False)
    mesh.merge_vertices()
    faces_source = int(len(mesh.faces))
    dim, metres = _dimension(e)
    scale = metres / _extent(mesh.vertices, dim)
    v = mesh.vertices * scale
    lo, hi = v.min(0), v.max(0)
    v = v - np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])     # origin: centre of the footprint, lowest point
    full = trimesh.Trimesh(v, mesh.faces, process=False)
    out, deviation = full, None
    if len(full.faces) > MAX_FACES:
        try:
            import fast_simplification  # noqa: F401  trimesh decimates through it
        except ImportError as err:
            raise AssetError(f"{name}: {len(full.faces)} faces need decimating to {MAX_FACES}, which needs fast-simplification "
                             f"(pip install fast-simplification)") from err
        out = full.simplify_quadric_decimation(face_count=MAX_FACES)
        _, dist, _ = trimesh.proximity.closest_point(out, full.sample(DEVIATION_SAMPLES, seed=0))
        deviation = {"mean_m": float(np.mean(dist)), "max_m": float(np.max(dist))}
    out.export(str(dest / "mesh.ply"))
    rec = {"repo": m["repo"], "revision": m["revision"], "name": name, "id": e["id"], "path": e["path"], "sha256": e["sha256"],
           "generator": e.get("generator"), "license": m["license"], "credit": m["credit"], "up": up, "scale": scale,
           "dimension": {dim: metres}, "faces_source": faces_source, "faces": int(len(out.faces)), "decimation_deviation": deviation,
           "extents_m": [float(x) for x in np.ptp(out.vertices, axis=0)], "watertight": bool(out.is_watertight),
           "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec_path.write_text(json.dumps(rec, indent=1))
    return dest


def read_asset(name: str) -> dict:
    """The fetched mesh's path and record and the manifest's physics, class and handle of an asset."""
    d = asset_dir(name)
    if not (d / RECORD).exists():
        raise FileNotFoundError(f"Scenario asset {name} is not in {d.parent}; run: python -m simready.cli assets fetch scenario:{name}")
    e, rec = entry(name), json.loads((d / RECORD).read_text())
    if rec.get("sha256") != e["sha256"]:
        raise AssetError(f"Scenario asset {name} was fetched from another file than the manifest lists; fetch it again")
    return {"mesh": d / "mesh.ply", "record": rec, "label": e["label"], "handle": e.get("handle"), "material": e["material"],
            "container": bool(e.get("container", False)),
            "physics": {"mass": float(e["mass_kg"]), "density": None, "static_friction": float(e["static_friction"]),
                        "dynamic_friction": float(e["dynamic_friction"]), "restitution": float(e["restitution"]), "rigid_bodies": 1}}


def body_from_scenario(name: str, asset: str, position, yaw_deg: float = 0.0, fixed: bool = False, tags=None):
    """A Body for a fetched Scenario asset at a layout pose: the asset's origin (the centre of its
    footprint on its lowest point) at `position`, turned by `yaw_deg` about z, with the manifest's
    physics in meta['physics'], the class in meta['label'] and the side its handle is on in
    meta['grasp_axis']. An asset that cannot stand as authored is re-expressed in its resting frame,
    by the rule SimReady packages follow, with its origin again at the centre of its footprint on its
    lowest point."""
    import trimesh
    from ..scene.model import Body
    from ..repair.upright_s4r import rotz
    info = read_asset(asset)
    m = trimesh.load(str(info["mesh"]), process=False, force="mesh")
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    handle = None
    if info["handle"]:
        k = "xyz".index(info["handle"][1])
        handle = np.zeros(3)
        handle[k] = (1.0 if info["handle"][0] == "+" else -1.0) * 0.5 * float(np.ptp(verts[:, k]))
        handle += 0.5 * (verts.min(0) + verts.max(0))
    R_rest = None if fixed else rest_rotation_for(verts, None)
    if R_rest is not None:
        verts = verts @ R_rest.T
        lo, hi = verts.min(0), verts.max(0)
        origin = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
        verts = verts - origin
        if handle is not None:
            handle = R_rest @ handle - origin
    lo, hi = verts.min(0), verts.max(0)
    c_model = 0.5 * (lo + hi)
    R = rotz(np.radians(yaw_deg))
    meta = {"c_model": c_model, "physics": dict(info["physics"]), "label": info["label"],
            "scenario": {k: info["record"][k] for k in ("name", "id", "repo", "revision", "license", "credit")}}
    if R_rest is not None:
        meta["rest_rotation"] = R_rest
    if handle is not None:
        word = _axis_word(handle - c_model, hi - lo)
        if word:
            meta["grasp_axis"] = word
    b = Body(name=name, verts=verts - c_model, faces=faces, center=np.asarray(position, dtype=np.float64) + R @ c_model, rotation=R,
             fixed=fixed, tags=set(tags or ()), source=str(info["mesh"]), meta=meta)
    if info["container"]:
        b.tags.add("container")
    return b
