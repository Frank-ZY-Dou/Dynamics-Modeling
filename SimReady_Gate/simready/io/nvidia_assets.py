"""NVIDIA SimReady assets: search (SimReady Central), fetch (Hugging Face), read (geometry, physics,
semantics).

A SimReady package is a folder of a Hugging Face dataset (nvidia/simready-assets, and the other
datasets SimReady Central indexes):

    <package>/com.nvidia.simready.packaging.json                 package id, license, content hash
    <package>/.metadata/com.nvidia.simready.packaging.bom.json   every file of the package with its sha256
    <package>/.metadata/com.nvidia.simready.root_usds.json       the root USD(s)
    <package>/simready_usd/sm_<package>_01.usd                   root: PhysX / MuJoCo / Newton variants
    <package>/simready_usd/payloads/...                          geometry, instances, physics, materials

`fetch` downloads a package without its textures (the gate reads geometry and physics), checks
every file against the bill of materials and keeps it under SIMREADY_ASSETS_DIR, with a record of
the dataset revision it came from. `read_package` returns the asset's physics: mass, centre of
mass, inertia, density, friction and restitution from the UsdPhysics schemas, its semantic labels,
its documentation string and the grasp hint SimReady props carry. `body_from_package` makes a
Body from it at a layout pose, with the physics in `meta` for the settle test and the export.
`search` asks SimReady Central's hybrid search and returns packages with their size, mass and
license. Nothing here holds a credential; the datasets are public.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_REPO = "nvidia/simready-assets"
CENTRAL = "https://nvidia-simready-central.hf.space"
HF = "https://huggingface.co"
ASSETS_ROOT = os.environ.get("SIMREADY_ASSETS_DIR", str(Path(__file__).resolve().parents[3] / "ext" / "simready-assets")).rstrip("/")
RECORD = ".simready_fetch.json"
SKIP_DIRS = ("/Textures/", "/source_assets/", "/.thumbs/")   # images and MDL sources: not read by the gate
# words of a package name or semantic class that make an asset a container or a support surface
CONTAINER_WORDS = ("bowl", "cup", "mug", "basket", "bin", "bucket", "pail", "tray", "pot", "pan", "jar", "crate", "caddy", "box_open")
SUPPORT_WORDS = ("table", "workbench", "counter", "shelf", "pallet", "desk", "cart")


class AssetError(RuntimeError):
    pass


def assets_root() -> Path:
    return Path(os.environ.get("SIMREADY_ASSETS_DIR", ASSETS_ROOT))


def package_dir(package: str, repo: str = DEFAULT_REPO) -> Path:
    return assets_root() / repo / package


def _get(url: str, timeout: float = 120.0, data: bytes | None = None, headers: dict | None = None, tries: int = 4):
    """Body and headers of an HTTP request, retried with backoff on network errors and 5xx."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise AssetError(f"{url}: HTTP {e.code}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2.0 * (k + 1))
    raise AssetError(f"{url}: {type(last).__name__}: {last}")


def resolve_revision(repo: str = DEFAULT_REPO, revision: str = "main") -> str:
    """The commit a branch or tag of the dataset points at (a commit is returned unchanged)."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    body, _ = _get(f"{HF}/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}")
    return json.loads(body)["sha"]


def list_files(package: str, repo: str = DEFAULT_REPO, revision: str = "main") -> list[dict]:
    """Every file of a package at a revision: [{path, size, sha256 (LFS files only)}]."""
    url = f"{HF}/api/datasets/{repo}/tree/{revision}/{urllib.parse.quote(package)}?recursive=true"
    out = []
    while url:
        body, headers = _get(url)
        for x in json.loads(body):
            if x.get("type") == "file":
                out.append({"path": x["path"], "size": x.get("size"), "sha256": (x.get("lfs") or {}).get("oid")})
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or headers.get("link", ""))
        url = m.group(1) if m else None
    if not out:
        raise AssetError(f"no package '{package}' in {repo}@{revision}")
    return out


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(package: str, repo: str = DEFAULT_REPO, revision: str = "main", textures: bool = False, force: bool = False) -> Path:
    """Download a package (without textures unless asked) into the cache and check it against its
    bill of materials; returns the package directory. A package already fetched at the same commit
    is not downloaded again."""
    if not re.fullmatch(r"[\w\-.]+", package) or package.startswith("."):
        raise AssetError(f"bad package name '{package}'")
    sha = resolve_revision(repo, revision)
    dest = package_dir(package, repo)
    rec_path = dest / RECORD
    if rec_path.exists() and not force:
        rec = json.loads(rec_path.read_text())
        if rec.get("revision") == sha and (rec.get("textures") or not textures):
            return dest
    files = list_files(package, repo, sha)
    wanted = [f for f in files if textures or not any(s in "/" + f["path"] for s in SKIP_DIRS)]
    dest.mkdir(parents=True, exist_ok=True)
    got = {}
    for f in wanted:
        rel = f["path"][len(package) + 1:]
        if ".." in Path(rel).parts:
            raise AssetError(f"unsafe path in the package listing: {f['path']}")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and f.get("sha256") and _sha256(out) == f["sha256"]):
            body, _ = _get(f"{HF}/datasets/{repo}/resolve/{sha}/{urllib.parse.quote(f['path'])}", timeout=600)
            tmp = out.with_name(out.name + ".part")
            tmp.write_bytes(body)
            os.replace(tmp, out)
        got[rel] = _sha256(out)
    bom = _read_json(dest / ".metadata" / "com.nvidia.simready.packaging.bom.json")
    mismatch = []
    for item in (bom or {}).get("items", []):
        rel = item["relative_path"]
        if rel in got and item.get("hash", {}).get("sha256") not in (None, got[rel]):
            mismatch.append(rel)
    if mismatch:
        raise AssetError(f"{package}: files differ from the bill of materials: {mismatch}")
    pkg = _read_json(dest / "com.nvidia.simready.packaging.json") or {}
    rec = {"repo": repo, "package": package, "revision": sha, "package_id": pkg.get("package_id"), "license": pkg.get("license"),
           "textures": textures, "bom_checked": bom is not None, "files": got, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec_path.write_text(json.dumps(rec, indent=1))
    return dest


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def root_usd(pkg: Path) -> Path:
    """The package's root USD (the first entry of root_usds.json, else the USD at simready_usd/)."""
    pkg = Path(pkg)
    roots = _read_json(pkg / ".metadata" / "com.nvidia.simready.root_usds.json")
    if roots and roots.get("entries"):
        p = pkg / roots["entries"][0]
        if p.exists():
            return p
    cand = sorted(list((pkg / "simready_usd").glob("*.usd")) + list((pkg / "simready_usd").glob("*.usda")))
    if not cand:
        raise AssetError(f"{pkg}: no root USD")
    return cand[0]


def _attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def read_package(pkg) -> dict:
    """Physics, semantics and the grasp hint of a fetched package (geometry is read by the USD
    loader). Mass properties are summed over the package's mesh prims in the root prim's frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    pkg = Path(pkg)
    path = root_usd(pkg)
    stage = Usd.Stage.Open(str(path))
    errors = stage.GetCompositionErrors()
    if errors:
        raise AssetError(f"{path}: unresolved USD composition ({len(errors)} missing layers); fetch the package again")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    labels = {}
    for a in root.GetAttributes():
        n = a.GetName()
        if n.startswith("semantics:labels:") and a.Get() is not None:
            labels[n.split(":", 2)[2]] = [str(t) for t in a.Get()]
    masses, coms, bodies, materials = [], [], 0, {}
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
        if p.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(p)
            materials[p.GetName()] = {k: (float(v) if v is not None else None) for k, v in (
                ("density", m.GetDensityAttr().Get()), ("static_friction", m.GetStaticFrictionAttr().Get()),
                ("dynamic_friction", m.GetDynamicFrictionAttr().Get()), ("restitution", m.GetRestitutionAttr().Get()))}
        if p.HasAPI(UsdPhysics.MassAPI) and p.IsA(UsdGeom.Xformable):
            ma = UsdPhysics.MassAPI(p)
            mass = ma.GetMassAttr().Get()
            if mass:
                com = ma.GetCenterOfMassAttr().Get()
                local = np.zeros(3) if com is None else np.array([float(c) for c in com])
                m4 = cache.GetLocalToWorldTransform(p)
                R = np.array([[m4[i][j] for j in range(3)] for i in range(3)]).T
                t = np.array([m4[3][j] for j in range(3)])
                masses.append(float(mass)); coms.append(R @ local + t)
    mass = float(sum(masses)) if masses else None
    com = (np.sum([m * c for m, c in zip(masses, coms)], axis=0) / mass).tolist() if masses else None
    friction = [v for v in materials.values() if v.get("static_friction") is not None]
    grasp = None
    g = stage.GetPrimAtPath(root.GetPath().AppendPath("Grasp/grasp_identifier"))
    if g and g.IsValid():
        loc, nrm = _attr(g, "intersection_location"), _attr(g, "intersection_normal")
        if loc is not None:
            grasp = {"location": [float(x) for x in loc], "normal": None if nrm is None else [float(x) for x in nrm]}
    layer_meta = {}
    for layer in stage.GetLayerStack():
        d = dict(layer.customLayerData or {})
        if d.get("asset_name"):
            layer_meta = {k: d[k] for k in ("asset_name", "asset_type", "profile", "asset_dir") if k in d}
            break
    pkg_json = _read_json(pkg / "com.nvidia.simready.packaging.json") or {}
    return {"root_usd": str(path), "package_id": pkg_json.get("package_id"), "license": pkg_json.get("license"),
            "labels": labels, "documentation": _attr(root, "omni:simready:documentation"), "rigid_bodies": bodies,
            "mass": mass, "com": com, "materials": materials,
            "static_friction": friction[0]["static_friction"] if friction else None,
            "dynamic_friction": friction[0].get("dynamic_friction") if friction else None,
            "restitution": friction[0].get("restitution") if friction else None,
            "density": friction[0].get("density") if friction else None,
            "grasp": grasp, **layer_meta}


TIPPING_DEG = 12.0     # the rule of simready/data/asset_rest_orientations.json: a pose that tips below this does not stand


def tipping_deg(verts: np.ndarray, com) -> float:
    """Tilt at which the body, resting on its lowest face as authored, tips over: the angle at its
    centre of mass between the vertical and the nearest edge of its bounding footprint."""
    lo, hi = verts.min(0), verts.max(0)
    c = np.asarray(com if com is not None else 0.5 * (lo + hi), dtype=np.float64)
    reach = min(c[0] - lo[0], hi[0] - c[0], c[1] - lo[1], hi[1] - c[1])
    return float(np.degrees(np.arctan2(max(reach, 0.0), max(c[2] - lo[2], 1e-9))))


def rest_rotation_for(verts: np.ndarray, com) -> np.ndarray | None:
    """None when the authored pose stands (tipping angle of at least TIPPING_DEG); otherwise the
    quarter turn that rests the asset on the side it is most stable on, as RoboLab's table does for
    its catalog."""
    from .asset_rest import rotation_to_up
    if tipping_deg(verts, com) >= TIPPING_DEG:
        return None
    best, best_tip = None, -1.0
    for axis in ("+x", "-x", "+y", "-y", "-z"):
        R = rotation_to_up(axis)
        tip = tipping_deg(verts @ R.T, None if com is None else R @ np.asarray(com))
        if tip > best_tip + 1e-9:
            best, best_tip = R, tip
    return best


def _axis_word(d: np.ndarray, ext: np.ndarray) -> str | None:
    """'+x' / '-x' / '+y' / '-y': the side of the body a point lies on, when it lies clearly off-centre."""
    k = int(np.argmax(np.abs(d[:2]) / np.maximum(ext[:2], 1e-9)))
    if abs(d[k]) < 0.25 * ext[k]:
        return None
    return ("+" if d[k] > 0 else "-") + "xy"[k]


def body_from_package(name: str, pkg, position, yaw_deg: float = 0.0, fixed: bool = False, tags=None):
    """A Body for a fetched package at a layout pose: the root prim's origin at `position`, turned
    by `yaw_deg` about z, with the package's physics in meta['physics'], the centre of mass relative
    to the reference centre in meta['com'], the semantic class in meta['label'] and the side the
    grasp hint points at in meta['grasp_axis']."""
    from .usd_io import body_from_object_usd
    info = read_package(pkg)
    tags = set(tags or ())
    b = body_from_object_usd(name, info["root_usd"], position, yaw_deg=yaw_deg, fixed=fixed, tags=tags)
    c_model = np.asarray(b.meta["c_model"], dtype=np.float64)
    com = None if info["com"] is None else np.asarray(info["com"], dtype=np.float64)
    R_rest = None if fixed else rest_rotation_for(b.verts + c_model, com)
    if R_rest is not None:
        # an authored pose that cannot stand (a cutting board on its edge): the asset is re-expressed
        # in its resting frame, as simready.io.asset_rest does for RoboLab's catalog
        v = (b.verts + c_model) @ R_rest.T
        lo, hi = v.min(0), v.max(0)
        c_model = 0.5 * (lo + hi)
        b.verts = v - c_model
        b.center = np.asarray(position, dtype=np.float64) + b.rotation @ c_model
        b.meta["c_model"] = c_model
        b.meta["rest_rotation"] = R_rest
        com = None if com is None else R_rest @ com
        if info["grasp"]:
            info["grasp"]["location"] = (R_rest @ np.asarray(info["grasp"]["location"])).tolist()
    ext = np.ptp(b.verts, axis=0)
    phys = {k: info[k] for k in ("mass", "density", "static_friction", "dynamic_friction", "restitution", "rigid_bodies")}
    b.meta["physics"] = phys
    if com is not None:
        b.meta["com"] = com - c_model
    label = (info["labels"].get("wikidata_class") or info["labels"].get("class") or [None])[0]
    if label:
        b.meta["label"] = label
    if info["grasp"]:
        word = _axis_word(np.asarray(info["grasp"]["location"]) - c_model, ext)
        if word:
            b.meta["grasp_axis"] = word
    words = (Path(pkg).name + " " + (label or "")).lower()
    if any(w in words for w in CONTAINER_WORDS):
        b.tags.add("container")
    if fixed and any(w in words for w in SUPPORT_WORDS):
        b.tags.update({"support", "fixture"})
    b.meta["package"] = {"name": Path(pkg).name, "package_id": info["package_id"], "license": info["license"],
                         "revision": (_read_json(Path(pkg) / RECORD) or {}).get("revision")}
    return b


def search(query: str, limit: int = 10, repo: str | None = DEFAULT_REPO, max_size: float | None = None,
           max_height: float | None = None, max_mass: float | None = None) -> list[dict]:
    """SimReady Central's hybrid (text + SigLIP2) search: [{package, repo, revision, root_usd, size_m,
    mass_kg, license, category, caption}], best first. `repo` keeps the hits of one dataset (None:
    every dataset Central indexes); `max_size` bounds every side, `max_height` the vertical one."""
    body = {"hybrid_text_query": query, "vector_queries": [{"field_name": "siglip2-embedding.embedding", "query_type": "text", "query": query}],
            "file_extension_include": "usd*", "return_images": False, "limit": int(limit) * (4 if repo else 1),
            "return_usd_dimensions": True, "return_usd_properties": True, "return_vision_generated_metadata": True,
            "deduplicate_by_hash": True}
    for key, val in (("max_bbox_x", max_size), ("max_bbox_y", max_size), ("max_bbox_z", max_height if max_height is not None else max_size)):
        if val is not None:
            body[key] = float(val)
    if max_mass is not None:
        body["filter_by_properties_numeric"] = f"simready_metadata_mass<={float(max_mass)}"
    raw, _ = _get(f"{CENTRAL}/search_hybrid", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = []
    for hit in json.loads(raw).get("hits", []):
        src = hit.get("source", {})
        m = re.match(r"hf://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([0-9a-f]+)/([^/]+)/(.+)$", src.get("path", ""))
        if not m or (repo and m.group(1) != repo):
            continue
        props = {p["name"]: p.get("value_numeric", p.get("value")) for p in src.get("usd_properties", [])}
        dims = src.get("usd_dimensions") or {}
        vlm = {v["name"]: v.get("value_text") for v in src.get("plugin_thumbnail_to_vision_metadata_metadata_vlm_generated", [])}
        out.append({"package": m.group(3), "repo": m.group(1), "revision": m.group(2), "root_usd": m.group(4),
                    "size_m": [dims.get(f"bbox_dimension_{a}") for a in "xyz"], "mass_kg": props.get("simready_metadata_mass"),
                    "license": props.get("simready_metadata_asset_license"), "category": props.get("simready_metadata_category"),
                    "caption": (vlm.get("caption") or [None])[0], "score": hit.get("score")})
        if len(out) >= limit:
            break
    return out
