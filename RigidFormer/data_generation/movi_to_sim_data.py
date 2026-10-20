"""
Convert raw MOVi scenes (MOVi-A / MOVi-B / MOVi-Sphere) into the RigidFormer
training format: per-frame vertex positions, mesh faces, physics parameters,
and deterministic farthest-point-sampling anchor caches.

Input layout: one directory per scene (1, 2, ...), each containing the
Kubric-style metadata.json (object poses, physics) and rgba.mp4.

The stored vertices are R(q) v_asset + p in asset units: the metadata `size` is not applied here; training and
evaluation apply it with data.object_scale_file (build_object_scales.py). A scene whose metadata lacks simulated (never
visible) objects is converted but marked incomplete in its conversion.json; replay_movi_scene.py recovers such objects.

Usage:
  python data_generation/movi_to_sim_data.py \
    --input ./raw/MOVi-B \
    --output ./datasets/MOVi-B \
    --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_scenes import replay_num_objects  # noqa: E402

# Recorded in every conversion.json; an existing output is reused only under the same version
CONVERTER_VERSION = 3   # 3: seed replay from metadata.seed; reuse verifies every stored file


def _compute_fps_anchors(vertices: np.ndarray, K: int) -> np.ndarray:
    """
    Compute K anchor points using Farthest Point Sampling (FPS).

    This algorithm is deterministic: the same input vertices always produce
    the same anchor indices. No randomness is involved.

    Algorithm:
    1. Always starts from vertex 0 (deterministic starting point)
    2. Iteratively selects the vertex farthest from all previously selected anchors
    3. np.argmax returns the first occurrence on ties (deterministic)

    Args:
        vertices: (Nv, 3) array of vertex positions
        K: Number of anchor points to sample

    Returns:
        anchors: (K,) array of anchor indices (np.int32)
    """
    Nv = vertices.shape[0]
    if K >= Nv:
        # If K >= number of vertices, just return all indices
        return np.arange(Nv, dtype=np.int32)

    # FPS algorithm (deterministic, no random numbers)
    anchors = np.zeros(K, dtype=np.int32)
    distances = np.full(Nv, np.inf, dtype=np.float32)

    # Always start from first vertex (index 0) - deterministic
    current = 0
    for k in range(K):
        anchors[k] = current

        # Update distances: min distance to any selected anchor
        dists_to_current = np.linalg.norm(vertices - vertices[current], axis=1)
        distances = np.minimum(distances, dists_to_current)

        # Select farthest point (argmax returns first occurrence if ties)
        current = np.argmax(distances)

    return anchors


# MOVI_SAMPLE_REQUIRED = ("collisions.json", "metadata.json", "rgba.mp4")
MOVI_SAMPLE_REQUIRED = ("metadata.json", "rgba.mp4")


@dataclass
class ObjectSeries:
    name: str
    shape: str
    positions: np.ndarray  # (T,3)
    quaternions_wxyz: np.ndarray  # (T,4)
    mesh: trimesh.Trimesh


def _as_np(a: List[List[float]]) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _pose_arrays(inst: Dict, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """The Kubric pose arrays of an instance: positions (T, 3) and quaternions (T, 4, WXYZ), T = num_frames.

    Only the Kubric keys are accepted (a heuristic search over (T, 3) / (T, 4) arrays accepted angular velocities as
    positions and image boxes as quaternions); lengths must equal the scene's num_frames and quaternions must be unit.
    """
    for key, width in (("positions", 3), ("quaternions", 4)):
        if key not in inst:
            raise KeyError(f"instance {inst.get('asset_id')!r} has no '{key}' array (keys: {sorted(inst)})")
        a = np.asarray(inst[key], dtype=np.float64)
        if a.shape != (num_frames, width) or not np.isfinite(a).all():
            raise ValueError(f"instance {inst.get('asset_id')!r}: '{key}' has shape {a.shape} (expected ({num_frames}, "
                             f"{width}), finite)")
    norm_err = float(np.abs(np.linalg.norm(np.asarray(inst["quaternions"], dtype=np.float64), axis=1) - 1.0).max())
    if not norm_err <= 1e-4:
        raise ValueError(f"instance {inst.get('asset_id')!r}: quaternions are not unit (max |norm - 1| = {norm_err:.2e})")
    pos, quat = _as_np(inst["positions"]), _as_np(inst["quaternions"])
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):          # (float32 overflow)
        raise ValueError(f"instance {inst.get('asset_id')!r}: poses are not finite in float32")
    return pos, quat


def _load_mesh_for_instance(obj: Dict, base: Path) -> Tuple[str, trimesh.Trimesh, str]:
    """(shape, collision mesh in asset units, sha256 of the mesh file)."""
    shape = obj.get("shape") or obj.get("asset_id")
    mesh_path = base / str(shape) / "collision_geometry.obj"

    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Missing mesh for shape='{shape}': {mesh_path}\n"
            f"Please check:\n"
            f"  1. Object exists in {base}\n"
            f"  2. File collision_geometry.obj is present"
        )

    mesh = trimesh.load(mesh_path.as_posix(), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise NotImplementedError(f"{mesh_path} loads as a {type(mesh).__name__}, not a single mesh: merging its parts "
                                  f"would not be the simulated collision geometry")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError(f"{mesh_path}: non-finite vertex coordinates")
    return str(shape), mesh, hashlib.sha256(mesh_path.read_bytes()).hexdigest()


def _apply_pose_vertices(mesh: trimesh.Trimesh, p: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    # trimesh uses [w, x, y, z] in quaternion_matrix
    T = trimesh.transformations.quaternion_matrix(q_wxyz)
    T[:3, 3] = p
    v = trimesh.transform_points(mesh.vertices, T)
    return v.astype(np.float32, copy=False)


ANCHOR_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16)


@dataclass
class Conversion:
    """Everything one scene's conversion writes (per-frame vertices are produced on demand by frame_vertices)."""
    ds_name: str
    T: int
    series: List[ObjectSeries]
    physics: Dict[str, Dict]
    faces: List[np.ndarray]
    anchors: Dict[Tuple[int, int], np.ndarray]
    traj: np.ndarray
    manifest: Dict

    def frame_vertices(self, t: int) -> List[np.ndarray]:
        out = [_apply_pose_vertices(s.mesh, s.positions[t], s.quaternions_wxyz[t]).astype(np.float32) for s in self.series]
        if not all(np.isfinite(V).all() for V in out):                    # (float32 overflow of transformed vertices)
            raise ValueError(f"{self.ds_name}: non-finite vertices at frame {t}")
        return out


def _prepare(sample_dir: Path, assets_dir: Path) -> Conversion:
    for fname in MOVI_SAMPLE_REQUIRED:
        if not (sample_dir / fname).exists():
            raise FileNotFoundError(f"{sample_dir/fname} not found")
    meta_bytes = (sample_dir / "metadata.json").read_bytes()
    return prepare_from_metadata(json.loads(meta_bytes), f"movi_{sample_dir.name}", assets_dir,
                                 source=str(sample_dir.resolve()), metadata_sha256=hashlib.sha256(meta_bytes).hexdigest())


def prepare_from_metadata(meta: Dict, ds_name: str, assets_dir: Path, source: str,
                          metadata_sha256: Optional[str]) -> Conversion:
    """One scene's conversion from its Kubric metadata (parsed); `source` and `metadata_sha256` go into the manifest
    (materialize_hf_dataset.py converts the Hugging Face dataset's rows with it)."""
    sample_dir = source
    instances: List[Dict] = meta.get("instances")
    if not instances:
        raise RuntimeError(f"No instances found in the metadata of {sample_dir}")
    T = int(meta["metadata"]["num_frames"])

    # Build object series. Kubric exports the foreground assets (visible in at least one frame); the floor is a static
    # background asset and is not among them. A static obstacle could not be represented (every object moves).
    series: List[ObjectSeries] = []
    physics_params: Dict[str, Dict] = {}  # Physics parameters per object
    objects_info = []
    for inst in instances:
        shape = (inst.get("shape") or inst.get("asset_id") or "").lower()
        if shape in ("floor", "dome"):
            continue  # the ground plane z = 0 is implicit in the model
        if inst.get("static") or inst.get("is_static"):
            raise NotImplementedError(f"{sample_dir}: static instance {inst.get('asset_id')!r} cannot be represented")
        pos, quat = _pose_arrays(inst, T)
        shape_name, mesh, mesh_sha = _load_mesh_for_instance(inst, assets_dir)
        name = inst.get("name") or shape_name
        # Physics parameters from the instance (all required, finite; nothing is defaulted or invented)
        missing = [k for k in ("mass", "friction", "restitution", "size") if k not in inst]
        if missing:
            raise KeyError(f"{sample_dir}: instance {inst.get('asset_id')!r} lacks {missing}")
        if not all(np.isfinite(float(inst[k])) for k in ("mass", "friction", "restitution")) or not np.isfinite(
                np.asarray(inst["size"], dtype=np.float64)).all():
            raise ValueError(f"{sample_dir}: instance {inst.get('asset_id')!r} has non-finite physics parameters")
        physics_params[f"obj_{len(series)}"] = {
            "mass": float(inst["mass"]),
            "friction": float(inst["friction"]),
            "restitution": float(inst["restitution"]),
            "shape": shape_name,
            "material": inst.get("material"),
            "size": inst["size"],
        }
        objects_info.append({"asset_id": shape_name, "size": inst["size"], "mesh_sha256": mesh_sha})
        series.append(ObjectSeries(name=name, shape=shape_name, positions=pos, quaternions_wxyz=quat, mesh=mesh))
    if len(series) == 0:
        raise RuntimeError(f"No objects found in {sample_dir}")

    faces = [s.mesh.faces.astype(np.int32) for s in series]
    # deterministic FPS anchors (K=1,2,3,4,6,8,12,16) on the reference (asset-frame) vertices
    anchors = {(i, K): _compute_fps_anchors(s.mesh.vertices.astype(np.float32), K)
               for i, s in enumerate(series) for K in ANCHOR_COUNTS}
    traj = np.zeros((T, len(series), 7), dtype=np.float32)
    for i, s in enumerate(series):
        traj[:, i, :3] = s.positions[:T]          # object origins
        traj[:, i, 3:7] = s.quaternions_wxyz[:T]  # quaternions (w, x, y, z)
    simulated = replay_num_objects(meta)
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "source": source,
        "metadata_sha256": metadata_sha256,
        "num_frames": T,
        "frame_rate": meta["metadata"].get("frame_rate"),
        "vertices": "R(q) v_asset + p in asset units: the metadata size is NOT applied (data.object_scale_file, "
                    "data_generation/build_object_scales.py)",
        "objects": objects_info,
        "exported_objects": len(series),
        "simulated_objects": simulated,
        "complete": len(series) == simulated,
    }
    return Conversion(ds_name, T, series, physics_params, faces, anchors, np.expand_dims(traj, axis=0), manifest)


def _verify_existing(out_ds: Path, conv: Conversion) -> None:
    """An existing conversion is reused only if every file this conversion writes is present and equal to what it
    would write now (manifest -- converter version, metadata and asset hashes, completeness -- physics, faces, anchors,
    trajectory, and every frame's vertices, bit for bit). Files added later by the loader (other anchor caches) are
    not part of the conversion."""
    def _same(path, ref):
        if not path.exists():
            return False
        a = np.load(path)
        return a.dtype == ref.dtype and a.shape == ref.shape and np.array_equal(a, ref)
    man = json.loads((out_ds / "conversion.json").read_text())
    want = dict(conv.manifest)
    if {k: v for k, v in man.items() if k != "source"} != {k: v for k, v in want.items() if k != "source"}:
        raise RuntimeError(f"{out_ds}: the manifest differs from this conversion (converter version, metadata, asset "
                           f"meshes or completeness): convert into an empty output directory")
    if json.loads((out_ds / "physics_params.json").read_text()) != conv.physics:
        raise RuntimeError(f"{out_ds}: physics_params.json differs from the metadata")
    if not _same(out_ds / "vis_traj.npy", conv.traj):
        raise RuntimeError(f"{out_ds}: vis_traj.npy differs from the metadata poses")
    for i, F in enumerate(conv.faces):
        if not _same(out_ds / "meshes" / f"obj_{i}_faces.npy", F):
            raise RuntimeError(f"{out_ds}: faces of object {i} differ from the asset mesh")
    for (i, K), idx in conv.anchors.items():
        if not _same(out_ds / "anchors" / f"obj_{i}_anchors_fps{K}.npy", idx):
            raise RuntimeError(f"{out_ds}: anchor cache obj_{i}_anchors_fps{K}.npy differs")
    frames = sorted(p.name for p in out_ds.iterdir() if p.name.startswith("frame_"))
    if frames != [f"frame_{t:05d}" for t in range(conv.T)]:
        raise RuntimeError(f"{out_ds}: frame directories are not frame_00000 .. frame_{conv.T - 1:05d}")
    expected = sorted(f"obj_{i}_pos.npy" for i in range(len(conv.series)))
    for t in range(conv.T):
        if sorted(p.name for p in (out_ds / f"frame_{t:05d}").iterdir()) != expected:
            raise RuntimeError(f"{out_ds}: frame {t} does not hold exactly the files {expected}")
        for i, V in enumerate(conv.frame_vertices(t)):
            if not _same(out_ds / f"frame_{t:05d}" / f"obj_{i}_pos.npy", V):
                raise RuntimeError(f"{out_ds}: frame {t} object {i} is missing or differs from the posed asset mesh")


def process_sample(sample_dir: Path, dst_root: Path, assets_dir: Path) -> None:
    """
    Convert one raw scene directory (e.g. MOVi-B/1) into dst_root/movi_<n>/; assets_dir is the mesh library with
    <shape>/collision_geometry.obj per shape. An existing output directory is reused only when its manifest exists
    and every stored file equals what this conversion writes (_verify_existing); anything else is an error.
    """
    assert sample_dir.exists(), sample_dir
    conv = _prepare(sample_dir, assets_dir)
    out_ds = dst_root / conv.ds_name
    manifest_file = out_ds / "conversion.json"
    if out_ds.exists():
        if not manifest_file.exists():
            raise RuntimeError(f"{out_ds} exists without a conversion manifest (interrupted or stale): "
                               f"convert into an empty output directory")
        _verify_existing(out_ds, conv)
        return
    write_conversion(out_ds, conv)


def write_conversion(out_ds: Path, conv: Conversion, verbose: bool = True, packed: bool = False) -> None:
    """Write one scene's conversion into the new folder out_ds; the manifest is written last. verbose: frame progress
    bar and a note for an incomplete scene. packed: the per-frame vertices go into frames.npy (T, sum_i Nv_i, 3) with
    frames_counts.npy (N,) instead of one file per object and frame (src/data/scene_frames.py reads both)."""
    out_ds.mkdir(parents=True, exist_ok=False)

    # mesh faces, physics parameters, anchors, per-frame vertices, trajectory; the manifest last
    (out_ds / "meshes").mkdir()
    for i, F in enumerate(conv.faces):
        np.save(out_ds / "meshes" / f"obj_{i}_faces.npy", F)
    with open(out_ds / "physics_params.json", "w") as f:
        json.dump(conv.physics, f, indent=2)
    (out_ds / "anchors").mkdir()
    for (i, K), idx in conv.anchors.items():
        np.save(out_ds / "anchors" / f"obj_{i}_anchors_fps{K}.npy", idx)
    if packed:
        frames, counts = None, None
        for t in tqdm(range(conv.T), desc=f"[{conv.ds_name}] Frames", unit="frame", leave=False, disable=not verbose):
            Vs = conv.frame_vertices(t)
            if frames is None:
                counts = np.array([V.shape[0] for V in Vs], dtype=np.int64)
                frames = np.empty((conv.T, int(counts.sum()), 3), dtype=np.float32)
            frames[t] = np.concatenate(Vs, axis=0)
        np.save(out_ds / "frames_counts.npy", counts)
        with open(out_ds / "frames.npy.part", "wb") as fh:
            np.save(fh, frames)
        os.replace(out_ds / "frames.npy.part", out_ds / "frames.npy")
    for t in tqdm(range(0 if packed else conv.T), desc=f"[{conv.ds_name}] Frames", unit="frame", leave=False,
                  disable=not verbose):
        frame_dir = out_ds / f"frame_{t:05d}"
        frame_dir.mkdir()
        for i, V in enumerate(conv.frame_vertices(t)):
            np.save(frame_dir / f"obj_{i}_pos.npy", V)
    np.save(out_ds / "vis_traj.npy", conv.traj)
    tmp = out_ds / "conversion.json.part"
    tmp.write_text(json.dumps(conv.manifest, indent=1))
    os.replace(tmp, out_ds / "conversion.json")
    if verbose and not conv.manifest["complete"]:
        print(f"  [INCOMPLETE] {conv.ds_name}: {conv.manifest['simulated_objects']} simulated objects, "
              f"{conv.manifest['exported_objects']} exported (invisible bodies are missing from the metadata; "
              f"replay_movi_scene.py recovers them)")


def main():
    parser = argparse.ArgumentParser(description="Convert raw MOVi scenes to the RigidFormer training format")
    parser.add_argument("--input", type=str, required=True, help="Raw MOVi scenes directory (per-scene subfolders 1, 2, ...)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for datasets")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--assets", type=str, default=str(Path(__file__).resolve().parent / "objects"),
                        help="Mesh library directory (<shape>/collision_geometry.obj per shape)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    assets_dir = Path(args.assets)
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect sample directories (numeric folders)
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    if args.limit:
        sample_dirs = sample_dirs[:args.limit]

    if not sample_dirs:
        raise RuntimeError(f"No sample directories found in {input_dir}")

    print(f"[movi_to_sim_data] Found {len(sample_dirs)} samples")
    print(f"[movi_to_sim_data] Output: {output_dir}")
    print(f"[movi_to_sim_data] Workers: {args.workers}")
    print("")

    # Process samples (serial or parallel)
    if args.workers == 1:
        # Serial processing
        for sd in tqdm(sample_dirs, desc="[movi_to_sim_data] Processing", unit="sample"):
            try:
                process_sample(sd, output_dir, assets_dir)
                print(f"  [OK] {sd.name}")
            except Exception as e:
                print(f"  [FAIL] {sd.name}: {e}")
                raise  # No fallback: re-raise immediately
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, sd, output_dir, assets_dir): sd for sd in sample_dirs}
            with tqdm(total=len(sample_dirs), desc="[movi_to_sim_data] Processing", unit="sample") as pbar:
                for future in as_completed(futures):
                    sd = futures[future]
                    try:
                        future.result()
                        pbar.set_postfix_str(f"[OK] {sd.name}")
                        pbar.update(1)
                    except Exception as e:
                        pbar.set_postfix_str(f"[FAIL] {sd.name}: {e}")
                        pbar.update(1)
                        raise  # No fallback: re-raise immediately

    print("")
    print(f"[movi_to_sim_data] All samples processed successfully")
    print(f"[movi_to_sim_data] Generated files per dataset:")
    print(f"  - frame_XXXXX/obj_i_pos.npy      (vertex positions)")
    print(f"  - meshes/obj_i_faces.npy         (mesh connectivity)")
    print(f"  - physics_params.json            (physics parameters: mass, friction, restitution, etc.)")
    print(f"  - anchors/obj_i_anchors_fps{{K}}.npy (K=1,2,3,4,6,8,12,16 - deterministic FPS)")
    print(f"  - vis_traj.npy                    (trajectory data)")


if __name__ == "__main__":
    main()
