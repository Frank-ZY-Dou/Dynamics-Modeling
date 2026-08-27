"""Upright-on-plane tabletop repair with in-plane translation + yaw.

The upright tabletop variant: every object is
kept standing on a common support plane, roll/pitch tilt is driven to
zero along the scale path by a smooth homotopy, and each scale step
solves a QP over in-plane translation and yaw. Snapshots at 30/60/90%
inflation and the final state are written to a JSON file for rendering.

Usage:
  python run_upright.py --output out_seed42.json --N 12 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp
import trimesh


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s4r"))

from mesh_collision import MeshObject, evaluate_mesh_object_scene  # noqa: E402

DEFAULT_ASSET_DIR = HERE.parent / "data" / "kubric_pool"


def scan_eligible_assets(asset_dir: Path) -> list:
    """Collision meshes from the bundled pool, one per object directory."""
    out = sorted(Path(asset_dir).glob("*/collision_geometry.obj"))
    if not out:
        out = sorted(Path(asset_dir).glob("*.glb")) + sorted(Path(asset_dir).glob("*.obj"))
    return out


UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class AssetInstance:
    name: str
    verts: np.ndarray
    faces: np.ndarray
    bbox_center: np.ndarray
    normalize_factor: float


def rotz(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def roty(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotx(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def compose_rotation(yaw: float, roll: float, pitch: float, tilt_factor: float) -> np.ndarray:
    return rotz(yaw) @ roty(tilt_factor * pitch) @ rotx(tilt_factor * roll)


def tilt_schedule(scale: float) -> float:
    x = min(max((scale - 0.01) / 0.99, 0.0), 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    return 1.0 - smooth


def load_asset(path: Path, target_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mesh = trimesh.load(str(path), process=True, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    max_extent = float((bbox_max - bbox_min).max())
    nf = target_size / max(max_extent, 1e-12)
    return verts - bbox_center, faces, bbox_center, nf


def support_z(inst: AssetInstance, R: np.ndarray, scale: float) -> float:
    v = (R @ (scale * inst.normalize_factor * inst.verts).T).T
    return -float(v[:, 2].min())


def world_vertices(
    inst: AssetInstance,
    xy: np.ndarray,
    yaw: float,
    roll: float,
    pitch: float,
    scale: float,
    tilt_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = compose_rotation(yaw, roll, pitch, tilt_factor)
    center = np.array([xy[0], xy[1], support_z(inst, R, scale)], dtype=np.float64)
    verts = (R @ (scale * inst.normalize_factor * inst.verts).T).T + center
    return verts, center, R


def make_objects(
    instances: list[AssetInstance],
    xy: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    scale: float,
    tilt_factor: float,
) -> list[MeshObject]:
    objects = []
    for i, inst in enumerate(instances):
        _, center, R = world_vertices(inst, xy[i], float(yaw[i]), float(roll[i]), float(pitch[i]), scale, tilt_factor)
        objects.append(
            MeshObject(
                name=inst.name,
                center=center,
                rotation=R,
                collision_verts_model=inst.verts,
                collision_faces=inst.faces,
                visual_verts_model=inst.verts.copy(),
                visual_faces=inst.faces.copy(),
                normalize_factor=scale * inst.normalize_factor,
                inv_mass=1.0,
            )
        )
    return objects


def fcl_contacts(
    instances: list[AssetInstance],
    xy: np.ndarray,
    yaw: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    scale: float,
    tilt_factor: float,
):
    import fcl

    verts_world: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    aabb_min: list[np.ndarray] = []
    aabb_max: list[np.ndarray] = []
    collision_objects = []

    for i, inst in enumerate(instances):
        verts, center, R = world_vertices(
            inst, xy[i], float(yaw[i]), float(roll[i]), float(pitch[i]), scale, tilt_factor
        )
        verts_world.append(verts)
        centers.append(center)
        rotations.append(R)
        aabb_min.append(verts.min(axis=0))
        aabb_max.append(verts.max(axis=0))

        model = fcl.BVHModel()
        model.beginModel(len(verts), len(inst.faces))
        model.addSubModel(verts.astype(np.float64), inst.faces.astype(np.int32))
        model.endModel()
        collision_objects.append(fcl.CollisionObject(model, fcl.Transform()))

    contacts = []
    n = len(instances)
    for i in range(n):
        for j in range(i + 1, n):
            gap = np.maximum(aabb_min[i] - aabb_max[j], aabb_min[j] - aabb_max[i])
            if float(gap.max()) > 0.06:
                continue

            req = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
            res = fcl.DistanceResult()
            dist = float(fcl.distance(collision_objects[i], collision_objects[j], req, res))
            if dist > 0.0:
                cp_i = np.asarray(res.nearest_points[0], dtype=np.float64)
                cp_j = np.asarray(res.nearest_points[1], dtype=np.float64)
                raw = cp_j - cp_i
                normal = raw / max(np.linalg.norm(raw), 1e-12)
                signed = dist
            else:
                creq = fcl.CollisionRequest(num_max_contacts=32, enable_contact=True)
                cres = fcl.CollisionResult()
                fcl.collide(collision_objects[i], collision_objects[j], creq, cres)
                if not cres.is_collision or not cres.contacts:
                    cp_i = 0.5 * (centers[i] + centers[j])
                    cp_j = cp_i.copy()
                    normal = centers[j] - centers[i]
                    normal = normal / max(np.linalg.norm(normal), 1e-12)
                    signed = 0.0
                else:
                    c_best = max(cres.contacts, key=lambda c: c.penetration_depth)
                    signed = -float(c_best.penetration_depth)
                    normal = np.asarray(c_best.normal, dtype=np.float64)
                    if np.dot(normal, centers[j] - centers[i]) < 0.0:
                        normal = -normal
                    normal = normal / max(np.linalg.norm(normal), 1e-12)
                    pos = np.asarray(c_best.pos, dtype=np.float64)
                    cp_i = pos
                    cp_j = pos

            if np.dot(normal, centers[j] - centers[i]) < 0.0:
                normal = -normal

            # Full-scale extents along the current contact normal.  The ds term
            # multiplies these, so scale is intentionally omitted here.
            ext = []
            for idx in (i, j):
                projs = (rotations[idx] @ (instances[idx].normalize_factor * instances[idx].verts).T).T @ normal
                ext.append(float(projs.max() - projs.min()))

            contacts.append((i, j, signed, normal, ext[0], ext[1], cp_i, cp_j, centers[i], centers[j]))

    return contacts


def solve_step(instances, xy, yaw, roll, pitch, scale, tilt_factor, ds, d_hat, yaw_weight, max_yaw_step, tail=False):
    contacts = fcl_contacts(instances, xy, yaw, roll, pitch, scale, tilt_factor)
    active = []
    for c in contacts:
        i, j, signed, normal, ext_i, ext_j, cp_i, cp_j, ci, cj = c
        rhs = d_hat - signed + ds * (ext_i + ext_j)
        if rhs > 0.0:
            active.append(c + (rhs,))

    if not active:
        return xy, yaw, 0, sum(1 for c in contacts if c[2] < 0.0)

    active_bodies = sorted({idx for c in active for idx in (c[0], c[1])})
    body_map = {b: k for k, b in enumerate(active_bodies)}
    n_vars = 3 * len(active_bodies)
    n_rows = len(active)

    p_diag = []
    for _ in active_bodies:
        p_diag.extend([1.0, 1.0, yaw_weight])
    P = sp.diags(p_diag, format="csc")
    q = np.zeros(n_vars, dtype=np.float64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []

    for row, c in enumerate(active):
        i, j, signed, normal, _ext_i, _ext_j, cp_i, cp_j, ci, cj, rhs = c
        ii = body_map[i]
        jj = body_map[j]
        nxy = normal[:2]
        yi = cp_i - ci
        yj = cp_j - cj
        yaw_i = float(np.dot(np.cross(yi, normal), UP))
        yaw_j = float(np.dot(np.cross(yj, normal), UP))

        entries = [
            (3 * ii + 0, -nxy[0]),
            (3 * ii + 1, -nxy[1]),
            (3 * ii + 2, -yaw_i),
            (3 * jj + 0, nxy[0]),
            (3 * jj + 1, nxy[1]),
            (3 * jj + 2, yaw_j),
        ]
        for col, val in entries:
            if abs(val) > 1e-12:
                rows.append(row)
                cols.append(col)
                data.append(val)
        lower.append(float(rhs))

    # Small-angle yaw trust region.
    box_start = n_rows
    for k in range(len(active_bodies)):
        row = box_start + k
        rows.append(row)
        cols.append(3 * k + 2)
        data.append(1.0)
        lower.append(-max_yaw_step)

    A = sp.csc_matrix((data, (rows, cols)), shape=(n_rows + len(active_bodies), n_vars))
    l = np.asarray(lower, dtype=np.float64)
    u = np.concatenate([np.full(n_rows, np.inf), np.full(len(active_bodies), max_yaw_step)])

    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=8000, polish=True)
    result = solver.solve()
    if result.info.status not in ("solved", "solved_inaccurate"):
        return xy, yaw, len(active), sum(1 for c in contacts if c[2] < 0.0)

    step = result.x.reshape(len(active_bodies), 3)
    xy_new = xy.copy()
    yaw_new = yaw.copy()
    max_xy = 0.018 if tail else 0.03
    for local_idx, body in enumerate(active_bodies):
        dxy = step[local_idx, :2]
        norm = float(np.linalg.norm(dxy))
        if norm > max_xy:
            dxy *= max_xy / norm
        xy_new[body] += dxy
        yaw_new[body] += float(step[local_idx, 2])
    return xy_new, yaw_new, len(active), sum(1 for c in contacts if c[2] < 0.0)


def snapshot(instances, xy, yaw, roll, pitch, scale, tilt_factor):
    centers = []
    rotations = []
    for i, inst in enumerate(instances):
        _, center, R = world_vertices(
            inst, xy[i], float(yaw[i]), float(roll[i]), float(pitch[i]), scale, tilt_factor
        )
        centers.append(center.tolist())
        rotations.append(R.tolist())
    return {"centers": centers, "rotations": rotations, "scale": float(scale), "tilt_factor": float(tilt_factor)}


def stats(instances, xy, yaw, roll, pitch, scale, tilt_factor):
    ev = evaluate_mesh_object_scene(make_objects(instances, xy, yaw, roll, pitch, scale, tilt_factor))
    return {"pen": int(ev.pen_pairs), "max_pen": float(ev.max_penetration)}


def signed_tilt(rng: np.random.RandomState, n: int, min_deg: float, max_deg: float) -> np.ndarray:
    mag = rng.uniform(math.radians(min_deg), math.radians(max_deg), size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    return sign * mag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=float, default=0.10)
    parser.add_argument("--spawn-xy", type=float, default=0.16)
    parser.add_argument("--min-init-pen", type=int, default=42)
    parser.add_argument("--tilt-min-deg", type=float, default=28.0)
    parser.add_argument("--tilt-max-deg", type=float, default=42.0)
    parser.add_argument("--d-hat", type=float, default=0.004)
    parser.add_argument("--ds-max", type=float, default=0.05)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    eligible = scan_eligible_assets(args.asset_dir)
    if not eligible:
        raise RuntimeError(f"No eligible GLB assets in {args.asset_dir}")

    chosen = rng.choice(len(eligible), args.N, replace=args.N > len(eligible))
    instances: list[AssetInstance] = []
    for idx in chosen:
        path = eligible[int(idx)]
        verts, faces, bbox_center, nf = load_asset(path, args.target_size)
        instances.append(AssetInstance(path.stem, verts, faces, bbox_center, nf))

    best = None
    for attempt in range(80):
        xy_try = rng.normal(0.0, args.spawn_xy * 0.45, size=(args.N, 2))
        xy_try = np.clip(xy_try, -args.spawn_xy, args.spawn_xy)
        yaw_try = rng.uniform(-math.pi, math.pi, size=args.N)
        roll_try = signed_tilt(rng, args.N, args.tilt_min_deg, args.tilt_max_deg)
        pitch_try = signed_tilt(rng, args.N, args.tilt_min_deg, args.tilt_max_deg)
        cur_stats = stats(instances, xy_try, yaw_try, roll_try, pitch_try, 1.0, 1.0)
        if best is None or cur_stats["pen"] > best[0]["pen"]:
            best = (cur_stats, xy_try, yaw_try, roll_try, pitch_try)
        if cur_stats["pen"] >= args.min_init_pen:
            break

    init_stats, xy0, yaw0, roll0, pitch0 = best
    xy = xy0.copy()
    yaw = yaw0.copy()

    init = snapshot(instances, xy0, yaw0, roll0, pitch0, 1.0, 1.0)
    print(f"init pen={init_stats['pen']} max={init_stats['max_pen']:.5f}")

    scale = 0.01
    snapshots: dict[str, dict] = {}
    snapshot_stats: dict[str, dict] = {}
    render_targets = {"s30": 0.30, "s60": 0.60, "s90": 0.90}
    yaw_weight = args.target_size ** 2
    t0 = time.time()

    while scale < 1.0 - 1e-9:
        next_targets = [target - scale for target in render_targets.values() if target > scale + 1e-9]
        ds = min([args.ds_max, 1.0 - scale] + next_targets)
        tilt_factor = tilt_schedule(scale)
        xy, yaw, n_active, n_pen = solve_step(
            instances, xy, yaw, roll0, pitch0, scale, tilt_factor, ds, args.d_hat, yaw_weight, max_yaw_step=0.08, tail=False
        )
        scale += ds
        for tag, target in render_targets.items():
            if tag not in snapshots and abs(scale - target) < 1e-8:
                cur_tilt = tilt_schedule(scale)
                snapshots[tag] = snapshot(instances, xy, yaw, roll0, pitch0, scale, cur_tilt)
                snapshot_stats[tag] = stats(instances, xy, yaw, roll0, pitch0, scale, cur_tilt)
        print(f"scale={scale:.2f} active={n_active} pen_at_scale={n_pen}")

    for it in range(60):
        cur = stats(instances, xy, yaw, roll0, pitch0, 1.0, 0.0)
        print(f"tail={it:02d} pen={cur['pen']} max={cur['max_pen']:.6f}")
        if cur["pen"] == 0:
            break
        xy, yaw, _n_active, _n_pen = solve_step(
            instances, xy, yaw, roll0, pitch0, 1.0, 0.0, 0.0, args.d_hat, yaw_weight, max_yaw_step=0.04, tail=True
        )

    final = snapshot(instances, xy, yaw, roll0, pitch0, 1.0, 0.0)
    final_stats = stats(instances, xy, yaw, roll0, pitch0, 1.0, 0.0)

    out = {
        "N": args.N,
        "seed": args.seed,
        "target_size": args.target_size,
        "asset_dir": str(args.asset_dir),
        "asset_names": [inst.name for inst in instances],
        "normalize_factors": [float(inst.normalize_factor) for inst in instances],
        "asset_bbox_centers": [inst.bbox_center.tolist() for inst in instances],
        "upright_on_plane": True,
        "support_plane": {"up": [0, 0, 1], "height": 0.0},
        "solver": "progressive_inflation_in_plane_translation_plus_yaw_qp",
        "render_tags": ["init", "s30", "s60", "s90", "final"],
        "render_tag_labels": {
            "init": "Initial",
            "s30": "30%",
            "s60": "60%",
            "s90": "90%",
            "final": "Final",
        },
        "initial_tilt_degrees": {
            "roll": [float(math.degrees(v)) for v in roll0],
            "pitch": [float(math.degrees(v)) for v in pitch0],
        },
        "note": "Initial states use tilted roll/pitch, driven to zero along the scale path while in-plane translation and yaw are optimized in every QP step.",
        "runtime_sec": time.time() - t0,
        "init": init,
        "s30": snapshots["s30"],
        "s60": snapshots["s60"],
        "s90": snapshots["s90"],
        "mid": snapshots["s60"],
        "final": final,
        "stats": {
            "init": init_stats,
            "s30": snapshot_stats["s30"],
            "s60": snapshot_stats["s60"],
            "s90": snapshot_stats["s90"],
            "mid": snapshot_stats["s60"],
            "final": final_stats,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"final pen={final_stats['pen']} max={final_stats['max_pen']:.6f}")
    print(args.output)


if __name__ == "__main__":
    main()
