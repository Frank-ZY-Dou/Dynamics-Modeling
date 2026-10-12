"""Replay HopNet MOVi scenes' Kubric simulations from their recorded flags; recover every simulated object.

HopNet's generator (data/movi_worker.py of https://github.com/AmauryWEI/hopnet, run with its Kubric fork
https://github.com/AmauryWEI/kubric @ 511ef27) simulates every sampled object but writes only the ones visible in at
least one rendered frame (`instances = visible_foreground_assets`). This script rebuilds a scene from the flags stored
in its metadata.json -- same seed, same random-number sequence, same PyBullet set-up -- runs the same simulation
without rendering (rendering draws no random numbers and does not touch the simulation), and returns every foreground
object in creation order.

A replay is accepted only if it reproduces every exported object exactly: the same shape, size, material, colour and
physics, and positions, quaternions, velocities and angular velocities equal to the stored values in every frame.
Only then are the objects the export left out added to the scene's metadata.

Environment (see data_generation/README.md): Python 3.9, the Kubric fork installed with `pip install --no-deps -e`,
pybullet, and Kubric's import-time dependencies; the KuBasic assets with a local manifest.

    python data_generation/replay_movi_scene.py --raw_root raw/MOVi-B --scenes 1-1200 \
        --manifest KuBasic/KuBasic_local.json --report replay_B.json --write_root replay/MOVi-B --workers 32
"""
import argparse
import copy
import json
import multiprocessing
import pathlib
import shutil
import sys
import tempfile

import numpy as np

# HopNet data/movi_worker.py
SPAWN_REGION = [(-5.0, -5.0, 1.0), (5.0, 5.0, 5.0)]
VELOCITY_RANGE = [(-4.0, -4.0, 0.0), (4.0, 4.0, 0.0)]
CLEVR_OBJECTS = ("cube", "cylinder", "sphere")
KUBASIC_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus", "gear", "torus_knot", "sponge", "spot", "teapot",
                   "suzanne")

TRAJ_KEYS = ("positions", "quaternions", "velocities", "angular_velocities")
SCALAR_KEYS = ("asset_id", "shape", "size", "size_label", "material", "color", "color_label", "mass", "friction",
               "restitution")


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def replay(metadata, manifest, work_dir):
    """Rebuild and simulate the scene; return every foreground object's instance info, in creation order."""
    import kubric as kb
    from kubric.simulator import PyBullet

    flags = argparse.Namespace(**metadata["flags"])
    flags.job_dir = str(work_dir / "job")
    flags.scratch_dir = str(work_dir / "scratch")
    flags.logging_level = "WARNING"
    scene, rng, _, scratch_dir = kb.setup(flags)
    simulator = PyBullet(scene, scratch_dir)
    kubasic = kb.AssetSource.from_manifest(manifest, scratch_dir=scratch_dir)

    floor_material = kb.PrincipledBSDFMaterial(roughness=1.0, specular=0.0)
    scene += kubasic.create("dome", name="floor", material=floor_material, scale=2.0,
                            friction=flags.floor_friction, restitution=flags.floor_restitution,
                            static=True, background=True)
    if flags.background == "clevr":
        floor_material.color = kb.Color.from_name("gray")
    elif flags.background == "colored":
        floor_material.color = kb.random_hue_color()  # Kubric's own generator, not the seeded one
    else:
        raise ValueError(f"unknown background {flags.background!r}")

    scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

    scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
    if flags.camera == "clevr":
        scene.camera.position = [7.48113, -6.50764, 5.34367] + rng.rand(3)
    elif flags.camera == "random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7.0, outer_radius=9.0, offset=0.1)
    else:
        raise ValueError(f"unknown camera {flags.camera!r}")
    scene.camera.look_at((0, 0, 0))

    num_objects = rng.randint(flags.min_num_objects, flags.max_num_objects + 1)
    for _ in range(num_objects):
        if flags.objects_set == "spheres":
            shape_name = "sphere"
            size_label, size = kb.randomness.sample_sizes("clevr", rng)
            color_label, random_color = kb.randomness.sample_color("clevr", rng)
        elif flags.objects_set == "clevr":
            shape_name = rng.choice(CLEVR_OBJECTS)
            size_label, size = kb.randomness.sample_sizes("clevr", rng)
            color_label, random_color = kb.randomness.sample_color("clevr", rng)
        elif flags.objects_set == "kubasic":
            shape_name = rng.choice(KUBASIC_OBJECTS)
            size_label, size = kb.randomness.sample_sizes("uniform", rng)
            color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)
        else:
            raise ValueError(f"unknown objects_set {flags.objects_set!r}")

        material_name = rng.choice(["metal", "rubber"])
        obj = kubasic.create(asset_id=shape_name, scale=size,
                             name=f"{size_label} {color_label} {material_name} {shape_name}")
        assert isinstance(obj, kb.FileBasedObject)
        if material_name == "metal":
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0, roughness=0.2, ior=2.5)
            obj.friction = 0.4
            obj.restitution = 0.3
            obj.mass *= 2.7 * size**3
        else:
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.0, ior=1.25, roughness=0.7,
                                                     specular=0.33)
            obj.friction = 0.8
            obj.restitution = 0.7
            obj.mass *= 1.1 * size**3
        obj.metadata = {"shape": shape_name.lower(), "size": size, "size_label": size_label,
                        "material": material_name.lower(), "color": random_color.rgb, "color_label": color_label}
        scene.add(obj)
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
        obj.velocity = rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0]

    simulator.run(frame_start=0, frame_end=scene.frame_end + 1)

    # image positions are projected with the recorded camera (the random camera draws from Kubric's own generator)
    cam = metadata["camera"]
    scene.camera.position = cam["positions"][0]
    scene.camera.quaternion = cam["quaternions"][0]
    objects = [a for a in scene.foreground_assets if isinstance(a, kb.FileBasedObject)]  # not lights or camera
    assert len(objects) == num_objects, (len(objects), num_objects)
    instances = _jsonable(kb.get_instance_info(scene, objects))
    kb.assets.ClosableResource.close_all()
    return instances


def _same_object(stored, replayed):
    return all(stored[k] == replayed[k] for k in SCALAR_KEYS)


def _start_gap(stored, replayed):
    return float(np.max(np.abs(np.asarray(stored["positions"][0]) - np.asarray(replayed["positions"][0]))))


def compare(stored_instances, replayed):
    """Match every stored instance to a replayed object; report whether each is reproduced exactly."""
    matched, report = set(), []
    for i, s in enumerate(stored_instances):
        cand = [j for j, r in enumerate(replayed) if j not in matched and _same_object(s, r)]
        if not cand:
            report.append({"instance": i, "asset_id": s["asset_id"], "match": 0})
            continue
        j = min(cand, key=lambda c: _start_gap(s, replayed[c]))
        matched.add(j)
        r = replayed[j]
        diffs = {k: float(np.max(np.abs(np.asarray(s[k], np.float64) - np.asarray(r[k], np.float64))))
                 for k in TRAJ_KEYS + ("bboxes_3d",)}
        exact = all(s[k] == r[k] for k in TRAJ_KEYS)
        report.append({"instance": i, "asset_id": s["asset_id"], "match": len(cand), "replayed_index": j,
                       "exact": exact, "max_abs_diff": diffs})
    return matched, report


def process_scene(args):
    scene_dir, manifest, write_root, tmp_root = args
    scene_dir = pathlib.Path(scene_dir)
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"replay_{scene_dir.name}_", dir=tmp_root))
    try:
        replayed = replay(metadata, manifest, work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    stored = metadata["instances"]
    matched, report = compare(stored, replayed)
    ok = (len(matched) == len(stored) and all(r.get("exact") for r in report))
    missing = [j for j in range(len(replayed)) if j not in matched]
    result = {"scene": scene_dir.name, "exported": len(stored), "simulated": len(replayed), "reproduced": ok,
              "omitted": [replayed[j]["asset_id"] for j in missing], "objects": report}
    if write_root is not None and ok and missing:
        out = copy.deepcopy(metadata)
        n_frames = len(stored[0]["positions"])
        for j in missing:
            inst = dict(replayed[j])
            inst["visibility"] = [0] * n_frames  # never visible: no 2D boxes
            inst["bboxes"] = []
            inst["bbox_frames"] = []
            out["instances"].append(inst)
        out["metadata"]["num_instances"] = len(out["instances"])
        import pybullet
        out["replay"] = {
            "tool": "data_generation/replay_movi_scene.py",
            "generator": "HopNet data/movi_worker.py with Kubric AmauryWEI/kubric@511ef27, rendering skipped",
            "pybullet": pybullet.__version__ if hasattr(pybullet, "__version__") else None,
            "exported_instances_reproduced_exactly": len(stored),
            "added_instances": [len(stored) + k for k in range(len(missing))],
        }
        dst = pathlib.Path(write_root) / scene_dir.name
        dst.mkdir(parents=True, exist_ok=False)
        (dst / "metadata.json").write_text(json.dumps(out))
        result["written"] = str(dst / "metadata.json")
    return result


def parse_scenes(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="raw MOVi dataset root (<root>/<seed>/metadata.json)")
    ap.add_argument("--scenes", required=True, help="e.g. 1-1200 or 522,203")
    ap.add_argument("--manifest", required=True, help="local KuBasic manifest (data_dir pointing at the assets)")
    ap.add_argument("--report", required=True, help="JSON report, one record per scene")
    ap.add_argument("--write_root", default=None,
                    help="write the completed metadata.json of every exactly reproduced incomplete scene here")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp_root", default=None, help="scratch for the per-scene asset copies (default: next to --report)")
    args = ap.parse_args()

    tmp_root = pathlib.Path(args.tmp_root or pathlib.Path(args.report).resolve().parent / "replay_tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    if args.write_root:
        pathlib.Path(args.write_root).mkdir(parents=True, exist_ok=True)     # also when no scene needs completing
    jobs = [(str(pathlib.Path(args.raw_root) / str(s)), args.manifest, args.write_root, str(tmp_root))
            for s in parse_scenes(args.scenes)]
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        results = pool.map(process_scene, jobs, chunksize=1)
    pathlib.Path(args.report).write_text(json.dumps(results, indent=1))
    n_ok = sum(r["reproduced"] for r in results)
    n_incomplete = sum(r["simulated"] > r["exported"] for r in results)
    print(f"{len(results)} scenes: {n_ok} reproduced exactly, {len(results) - n_ok} not; "
          f"{n_incomplete} with omitted objects; written {sum('written' in r for r in results)}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
