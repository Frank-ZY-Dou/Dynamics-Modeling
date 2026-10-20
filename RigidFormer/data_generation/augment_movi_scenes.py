"""New MOVi-B-style training scenes: MOVi-B's generator without rendering, with continuous materials and two more
kinds of motion. For training only.

The generator is HopNet's data/movi_worker.py (https://github.com/AmauryWEI/hopnet) on its Kubric fork
(https://github.com/AmauryWEI/kubric @ 511ef27) with PyBullet 3.2.6, set up as for MOVi-B: KuBasic shapes and sizes
(0.7-1.4), the floor (friction 0.3, restitution 0.5), gravity -10 m/s^2, 1200 simulation steps per second, poses at
240 Hz over 2 s (480 frames). Every object draws its own friction U(0.05, 1.0), restitution U(0, 0.9) and density
U(1.1, 2.7) (MOVi-B's rubber and metal densities as the bounds; mass = KuBasic asset mass x density x size^3 as in
MOVi-B) instead of one of MOVi-B's two materials (friction 0.8 / 0.4, restitution 0.7 / 0.3). Kinds of scene:
    movi   3-10 objects placed and thrown as in MOVi-B;
    toss   2-4 objects spawned 0.5-3 m above the floor, with MOVi-B's velocities and a spin of 0-10 rad/s about a
           random axis;
    slide  2-5 objects set down upright (random yaw) on the floor and pushed: 0.5-5 m/s horizontally in a random
           direction and 0-4 rad/s about the vertical.
The ranges are generic, not fitted to any evaluation data. Seeds start at 10,000,000, far from MOVi-B's, so no scene
repeats a MOVi-B scene.

Each scene goes to <out>/<scene>/metadata.json in the layout of the Kubric metadata the converter reads
(data_generation/movi_to_sim_data.py: instances with the positions, quaternions, velocities and angular velocities of
every frame and each object's shape, size, mass, friction and restitution). Environment as for replay_movi_scene.py
(data_generation/README.md).

    python data_generation/augment_movi_scenes.py --manifest KuBasic/KuBasic_local.json --out raw/MOVi-B-aug \\
        --kinds movi:1200,toss:600,slide:600 --workers 32
"""
import argparse
import json
import multiprocessing
import os
import pathlib
import shutil
import tempfile
import time

import numpy as np

SPAWN_REGION = [(-5.0, -5.0, 1.0), (5.0, 5.0, 5.0)]          # HopNet data/movi_worker.py
VELOCITY_RANGE = [(-4.0, -4.0, 0.0), (4.0, 4.0, 0.0)]
KUBASIC_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus", "gear", "torus_knot", "sponge", "spot", "teapot",
                   "suzanne")
FRAME_RATE, STEP_RATE, FRAME_END = 240, 1200, 479            # HopNet data/generate_dataset.py: 240 Hz, 5 x 240, 2 s
FLOOR_FRICTION, FLOOR_RESTITUTION = 0.3, 0.5
FRICTION, RESTITUTION, DENSITY = (0.05, 1.0), (0.0, 0.9), (1.1, 2.7)
TOSS_REGION = [(-5.0, -5.0, 0.5), (5.0, 5.0, 3.0)]
TOSS_SPIN = 10.0                                             # rad/s, upper bound
SLIDE_SPEED, SLIDE_SPIN = (0.5, 5.0), 4.0                    # m/s; rad/s about z, upper bound
KINDS = {"movi": (3, 10, 0), "toss": (2, 4, 1_000_000), "slide": (2, 5, 2_000_000)}   # min, max objects; seed offset
SEED_BASE = 10_000_000


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


def scene_seed(kind, index):
    return SEED_BASE + KINDS[kind][2] + index


def set_down(obj, simulator, rng, max_trials=100):
    """Place obj upright with a random yaw, its lowest point 5 mm above the floor, where it overlaps nothing."""
    client, idx = simulator._physics_client, obj.linked_objects[simulator]
    for _ in range(max_trials):
        x, y = rng.uniform(-5.0, 5.0, size=2)
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        obj.quaternion = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))      # (w, x, y, z)
        obj.position = (x, y, 10.0)
        lo, _ = client.getAABB(idx)
        obj.position = (x, y, 10.0 - lo[2] + 0.005)
        if not simulator.check_overlap(obj):
            return
    raise RuntimeError(f"no free place on the floor for {obj.asset_id} after {max_trials} trials")


def simulate(job):
    """Simulate one scene and write its metadata.json; returns (scene, objects, seconds)."""
    kind, index, manifest, out_root = job
    name = f"aug_{kind}_{index}"
    out = pathlib.Path(out_root) / name
    if (out / "metadata.json").exists():
        return name, -1, 0.0
    import kubric as kb
    from kubric.simulator import PyBullet
    t0 = time.time()
    seed = scene_seed(kind, index)
    scratch = tempfile.mkdtemp(prefix="kubric_aug_")
    try:
        flags = kb.ArgumentParser().parse_args([
            "--seed", str(seed), "--frame_rate", str(FRAME_RATE), "--step_rate", str(STEP_RATE), "--frame_start", "0",
            "--frame_end", str(FRAME_END), "--resolution", "64", "--scratch_dir", scratch, "--job_dir", scratch,
            "--logging_level", "WARNING"])
        scene, rng, _, scratch_dir = kb.setup(flags)
        simulator = PyBullet(scene, scratch_dir)
        kubasic = kb.AssetSource.from_manifest(manifest, scratch_dir=scratch_dir)
        scene += kubasic.create("dome", name="floor", scale=2.0, friction=FLOOR_FRICTION,
                                restitution=FLOOR_RESTITUTION, static=True, background=True)
        scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
        scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7.0, outer_radius=9.0, offset=0.1)
        scene.camera.look_at((0, 0, 0))
        lo, hi, _ = KINDS[kind]
        objects = []
        for i in range(rng.randint(lo, hi + 1)):
            shape = rng.choice(KUBASIC_OBJECTS)
            _, size = kb.randomness.sample_sizes("uniform", rng)
            friction, restitution, density = (float(rng.uniform(*r)) for r in (FRICTION, RESTITUTION, DENSITY))
            obj = kubasic.create(asset_id=shape, scale=size, name=f"{shape} {i}")
            obj.friction, obj.restitution = friction, restitution
            obj.mass *= density * size ** 3
            obj.metadata = {"shape": shape, "size": size, "material": "continuous", "density": density}
            scene.add(obj)
            if kind == "slide":
                set_down(obj, simulator, rng)
                d = rng.uniform(0.0, 2.0 * np.pi)
                v = rng.uniform(*SLIDE_SPEED)
                obj.velocity = (v * np.cos(d), v * np.sin(d), 0.0)
                obj.angular_velocity = (0.0, 0.0, rng.uniform(-SLIDE_SPIN, SLIDE_SPIN))
            else:
                kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION if kind == "movi" else TOSS_REGION,
                                         rng=rng)
                obj.velocity = rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0]
                if kind == "toss":
                    axis = rng.normal(size=3)
                    obj.angular_velocity = axis / np.linalg.norm(axis) * rng.uniform(0.0, TOSS_SPIN)
            objects.append(obj)
        simulator.run(frame_start=0, frame_end=scene.frame_end + 1)
        instances = _jsonable(kb.get_instance_info(scene, objects))
        for inst in instances:
            if len(inst["positions"]) != FRAME_END + 1:
                raise RuntimeError(f"{name}: {len(inst['positions'])} frames, {FRAME_END + 1} expected")
        meta = {"metadata": {"num_frames": FRAME_END + 1, "frame_rate": FRAME_RATE, "step_rate": STEP_RATE,
                             "seed": seed, "gravity": [float(g) for g in scene.gravity]},
                "flags": {"generator": "augment_movi_scenes", "kind": kind, "seed": seed},
                "instances": instances}
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "metadata.json.part"
        tmp.write_text(json.dumps(_jsonable(meta)))
        os.replace(tmp, out / "metadata.json")
        return name, len(objects), time.time() - t0
    finally:
        kb.assets.ClosableResource.close_all()
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="KuBasic manifest with a local data_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="movi:1200,toss:600,slide:600", help="kind:count,... (scenes 0..count-1)")
    ap.add_argument("--first", type=int, default=0, help="first scene index of every kind")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = []
    for spec in args.kinds.split(","):
        kind, count = spec.split(":")
        if kind not in KINDS:
            raise SystemExit(f"unknown kind {kind}; one of {sorted(KINDS)}")
        jobs += [(kind, i, args.manifest, args.out) for i in range(args.first, args.first + int(count))]
    t0, done = time.time(), 0
    with multiprocessing.Pool(args.workers, maxtasksperchild=20) as pool:
        for name, n, secs in pool.imap_unordered(simulate, jobs):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} scenes ({time.time() - t0:.0f} s); last {name}: {n} objects, {secs:.1f} s",
                      flush=True)


if __name__ == "__main__":
    main()
