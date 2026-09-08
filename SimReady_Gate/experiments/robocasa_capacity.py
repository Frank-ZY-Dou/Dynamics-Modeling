"""RoboCasa placement versus the gate on a counter region, with RoboCasa's REAL validity test.

For each region size and object count N, over seeds:
  sampler   RoboCasa's UniformRandomSampler replica (experiments/robocasa_sampler.py: reg_bbox
            OBB separating-axis test, corner-in-region, min(w,d)/2 range buffer, 5000 attempts):
            success rate, and — on the layouts it accepts — mesh-level penetration on the
            objects' own collision geometry (what MuJoCo simulates) and how far the collision
            geometry sits below the reg_bbox bottom (the counter penetration its z rule leaves)
  gate      objects placed at random, overlapping, then S4R under within/on_support/upright:
            success = pen 0 on the collision geometry and every predicate satisfied
Objects: the same draw per seed for both columns. Writes results/robocasa_capacity.json.
"""
import json, math, sys, time, copy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); import simready  # noqa
from simready.scene import Body, Scene, MeshProxy
from simready.gates import verify_scene
from simready.io.mjcf_io import load_mjcf_object, body_from_mjcf_object
from simready.dsl import parse_program, compile_program, check_predicates
from simready.repair import repair_upright
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh
import robocasa_layouts as RL
from robocasa_sampler import ObjectSpec, sample_layout, PlacementError
import trimesh

TOP = RL.COUNTER[2]
PROGRAM = ("program\n  no_penetration(*, margin=0.003)\n  fixed(counter)  fixed(region)\n  on_support(*, counter)   upright(*)\n"
           "  within(*, region, inset=0.0)\n  minimize displacement(*)\n")


def counter(region):
    slab = trimesh.creation.box(extents=(2 * region[0] + 0.2, 2 * region[1] + 0.2, 0.05))
    return Body.from_mesh("counter", slab.vertices, slab.faces, center=np.array([0.0, 0.0, TOP - 0.025]), fixed=True, tags={"support", "fixture"})


def region_body(region):
    """A thin marker body giving `within` the placement rectangle (sunk below the counter so it is never a contact)."""
    box = trimesh.creation.box(extents=(2 * region[0], 2 * region[1], 0.001))
    return Body.from_mesh("region", box.vertices, box.faces, center=np.array([0.0, 0.0, TOP - 0.5]), fixed=True, tags={"marker"})


def pick(rng, n, max_ext=0.25):
    out = []
    while len(out) < n:
        xml = RL.ALL[rng.randint(len(RL.ALL))]
        try:
            v, f = load_mjcf_object(xml); spec = ObjectSpec(xml)
        except Exception:  # noqa: BLE001
            continue
        if (v.max(0) - v.min(0)).max() <= max_ext:
            out.append((xml, v, spec))
    return out


def scene_of(objs, region, placements):
    bodies = [counter(region), region_body(region)]
    for k, ((xml, v, spec), (x, y, z, yaw)) in enumerate(zip(objs, placements)):
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        bodies.append(body_from_mjcf_object(f"o{k}", xml, (x, y, z), quat_wxyz=q))
    return Scene(bodies)


def sampler_try(rng, objs, region):
    """RoboCasa's own placement; returns (ok, scene_or_None)."""
    try:
        placements = sample_layout(rng, [o[2] for o in objs], (-region[0], region[0]), (-region[1], region[1]), base=(0.0, 0.0, TOP))
    except PlacementError:
        return False, None
    return True, scene_of(objs, region, placements)


def gate_try(rng, objs, region):
    placements = []
    for xml, v, spec in objs:
        bot = float(v[:, 2].min())
        placements.append((float(rng.uniform(-region[0], region[0])), float(rng.uniform(-region[1], region[1])), TOP - bot, float(rng.uniform(-math.pi, math.pi))))
    sc = scene_of(objs, region, placements)
    prog = parse_program(PROGRAM)
    spec = compile_program(prog, sc, s_min=0.05, ds_max=0.05, tail_iters=30)
    t0 = time.time()
    with MeshProxy(sc, faces=1500, slab_tops={"counter": TOP}):
        repair_upright(sc, spec)
    reseat_on_supports(sc, spec)
    rep = verify_scene(sc)
    if rep.pen_pairs > 0:
        polish_full_mesh(sc, spec, iters=15); reseat_on_supports(sc, spec); rep = verify_scene(sc)
    preds = check_predicates(prog, sc)
    ok = rep.pen_pairs == 0 and all(p[1] for p in preds)
    return ok, rep.pen_pairs, [p[0] for p in preds if not p[1]], time.time() - t0


if __name__ == "__main__":
    regions = [(0.15, 0.15), (0.2, 0.2), (0.25, 0.25)]
    Ns = [4, 6, 8, 10, 12]
    seeds = [0, 1, 2, 3, 4]
    rows = []
    for region in regions:
        for n in Ns:
            s_ok = g_ok = 0; s_pen_layouts = 0; s_pairs = []; s_below = []; t0 = time.time(); fails = []; times = []
            for seed in seeds:
                rng = np.random.RandomState(1000 * seed + n)
                objs = pick(rng, n)
                ok, sc = sampler_try(np.random.RandomState(1000 * seed + n + 1), objs, region)
                s_ok += int(ok)
                if ok:
                    rep = verify_scene(sc)
                    obj_pairs = [(x, y, s) for x, y, s in rep.pairs if s < 0 and not sc[x].fixed and not sc[y].fixed]
                    s_pairs.append(len(obj_pairs)); s_pen_layouts += int(len(obj_pairs) > 0)
                    s_below.extend(round(1000 * float(TOP - b.world_aabb()[0][2]), 2) for b in sc.free())
                g, pen, f, dt = gate_try(np.random.RandomState(1000 * seed + n + 2), objs, region)
                g_ok += int(g); fails += f; times.append(dt)
            row = {"region": [2 * region[0], 2 * region[1]], "n": n, "seeds": len(seeds), "sampler_ok": s_ok, "gate_ok": g_ok,
                   "sampler_layouts_with_obj_penetration": s_pen_layouts, "sampler_obj_obj_pairs": s_pairs,
                   "sampler_collision_below_counter_mm": s_below, "gate_fails": fails, "gate_time_s": times}
            rows.append(row)
            print(f"region {2*region[0]:.1f}x{2*region[1]:.1f} m  N={n:2d}  sampler {s_ok}/{len(seeds)} (layouts with mesh penetration {s_pen_layouts}, pairs {s_pairs})"
                  f"  gate {g_ok}/{len(seeds)}  {time.time()-t0:.0f}s {fails[:2]}", flush=True)
    Path(HERE.parent / "results").mkdir(exist_ok=True)
    json.dump(rows, open(HERE.parent / "results" / "robocasa_capacity.json", "w"), indent=1)
