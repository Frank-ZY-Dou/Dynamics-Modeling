"""Synthetic tabletop check: Kubric meshes spawned densely on a table, verified,
repaired with a DSL program, verified again, predicates checked."""
import math, os, sys, time
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simready.scene import Body, Scene
from simready.gates import verify_scene
from simready.dsl import parse_program, compile_program, check_predicates
from simready.repair import repair_upright

# the Kubric collision meshes shipped with the S4R release (Penetration_Solving/data/kubric_pool)
POOL = Path(os.environ.get("KUBRIC_POOL", str(Path(__file__).resolve().parents[2] / "Penetration_Solving" / "data" / "kubric_pool")))

def main(N=8, seed=3, target=0.12):
    rng = np.random.RandomState(seed)
    table = trimesh.creation.box(extents=(0.8, 0.8, 0.04))
    bodies = [Body.from_mesh("table", table.vertices, table.faces, center=np.array([0.55, 0.0, 0.73]),
                             fixed=True, tags={"support"})]
    files = sorted(POOL.glob("*/collision_geometry.obj"))
    pick = rng.choice(len(files), size=N, replace=False)
    for i, fi in enumerate(pick):
        m = trimesh.load(str(files[fi]), process=True, force="mesh")
        v = np.asarray(m.vertices); f = np.asarray(m.faces)
        v = v - 0.5 * (v.min(0) + v.max(0)); v *= target / (v.max(0) - v.min(0)).max()
        yaw = rng.uniform(0, 2 * math.pi); tilt = math.radians(rng.uniform(-25, 25))
        from simready.repair.upright_s4r import rotz, rotx
        R = rotz(yaw) @ rotx(tilt)
        xy = np.array([0.55, 0.0]) + rng.uniform(-0.09, 0.09, size=2)   # dense: overlaps guaranteed
        b = Body.from_mesh(f"obj{i}", v, f, center=np.array([xy[0], xy[1], 0.0]), rotation=R)
        b.center[2] = 0.75 + b.support_offset(np.array([0, 0, 1.0]))
        bodies.append(b)
    scene = Scene(bodies)
    prog = parse_program(f"""
program
  no_penetration(*, margin=0.005)
  fixed(table)
  on_support(*, table)   upright(*)
  within(*, table.top, inset=0.03)
  left_of(obj0, obj1, gap=0.05)
  minimize displacement(*)
""")
    sup = {b.name: 0.75 for b in scene.free()}
    before = verify_scene(scene, supports=sup)
    print("G2 before:", before.summary())
    spec = compile_program(prog, scene, s_min=0.05, ds_max=0.05, tail_iters=25)
    t0 = time.time()
    res = repair_upright(scene, spec)
    dt = time.time() - t0
    after = verify_scene(scene, supports=sup)
    print(f"G3 repair: pen {res.pen_before} -> {res.pen_after}, steps={res.steps}, rmsd={res.rmsd:.4f}, {dt:.1f}s")
    print("G2 after: ", after.summary())
    for name, ok, val in check_predicates(prog, scene):
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {'not over the support' if val is None else f'{val:+.4f}'}")

if __name__ == "__main__":
    main()
