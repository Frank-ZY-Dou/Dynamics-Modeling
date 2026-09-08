"""RoboCasa-style placement replica (robosuite UniformRandomSampler): objects on a counter
region, validity = bounding-cylinder test (horizontal_radius sum in xy, z-extent overlap),
5000 attempts per object, then mesh-level G2, S4R repair and MuJoCo G5 on the real
collision pieces (exactly what MuJoCo simulates)."""
import json, os, math, sys, time, copy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); import simready  # noqa
from simready.scene import Body, Scene, MeshProxy
from simready.gates import verify_scene
from simready.gates.settle_mujoco import settle_and_measure
from simready.io.mjcf_io import load_mjcf_object, load_mjcf_pieces, body_from_mjcf_object
from simready.dsl import parse_program, compile_program, check_predicates
from simready.repair import repair_upright
from simready.repair.upright_s4r import reseat_on_supports, polish_full_mesh
import trimesh

ROOT = Path(os.environ.get("ROBOCASA_AIGEN_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "robocasa_assets" / "aigen_objs" / "aigen_objs")))   # RoboCasa aigen objects
ALL = sorted(ROOT.glob("*/*/model.xml"))
COUNTER = (0.0, 0.0, 0.9)            # counter centre x, y and top z
REGION = (0.30, 0.50)                # half extents of the placement region (x, y)


def counter_body():
    slab = trimesh.creation.box(extents=(2 * REGION[0] + 0.1, 2 * REGION[1] + 0.1, 0.05))
    return Body.from_mesh("counter", slab.vertices, slab.faces, center=np.array([COUNTER[0], COUNTER[1], COUNTER[2] - 0.025]),
                          fixed=True, tags={"support", "fixture"})


def sample_layout(rng, n, margin=0.0, attempts=5000):
    """robosuite-style: horizontal_radius from the collision geometry, z from bottom offset."""
    picks = [ALL[i] for i in rng.choice(len(ALL), size=n, replace=False)]
    placed = []
    for xml in picks:
        v, f = load_mjcf_object(xml)
        hr = float(np.linalg.norm(v[:, :2], axis=1).max())
        bot, top = float(v[:, 2].min()), float(v[:, 2].max())
        ok = False
        for _ in range(attempts):
            x = float(rng.uniform(-REGION[0] + hr, REGION[0] - hr)); y = float(rng.uniform(-REGION[1] + hr, REGION[1] - hr))
            yaw = float(rng.uniform(-math.pi, math.pi))
            valid = True
            for q in placed:
                if math.hypot(q["x"] - x, q["y"] - y) <= q["hr"] + hr + margin and True:
                    valid = False; break
            if valid:
                placed.append({"name": f"{xml.parent.parent.name}_{xml.parent.name}", "xml": str(xml), "x": x + COUNTER[0],
                               "y": y + COUNTER[1], "z": COUNTER[2] - bot, "yaw": yaw, "hr": hr})
                ok = True; break
        if not ok:
            return None
    return placed


def scene_from(placed):
    bodies = [counter_body()]
    for o in placed:
        q = (math.cos(o["yaw"] / 2), 0.0, 0.0, math.sin(o["yaw"] / 2))
        bodies.append(body_from_mjcf_object(o["name"], o["xml"], (o["x"], o["y"], o["z"]), quat_wxyz=q))
    return Scene(bodies)


def pieces_for(scene, placed):
    out = {}
    for o in placed:
        b = scene[o["name"]]
        c = b.meta["c_model"]
        out[o["name"]] = [(v - c, f) for v, f in load_mjcf_pieces(o["xml"])]   # body frame = AABB-centred
    return out


def run_cell(n, seed, verbose=False):
    rng = np.random.RandomState(seed)
    placed = sample_layout(rng, n)
    if placed is None:
        return {"n": n, "seed": seed, "sampler": "RandomizationError"}
    raw = scene_from(placed); pcs = pieces_for(raw, placed)
    rep = copy.deepcopy(raw)
    prog = parse_program("program\n  no_penetration(*, margin=0.003)\n  fixed(counter)\n  on_support(*, counter)   upright(*)\n  within(*, counter.top, inset=0.0)\n  minimize displacement(*)\n")
    spec = compile_program(prog, rep, s_min=0.05, ds_max=0.05, tail_iters=30)
    t0 = time.time()
    with MeshProxy(rep, faces=2000, slab_tops={"counter": COUNTER[2]}):
        res = repair_upright(rep, spec)
    reseat_on_supports(rep, spec)
    if verify_scene(rep).pen_pairs > 0:
        polish_full_mesh(rep, spec, iters=15); reseat_on_supports(rep, spec)
    dt = time.time() - t0
    vr, vp = verify_scene(raw), verify_scene(rep)
    out = {"n": n, "seed": seed, "sampler": "ok", "pen_raw": vr.pen_pairs, "maxpen_raw": vr.max_pen,
           "pen_rep": vp.pen_pairs, "rmsd": res.rmsd, "repair_s": dt,
           "preds_fail": [p[0] for p in check_predicates(prog, rep) if not p[1]]}
    for tag, sc in (("raw", raw), ("rep", rep)):
        r = settle_and_measure(sc, seconds=2.0, support_tops={"counter": COUNTER[2]}, pieces=pcs)
        out[f"{tag}_vmax"] = r.peak_speed; out[f"{tag}_dmax"] = r.peak_disp
    return out


if __name__ == "__main__":
    ns = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["6"])]
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0"])]
    rows = []
    for n in ns:
        for s in seeds:
            r = run_cell(n, s); rows.append(r)
            if r["sampler"] != "ok":
                print(f"N={n} s={s} sampler FAILED", flush=True); continue
            print(f"N={n} s={s} pen {r['pen_raw']:2d}->{r['pen_rep']:2d} maxPen={r['maxpen_raw']:.3f} rmsd={r['rmsd']:.3f} {r['repair_s']:.1f}s | MuJoCo raw {r['raw_vmax']:.2f} m/s {r['raw_dmax']:.3f} m -> repaired {r['rep_vmax']:.2f} m/s {r['rep_dmax']:.3f} m predsFail={r['preds_fail']}", flush=True)
    Path("results").mkdir(exist_ok=True); json.dump(rows, open("results/robocasa_diag.json", "w"), indent=1)
