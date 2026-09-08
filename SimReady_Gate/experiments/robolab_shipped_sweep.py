"""G2 over every shipped RoboLab scene (already physics-settled by their pipeline)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import simready  # noqa
from simready.io.usd_io import load_scene_usda
from simready.gates import verify_scene
import os
SC = Path(os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "RoboLab"))) / "assets/scenes"
rows = []
for f in sorted(SC.glob("*.usda")):
    try:
        t0 = time.time(); sc = load_scene_usda(f)
        rep = verify_scene(sc)
        objp = [(a, b, s) for a, b, s in rep.pairs if not (sc[a].fixed or sc[b].fixed)]
        tabp = [(a, b, s) for a, b, s in rep.pairs if (sc[a].fixed != sc[b].fixed)]
        rows.append({"scene": f.name, "n_free": len(sc.free()), "pen_strict": rep.pen_pairs,
                     "pen_obj_obj": len(objp), "pen_obj_fixture": len(tabp),
                     "max_pen_obj_obj_mm": 1000 * max((-s for *_, s in objp), default=0.0),
                     "max_pen_obj_fixture_mm": 1000 * max((-s for *_, s in tabp), default=0.0),
                     "obj_obj_ge_1mm": sum(1 for *_, s in objp if -s >= 1e-3),
                     "floating": rep.floating, "contained": rep.contained, "dropped": sc.meta.get("dropped", []),
                     "t": round(time.time() - t0, 1)})
        r = rows[-1]; print(f"{f.name:52s} free={r['n_free']:2d} objobj={r['pen_obj_obj']} ({r['max_pen_obj_obj_mm']:.2f}mm) objfix={r['pen_obj_fixture']} ({r['max_pen_obj_fixture_mm']:.2f}mm) floating={r['floating']} contained={r['contained']} dropped={r['dropped']}", flush=True)
    except Exception as e:  # noqa: BLE001
        rows.append({"scene": f.name, "error": str(e)[:100]}); print(f.name, "ERROR", str(e)[:80], flush=True)
Path("results").mkdir(exist_ok=True); json.dump(rows, open("results/robolab_shipped_g2.json", "w"), indent=1)
ok = [r for r in rows if "error" not in r]
print(f"\nscenes={len(ok)} with object-object penetration: {sum(1 for r in ok if r['pen_obj_obj']>0)} (>=1 mm: {sum(1 for r in ok if r['obj_obj_ge_1mm']>0)}) | with floating objects: {sum(1 for r in ok if r['floating'])} | with contained bodies: {sum(1 for r in ok if r['contained'])} | with unresolved assets: {sum(1 for r in ok if r['dropped'])} | obj-fixture resting pen median {sorted(r['max_pen_obj_fixture_mm'] for r in ok)[len(ok)//2]:.2f} mm")
