"""G0 sweep over RoboCasa AI-generated objects: collision-piece count, watertightness of
the visual mesh, and collision-vs-visual volume ratio (convex decomposition bloat)."""
import json, os, sys, time
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import simready  # noqa
from simready.io.mjcf_io import load_mjcf_object
ROOT = Path(os.environ.get("ROBOCASA_AIGEN_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "robocasa_assets" / "aigen_objs" / "aigen_objs")))   # RoboCasa aigen objects
rows = []
t0 = time.time()
for xml in sorted(ROOT.glob("*/*/model.xml")):
    try:
        vc, fc = load_mjcf_object(xml, collision_group=0, prefer_visual_if_empty=False)
        vv, fv = load_mjcf_object(xml, collision_group=1, prefer_visual_if_empty=False)
        mc = trimesh.Trimesh(vc, fc, process=False); mv = trimesh.Trimesh(vv, fv, process=False)
        n_col = sum(1 for line in open(xml) if '<geom' in line and 'group="0"' in line)
        vol_c = float(abs(mc.volume)) if mc.is_volume else float(mc.convex_hull.volume)
        vol_v = float(abs(mv.volume)) if mv.is_volume else float(mv.convex_hull.volume)
        ext = (vv.max(0) - vv.min(0)).tolist()
        rows.append({"obj": str(xml.parent.relative_to(ROOT)), "category": xml.parent.parent.name, "n_collision": n_col,
                     "visual_watertight": bool(mv.is_watertight), "visual_volume_ok": bool(mv.is_volume),
                     "vol_ratio": vol_c / max(vol_v, 1e-12), "extent": ext, "visual_faces": int(len(fv)), "collision_faces": int(len(fc))})
    except Exception as e:  # noqa: BLE001
        rows.append({"obj": str(xml.parent.relative_to(ROOT)), "category": xml.parent.parent.name, "error": str(e)[:80]})
    if len(rows) % 200 == 0: print(len(rows), f"{time.time()-t0:.0f}s", flush=True)
Path("results").mkdir(exist_ok=True)
json.dump(rows, open("results/robocasa_aigen_g0.json", "w"), indent=1)
ok = [r for r in rows if "error" not in r]
print(f"objects={len(rows)} errors={len(rows)-len(ok)}")
print("visual watertight:", sum(r['visual_watertight'] for r in ok), "/", len(ok))
import statistics as st
print("collision pieces: median", st.median(r['n_collision'] for r in ok), "max", max(r['n_collision'] for r in ok), "single-piece:", sum(1 for r in ok if r['n_collision']==1))
vr = [r['vol_ratio'] for r in ok]; print("vol ratio collision/visual: median", round(st.median(vr),3), "p90", round(sorted(vr)[int(0.9*len(vr))],3), ">1.5:", sum(1 for x in vr if x>1.5))
