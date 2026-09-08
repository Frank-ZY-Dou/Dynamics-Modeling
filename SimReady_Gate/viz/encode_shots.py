"""Encode rendered frame folders into mp4: gate (f_%05d.png) and settle (p0_/p1_/... side by side)."""
import glob, os, sys
sys.path.insert(0, "viz")
from render_frames import encode, hstack

def main(root, fps=30):
    g = os.path.join(root, "blender_gate")
    if os.path.isdir(g) and glob.glob(os.path.join(g, "f_*.png")):
        encode(g, os.path.join(root, "gate.mp4"), fps=fps); print("wrote", os.path.join(root, "gate.mp4"), len(glob.glob(os.path.join(g, "f_*.png"))), "frames")
    s = os.path.join(root, "blender_settle")
    if os.path.isdir(s):
        panels = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(s, "p*_*.png"))})
        n = min(len(glob.glob(os.path.join(s, f"{p}_*.png"))) for p in panels) if panels else 0
        for f in range(n):
            hstack([os.path.join(s, f"{p}_{f:05d}.png") for p in panels], os.path.join(s, f"f_{f:05d}.png"))
        if n:
            encode(s, os.path.join(root, "settle.mp4"), fps=fps); print("wrote", os.path.join(root, "settle.mp4"), n, "frames x", len(panels), "panels")

if __name__ == "__main__":
    main(sys.argv[1])
