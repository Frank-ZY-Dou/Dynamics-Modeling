"""Render an OMX rollout npz from rollout_superdex.py. The frames are drawn by
MuJoCo's offscreen renderer from the joint trajectories in the npz (the
dynamics were computed by SuperDex), with the framing of the released videos:
left = prediction (white arm), right = ground truth (green arm), each with a
red force arrow at the gripper. The renderer is the MuJoCo Warp one
(mjwarp/render_mjwarp.py), which reads only the npz; this entry point keeps the
SuperDex workflow self-contained.

    python superdex/render_superdex.py --npz rollout.npz --out rollout.mp4
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mjwarp")))

from render_mjwarp import main  # noqa: E402

if __name__ == "__main__":
    main()
