"""Videos of exported rollouts (scripts/export_rollouts.py) with the fast renderer (src/utils/rollout_render.py).

The predictions only, in real time at --fps (default 60). With step sizes above 1 the model predicts fewer frames per
second than the video shows (48 per second at step size 5, 24 at step size 10); the video interpolates the object
poses between predicted frames (positions linearly, rotations along the shortest arc). No text is drawn.

    python scripts/render_rollouts.py --rollouts rollouts/movi_b --scenes movi_1150,movi_772 --out videos/movi_b
    python scripts/render_rollouts.py --rollouts rollouts/movi_b --best 4 --data datasets/MOVi-B --out videos/movi_b

--best N renders the scenes scripts/build_demo_data.py would pick (lowest prediction error among the busier half of
the test scenes). Videos: <out>/<scene>_step_<s>[_dark].mp4.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from src.utils import rollout_render  # noqa: E402

MESHES = os.path.join(REPO, 'data_generation', 'objects')
FRAME_RATE = 240                                     # physical frames per second of the MOVi conversions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rollouts', required=True, help='export folder of scripts/export_rollouts.py')
    ap.add_argument('--scenes', default='', help='comma-separated scene names')
    ap.add_argument('--best', type=int, default=0, help='the N lowest-error scenes (needs --data)')
    ap.add_argument('--data', default='', help='dataset folder with the ground truth (vis_traj.npy), for --best')
    ap.add_argument('--steps', default='1,5,10')
    ap.add_argument('--floor', choices=sorted(rollout_render.LOOKS), default='light')
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--size', default='1280x720')
    ap.add_argument('--supersample', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.rollouts, 'index.json')))
    if args.best:
        from build_demo_data import select
        if not args.data:
            raise SystemExit('--best needs --data (the dataset folder with vis_traj.npy)')
        scenes = select(args.rollouts, index, args.data, args.best)
    else:
        scenes = [s for s in args.scenes.split(',') if s]
    if not scenes:
        raise SystemExit('no scenes: pass --scenes or --best')
    steps = [int(s) for s in args.steps.split(',')]
    width, height = (int(v) for v in args.size.split('x'))
    renderer = rollout_render.get_renderer(width, height, args.supersample, floor=args.floor)
    os.makedirs(args.out, exist_ok=True)
    for scene in scenes:
        for step in steps:
            d = np.load(os.path.join(args.rollouts, f'{scene}_ss{step}.npz'))
            shapes = [str(s) for s in d['shape']]
            meshes = rollout_render.load_render_meshes(MESHES, shapes)
            q = d['rotation'].astype(np.float64)                                   # (K, n, 4) w, x, y, z
            K, n = q.shape[:2]
            poses = []
            for i in range(n):
                P = np.tile(np.eye(4), (K, 1, 1))
                P[:, :3, :3] = Rotation.from_quat(q[:, i][:, [1, 2, 3, 0]]).as_matrix()
                P[:, :3, 3] = d['translation'][:, i]
                poses.append(P)
            name = f'{scene}_step_{step}' + ('_dark' if args.floor == 'dark' else '') + '.mp4'
            t0 = time.time()
            renderer.render(os.path.join(args.out, name), [meshes[s] for s in shapes], d['size'],
                            [str(m) == 'metal' for m in d['material']], poses, args.fps,
                            times=d['frame'] / FRAME_RATE)
            print(f'{name}: {K} predicted frames at step size {step}, {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
