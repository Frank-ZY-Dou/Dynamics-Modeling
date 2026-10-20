"""Videos of exported rollouts (scripts/export_rollouts.py) with the fast renderer (src/utils/rollout_render.py).

The predictions only, in real time at --fps (default 60). With step sizes above 1 the model predicts fewer frames per
second than the video shows (48 per second at step size 5, 24 at step size 10); the video interpolates the object
poses between predicted frames (positions linearly, rotations along the shortest arc). No text is drawn.

    python scripts/render_rollouts.py --rollouts rollouts/movi_b --scenes movi_1150,movi_772 --out videos/movi_b
    python scripts/render_rollouts.py --rollouts rollouts/movi_b --best 4 --data datasets/MOVi-B --out videos/movi_b

--best N renders the scenes scripts/build_demo_data.py would pick (lowest prediction error among the busier half of
the test scenes). With --config, a partial-view config (data.partial_view), rollouts from partial inputs are filmed
from the side of the depth camera that observed them, so the observed points face the viewer; --style points (the
default) draws the observed points opaque in the object's colour and the object see-through around them, --style solid
draws opaque objects with light points on them. Rollouts of a model with a completion network
(data.partial_view.completion) also show the points it added, smaller (and lighter with --style points).

Videos: <out>/<scene>_step_<s>[_dark].mp4.
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
MIN_ELEVATION = np.radians(5.0)                      # --config: lowest elevation of the filming direction


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
    ap.add_argument('--no-points', action='store_true',
                    help='rollouts from partial inputs: do not draw the observed points on the objects')
    ap.add_argument('--point-radius', type=float, default=0.025, help='radius (m) of the drawn observed points')
    ap.add_argument('--style', choices=['points', 'solid'], default='points',
                    help='rollouts from partial inputs: opaque points in the object colour and see-through objects '
                         '(points), or opaque objects with light points (solid)')
    ap.add_argument('--see-through', type=float, default=rollout_render.SEE_THROUGH,
                    help='--style points: opacity of the objects around their points')
    ap.add_argument('--config', default='',
                    help='partial-view config: film rollouts from partial inputs from the side of the observing camera')
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
    camera = None
    if args.config:
        import yaml
        from pathlib import Path
        from src.data.partial_view import PartialView
        path = args.config if os.path.isabs(args.config) else os.path.join(REPO, args.config)
        camera = PartialView(yaml.safe_load(open(path))['data']['partial_view'], Path(REPO))
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
            points, added, colours, view = None, None, None, None
            if 'object_index' in d.files:                               # partial inputs: the observed objects
                pal = rollout_render.PALETTE                            # colours by index in the scene
                colours = [pal[int(i) % len(pal)] for i in d['object_index']]
                if not args.no_points:
                    points = np.split(d['points'].astype(np.float64), np.cumsum(d['points_count'])[:-1])
                    if 'added_points' in d.files:                       # points a completion network added
                        added = np.split(d['added_points'].astype(np.float64), np.cumsum(d['added_count'])[:-1])
                if camera is not None:
                    # film from the observing camera's side, at an elevation of at least MIN_ELEVATION (some MOVi
                    # cameras sit below the target)
                    _, Rc, _, _ = camera.camera(scene, False, np.random.default_rng(0))
                    view = Rc[:3, 3] - np.asarray(camera.cfg['target'], np.float64)
                    h = np.linalg.norm(view[:2])
                    if view[2] < np.tan(MIN_ELEVATION) * h:
                        view = np.array([view[0], view[1], np.tan(MIN_ELEVATION) * h])
            name = f'{scene}_step_{step}' + ('_dark' if args.floor == 'dark' else '') + '.mp4'
            t0 = time.time()
            renderer.render(os.path.join(args.out, name), [meshes[s] for s in shapes], d['size'],
                            [str(m) == 'metal' for m in d['material']], poses, args.fps,
                            times=d['frame'] / FRAME_RATE, points=points, point_radius=args.point_radius,
                            colours=colours, view=view, style=args.style, see_through=args.see_through, added=added)
            print(f'{name}: {K} predicted frames at step size {step}, {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
