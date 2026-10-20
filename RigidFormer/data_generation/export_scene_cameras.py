"""Write Kubric's camera of every scene of the Hugging Face dataset to a JSON file (data.partial_view.camera_file).

The cameras are static: K (pixels, Kubric's convention: the camera looks along its -z axis) and the camera-to-world
matrix R of every scene, keyed by the converted folder name (movi_<n>).

    python data_generation/export_scene_cameras.py --dataset RigidFormer-MOVi --config movi_b \\
        --out data_generation/scene_cameras_MoviB.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='local copy of the dataset repository (data/, assets/)')
    ap.add_argument('--config', required=True, choices=['movi_sphere', 'movi_a', 'movi_b'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    scenes = {}
    for f in sorted((Path(args.dataset) / 'data' / args.config).glob('*.parquet')):
        for row in pq.read_table(f, columns=['scene', 'camera', 'resolution']).to_pylist():
            cam = row['camera']
            Q = np.asarray(cam['quaternions'], np.float64)
            P = np.asarray(cam['positions'], np.float64)
            if np.abs(Q - Q[0]).max() > 1e-6 or np.abs(P - P[0]).max() > 1e-6:
                raise ValueError(f"{row['scene']}: the camera moves; a static camera is expected")
            if not np.allclose(np.asarray(cam['R'], np.float64)[:3, 3], P[0], atol=1e-4):
                raise ValueError(f"{row['scene']}: R does not place the camera at its recorded position")
            scenes[row['scene']] = {'K': np.round(np.asarray(cam['K'], np.float64), 6).tolist(),
                                    'R': np.round(np.asarray(cam['R'], np.float64), 8).tolist(),
                                    'resolution': list(row['resolution'])}
    order = sorted(scenes, key=lambda name: int(name.split('_')[1]))
    lines = [f'{json.dumps(name)}: {json.dumps(scenes[name])}' for name in order]        # one scene per line
    Path(args.out).write_text('{"dataset": %s, "source": "frankzydou/RigidFormer-MOVi", "scenes": {\n%s\n}}\n'
                              % (json.dumps(args.config), ',\n'.join(lines)))
    print(f'{args.config}: {len(scenes)} cameras -> {args.out}')


if __name__ == '__main__':
    main()
