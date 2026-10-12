"""Visual check of a converted MOVi scene against the original rendering.

The converted per-frame vertices (with the metadata object sizes from the sidecar) are projected into the scene's
Kubric camera (metadata.json: intrinsics K, camera pose per frame) and drawn over the original video (rgba.mp4): each
object is filled with its own translucent colour, so it must cover exactly the rendered object at every frame. A wrong
size, pose or frame index shows up as a coloured rim or an uncovered part of the object; without --object_scale_file
the stored meshes are drawn in asset units (MOVi-Sphere: every sphere ~1 m across).

  python data_generation/visualize_overlay.py --raw <raw>/MOVi-B/7 --converted datasets/MOVi-B/movi_7 \\
      --object_scale_file data_generation/object_scales_MoviB.json --frames 0 60 120 240 479 --out movi_7.png
  # every frame as a video (original | overlay), needs ffmpeg:
  python data_generation/visualize_overlay.py ... --video movi_7.mp4

Needs ffmpeg (decoding rgba.mp4), numpy, Pillow.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.object_scale import ObjectScale  # noqa: E402

COLORS = [(230, 50, 50), (50, 120, 230), (240, 200, 30), (180, 60, 230), (30, 200, 200), (240, 130, 40),
          (240, 100, 180), (110, 190, 110), (150, 110, 70), (90, 90, 90)]


def video_frames(path: Path, size):
    """All frames of rgba.mp4 as an (T, H, W, 3) uint8 array."""
    w, h = size
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(points, cam, t):
    """World points (N, 3) -> pixel coordinates (N, 2) and camera depth (N,). Kubric camera: R(q) maps camera to world,
    the camera looks along -z; K maps camera coordinates to pixels."""
    R = quat_wxyz_to_matrix(np.asarray(cam['quaternions'][t], dtype=np.float64))
    pc = (np.asarray(points, np.float64) - np.asarray(cam['positions'][t], np.float64)) @ R    # R^T (x - c), row form
    h = pc @ np.asarray(cam['K'], np.float64).T
    return h[:, :2] / h[:, 2:3], -pc[:, 2]


def overlay(frame, objects, faces, cam, t, alpha=0.45):
    """frame (H, W, 3) uint8; objects: list of (Nv, 3) world vertices; faces: list of (F, 3)."""
    img = Image.fromarray(frame).convert('RGBA')
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # far objects first, so nearer ones are drawn on top
    depth = [project(V, cam, t)[1].mean() for V in objects]
    for i in sorted(range(len(objects)), key=lambda k: -depth[k]):
        uv, z = project(objects[i], cam, t)
        if (z <= 0).any():
            raise ValueError(f'object {i} is (partly) behind the camera at frame {t}')
        col = COLORS[i % len(COLORS)] + (int(255 * alpha),)
        for f in faces[i]:
            draw.polygon([tuple(p) for p in uv[f]], fill=col)
    return np.asarray(Image.alpha_composite(img, layer).convert('RGB'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raw', required=True, help='raw scene directory (metadata.json, rgba.mp4)')
    ap.add_argument('--converted', required=True, help='converted scene directory (frame_*/obj_*_pos.npy, meshes/)')
    ap.add_argument('--object_scale_file', default=None, help='size sidecar of the converted dataset')
    ap.add_argument('--frames', type=int, nargs='+', default=[0, 60, 120, 240, 479])
    ap.add_argument('--out', default=None, help='PNG grid: rows = frames, columns = original | overlay')
    ap.add_argument('--video', default=None, help='MP4 of every frame (original | overlay)')
    ap.add_argument('--zoom', type=int, default=2)
    args = ap.parse_args()
    if not (args.out or args.video):
        ap.error('pass --out and/or --video')

    raw, conv = Path(args.raw), Path(args.converted)
    meta = json.load(open(raw / 'metadata.json'))
    cam = meta['camera']
    frames = video_frames(raw / 'rgba.mp4', meta['metadata']['resolution'])
    n_obj = len([f for f in os.listdir(conv / 'frame_00000') if re.fullmatch(r'obj_\d+_pos\.npy', f)])
    faces = [np.load(conv / 'meshes' / f'obj_{i}_faces.npy') for i in range(n_obj)]
    scale = ObjectScale(args.object_scale_file) if args.object_scale_file else None
    if frames.shape[0] != len(cam['positions']):
        raise ValueError(f'rgba.mp4 has {frames.shape[0]} frames, the camera {len(cam["positions"])}')

    def objects(t):
        out = []
        for i in range(n_obj):
            p = conv / f'frame_{t:05d}' / f'obj_{i}_pos.npy'
            V = np.load(p)
            out.append(scale.apply(p, V) if scale is not None else V)
        return out

    def panel(t):
        both = np.concatenate([frames[t], overlay(frames[t], objects(t), faces, cam, t)], axis=1)
        im = Image.fromarray(both)
        return im.resize((im.width * args.zoom, im.height * args.zoom), Image.NEAREST)

    if args.out:
        rows = [panel(t) for t in args.frames]
        grid = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
        y = 0
        for t, r in zip(args.frames, rows):
            ImageDraw.Draw(r).text((4, 4), f'frame {t}', fill=(0, 0, 0))
            grid.paste(r, (0, y))
            y += r.height
        grid.save(args.out)
        print(f'wrote {args.out} (frames {args.frames})')
    if args.video:
        first = panel(0)
        proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
                                 f'{first.width}x{first.height}', '-r', '30', '-i', '-', '-pix_fmt', 'yuv420p',
                                 args.video], stdin=subprocess.PIPE)
        for t in range(frames.shape[0]):
            proc.stdin.write(np.asarray(panel(t) if t else first).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('ffmpeg failed to write the video')
        print(f'wrote {args.video} ({frames.shape[0]} frames at 30 fps)')


if __name__ == '__main__':
    main()
