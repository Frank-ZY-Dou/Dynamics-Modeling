"""A GIF from rendered frames, as every clip of the README is encoded (no ffmpeg): every `--step`-th f_*.png of a
folder, resized to `--width`, one palette for the whole clip (median cut; half from the scene, half from the pixels
that differ between the first and the last frame, so that small moving objects keep their colours), 8x8 Bayer ordered
dithering, and the pixels a frame shares with the one shown before written as transparent (a pixel within `--gate` of
what is shown keeps it, so that render noise does not count as change).

    python viz/make_gif.py results/videos/pile_n20/blender_gate docs/media/pile_n20_gate.gif --width 600 --ms 130
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image

BAYER8 = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38],
                   [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                   [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) / 64.0 - 0.5
TRANSPARENT = 255


def _palette(pixels, k, w):
    """k colours by median cut over `pixels` (any number of RGB rows, laid out `w` wide for Pillow)."""
    pixels = pixels.reshape(-1, 3)
    pixels = pixels[: (len(pixels) // w) * w].reshape(-1, w, 3).astype(np.uint8)
    return np.asarray(Image.fromarray(pixels).quantize(colors=k, method=Image.Quantize.MEDIANCUT).getpalette()[: 3 * k],
                      dtype=np.float32).reshape(-1, 3)


def encode(paths, out, width, ms, spread=10.0, gate=3.0):
    """Write the frames `paths` (in order) as a looping GIF `out`, `width` pixels wide, `ms` per frame."""
    if not paths:
        raise ValueError(f"no frames for {out}")
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        frames.append(np.asarray(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS), dtype=np.float32))
    h, w, _ = frames[0].shape
    moving = np.abs(frames[0] - frames[-1]).max(-1) > 24
    pal = np.concatenate([_palette(np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0), 128, w),
                          _palette(np.concatenate([frames[0][moving], frames[len(frames) // 2][moving], frames[-1][moving]]), 127, w)])
    g = np.arange(64, dtype=np.float32) * 4 + 2                # nearest palette entry for every colour of a 6-bit grid
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lut = np.empty(len(grid), dtype=np.uint8)
    for i in range(0, len(grid), 16384):
        lut[i:i + 16384] = ((grid[i:i + 16384, None, :] - pal[None]) ** 2).sum(-1).argmin(1)
    thr = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w, None] * spread
    pal_bytes = list(pal.astype(np.uint8).reshape(-1)) + [0, 0, 0]
    out_frames, prev, seen = [], None, None
    for f in frames:
        q = np.clip(f + thr, 0, 255).astype(np.int32) >> 2
        idx = lut[(q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]]
        if prev is None:
            shown, seen = idx, f.copy()
        else:
            keep = (idx == prev) | (np.abs(f - seen).max(-1) <= gate)
            idx = np.where(keep, prev, idx).astype(np.uint8)
            seen = np.where(keep[..., None], seen, f)
            shown = np.where(keep, TRANSPARENT, idx).astype(np.uint8)
        prev = idx
        im = Image.fromarray(shown, mode="P"); im.putpalette(pal_bytes)
        out_frames.append(im)
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=ms, loop=0, transparency=TRANSPARENT,
                       disposal=1, optimize=False)
    print(out, len(out_frames), "frames", (w, h), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="a folder of f_*.png"); ap.add_argument("out")
    ap.add_argument("--width", type=int, required=True); ap.add_argument("--ms", type=int, required=True, help="milliseconds per frame")
    ap.add_argument("--step", type=int, default=1, help="every step-th frame")
    ap.add_argument("--spread", type=float, default=10.0, help="dither amplitude"); ap.add_argument("--gate", type=float, default=3.0)
    a = ap.parse_args()
    encode(sorted(glob.glob(os.path.join(a.frames, "f_*.png")))[:: a.step], a.out, a.width, a.ms, a.spread, a.gate)


if __name__ == "__main__":
    main()
