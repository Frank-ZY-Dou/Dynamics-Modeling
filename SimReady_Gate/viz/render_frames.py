"""Headless frame rendering for SimReady Gate videos (Polyscope, EGL backend, no X server).

A *frame* is {body_name: (center, rotation, scale)}; every body is drawn as its model-frame
mesh scaled about its own origin (the S4R reference centre), rotated and translated. Fixed
bodies are registered once. Text overlays are drawn with PIL after the screenshot.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

_PS = None
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PALETTE = np.array([
    [0.36, 0.55, 0.78], [0.93, 0.62, 0.25], [0.45, 0.70, 0.45], [0.75, 0.45, 0.72], [0.85, 0.75, 0.30],
    [0.40, 0.72, 0.75], [0.80, 0.50, 0.45], [0.55, 0.55, 0.80], [0.60, 0.75, 0.35], [0.90, 0.55, 0.60],
    [0.50, 0.65, 0.60], [0.75, 0.60, 0.40],
])
RED = np.array([0.86, 0.18, 0.16])
TABLE = np.array([0.78, 0.74, 0.68])
FLOOR = np.array([0.93, 0.93, 0.92])


def polyscope(width=1280, height=720):
    global _PS
    import polyscope as ps
    if _PS is None:
        ps.set_use_prefs_file(False)
        ps.set_verbosity(0)
        ps.set_up_dir("z_up")
        ps.set_front_dir("x_front")
        ps.set_window_size(width, height)
        ps.set_background_color((1.0, 1.0, 1.0))
        ps.set_ground_plane_mode("none")
        ps.set_SSAA_factor(2)
        ps.init(backend="openGL3_egl")
        _PS = ps
    return _PS


class FrameRenderer:
    """Register the bodies once; call `render(frame, colors, out_png, texts)` per frame."""

    def __init__(self, bodies, width=1280, height=720, floor_z=None, floor_extent=2.0):
        self.ps = polyscope(width, height)
        self.ps.remove_all_structures()
        self.ps.set_window_size(width, height)      # the init size only applies to the first renderer
        self.width, self.height = width, height
        self.bodies = {b.name: b for b in bodies}
        self.meshes = {}
        for b in bodies:
            m = self.ps.register_surface_mesh(b.name, b.world_vertices(), b.faces, smooth_shade=True)
            m.set_material("clay")
            m.set_edge_width(0.0)
            self.meshes[b.name] = m
        if floor_z is not None:
            e = floor_extent
            V = np.array([[-e, -e, floor_z], [e, -e, floor_z], [e, e, floor_z], [-e, e, floor_z]])
            F = np.array([[0, 1, 2], [0, 2, 3]])
            fl = self.ps.register_surface_mesh("floor", V, F, smooth_shade=False)
            fl.set_material("clay"); fl.set_color(FLOOR.tolist()); fl.set_edge_width(0.0)
        self.set_camera()

    def set_camera(self, target=(0.35, 0.0, 0.05), dist=1.45, elev_deg=32.0, azim_deg=-35.0):
        t = np.asarray(target, dtype=float)
        el, az = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
        eye = t + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.ps.look_at(camera_location=eye.tolist(), target=t.tolist(), fly_to=False)

    def render(self, frame, colors, out_png, texts=(), hidden=()):
        """frame: {name: (center, R, scale)} for the bodies to move; colors: {name: rgb}."""
        for name, (c, R, s) in frame.items():
            b = self.bodies[name]
            V = (np.asarray(R) @ (float(s) * b.verts).T).T + np.asarray(c)
            self.meshes[name].update_vertex_positions(V)
        for name, col in colors.items():
            self.meshes[name].set_color([float(x) for x in col])
        for name in self.meshes:
            self.meshes[name].set_enabled(name not in hidden)
        out_png = str(out_png)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        self.ps.set_window_size(self.width, self.height)
        self.ps.screenshot(out_png, transparent_bg=False)
        if texts:
            overlay(out_png, texts)


def overlay(png, texts, pad=8):
    """texts: list of (text, (x, y), size, rgb, bold); each line sits on a translucent white band
    so it stays readable over the geometry."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(png).convert("RGBA")
    layer = Image.new("RGBA", im.size, (255, 255, 255, 0)); d = ImageDraw.Draw(layer)
    for text, (x, y), size, rgb, bold in texts:
        try:
            font = ImageFont.truetype(_FONT_BOLD if bold else _FONT, size)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
        x0, y0, x1, y1 = d.textbbox((x, y), text, font=font)
        d.rounded_rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), radius=6, fill=(255, 255, 255, 200))
        d.text((x, y), text, fill=tuple(int(255 * v) for v in rgb) + (255,), font=font)
    Image.alpha_composite(im, layer).convert("RGB").save(png)


def hstack(pngs, out_png, gap=8):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs]
    h = max(im.height for im in ims); w = sum(im.width for im in ims) + gap * (len(ims) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255)); x = 0
    for im in ims:
        canvas.paste(im, (x, 0)); x += im.width + gap
    canvas.save(out_png)


def encode(frame_dir, out_mp4, fps=30, pattern="f_%05d.png"):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frame_dir, pattern),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4)]
    subprocess.run(cmd, check=True)
    return out_mp4


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def interpolate(a, b, t):
    """Linear pose interpolation between two frames (same keys); rotations via slerp on yaw-only is
    unnecessary here: blend the matrices and re-orthonormalize (steps are small)."""
    out = {}
    for k in a:
        c0, R0, s0 = a[k]; c1, R1, s1 = b[k]
        R = (1 - t) * np.asarray(R0) + t * np.asarray(R1)
        U, _, Vt = np.linalg.svd(R); R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        out[k] = ((1 - t) * np.asarray(c0) + t * np.asarray(c1), R, (1 - t) * s0 + t * s1)
    return out
