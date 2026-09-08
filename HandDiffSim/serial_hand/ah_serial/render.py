"""Offscreen rendering helpers (EGL) and text overlays for stills / videos."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int, kind: str = "cjk") -> ImageFont.FreeTypeFont:
    path = {"cjk": FONT_CJK, "mono": FONT_MONO, "sans": FONT_SANS}[kind]
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x2E80 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F


def text_runs(text: str):
    """Split into (run, is_cjk) so CJK glyphs use the fallback font and everything else a Latin font."""
    runs, cur, cur_cjk = [], "", None
    for ch in text:
        c = _is_cjk(ch)
        if cur_cjk is None or c == cur_cjk:
            cur += ch
        else:
            runs.append((cur, cur_cjk))
            cur = ch
        cur_cjk = c
    if cur:
        runs.append((cur, cur_cjk))
    return runs


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, size: int, fill=(235, 235, 235), latin="sans"):
    """Draw mixed Chinese/Latin text with per-run fonts; returns the total width."""
    f_cjk, f_lat = font(size, "cjk"), font(size, latin)
    x, y = xy
    for run, is_cjk in text_runs(text):
        f = f_cjk if is_cjk else f_lat
        draw.text((x, y), run, fill=fill, font=f)
        x += draw.textlength(run, font=f)
    return x - xy[0]


def text_width(draw: ImageDraw.ImageDraw, text: str, size: int, latin="sans") -> float:
    f_cjk, f_lat = font(size, "cjk"), font(size, latin)
    return sum(draw.textlength(run, font=(f_cjk if is_cjk else f_lat)) for run, is_cjk in text_runs(text))


def make_camera(lookat=(0.03, 0.0, 0.10), distance=0.30, azimuth=160.0, elevation=-20.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


class Offscreen:
    def __init__(self, model: mujoco.MjModel, size=(720, 720), cam: mujoco.MjvCamera | None = None, show_sites=False):
        self.model = model
        # the offscreen framebuffer defaults to 640x480 unless the XML sets <visual><global offwidth/>; enlarge in memory
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(size[0]))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(size[1]))
        self.renderer = mujoco.Renderer(model, height=size[1], width=size[0])
        self.cam = cam or make_camera()
        self.opt = mujoco.MjvOption()
        if not show_sites:
            self.opt.sitegroup[3] = 0  # hide the CAD frame sites
        self.opt.geomgroup[3] = 0

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def compose_side_by_side(left: np.ndarray, right: np.ndarray, title_left: str, title_right: str, footer_lines: list[str],
                         header_h=72, footer_h=None, footer_font=22, title_font=30) -> np.ndarray:
    """Two panels with titles above and monospace status lines below."""
    footer_h = footer_h or (16 + footer_font * 1.35 * max(1, len(footer_lines)) + 10)
    footer_h = int(footer_h)
    h, w = left.shape[:2]
    canvas = Image.new("RGB", (2 * w, h + header_h + footer_h), (18, 20, 26))
    canvas.paste(Image.fromarray(left), (0, header_h))
    canvas.paste(Image.fromarray(right), (w, header_h))
    draw = ImageDraw.Draw(canvas)
    for x0, title in ((0, title_left), (w, title_right)):
        tw = text_width(draw, title, title_font)
        draw_text(draw, (x0 + (w - tw) / 2, (header_h - title_font) / 2), title, title_font)
    draw.line([(w, 0), (w, header_h + h)], fill=(70, 74, 84), width=2)
    y = header_h + h + 10
    for line in footer_lines:
        has_cjk = any(_is_cjk(ch) for ch in line)
        draw_text(draw, (18, y), line, footer_font, fill=(220, 220, 220), latin="sans" if has_cjk else "mono")
        y += int(footer_font * 1.35)
    return np.asarray(canvas)


class FFmpegWriter:
    """Pipe RGB frames to ffmpeg (libx264, yuv420p)."""

    def __init__(self, path: str | Path, size: tuple[int, int], fps: int = 30, crf: int = 18):
        w, h = size
        w, h = w - (w % 2), h - (h % 2)
        self.size = (w, h)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame: np.ndarray):
        w, h = self.size
        self.proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc}")
