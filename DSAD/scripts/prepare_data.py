#!/usr/bin/env python
"""Build the GoP folders of UVG and DAVIS from the downloaded sources, and verify them.

    python scripts/prepare_data.py uvg    --src <dir with the UVG .7z or .yuv files>
    python scripts/prepare_data.py davis  --src <DAVIS-2017-trainval-Full-Resolution.zip or its folder>
    python scripts/prepare_data.py verify [--set uvg_davis bunny | --set all]

uvg and davis write one folder per GoP under --out (default data/), with frames f00001.png,
f00002.png, and so on, at 1280x720. verify compares the pixels of each GoP with
data/checksums/uvg_davis.sha256 (--set uvg_davis, the default) or bunny.sha256. Frames identical
to the paper's need ffmpeg 4.4.2 and Pillow 12.

Requires ffmpeg on PATH, and 7z (p7zip) or the py7zr package for the UVG archives. Downloads,
folders and licenses: data/README.md.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
BUNNY = REPO / "baselines" / "D2GV" / "dataset" / "bunny"

UVG = {  # scene: frames of the 120 fps source sequence (every 4th is used)
    "Beauty": 600, "Bosphorus": 600, "HoneyBee": 600, "Jockey": 600, "ReadySetGo": 600,
    "YachtRide": 600, "ShakeNDry": 300,
}
UVG_STEP, UVG_GOP = 4, 10
UVG_SHA256 = {  # the official 8-bit yuv420p 1080p .yuv files
    "Beauty": "ad5c466ce78e100123202441035b58a485d620aeb6c7b2ef84176c4b9da1ff00",
    "Bosphorus": "0cc508da880e0a466faf0618eb28b73b761b2e7a64d3c8bd45baf2e91c037856",
    "HoneyBee": "50e47af98d7828123f898869b7d3a820bbbba635d35d240c5c4c965e8ae712de",
    "Jockey": "4a8b31553559114d1376e10fd15b324e172b659f960fcb009eac8da3007a4c7d",
    "ReadySetGo": "b83f7800f742c1c2c3bd30dd3bfcb660d6559fb1a68c74fb022da97228ec9061",
    "ShakeNDry": "d0a024c975e0c6b671c53272b6265ce04066785c2e5ae0d0f30b061a9e47b925",
    "YachtRide": "a09567ad2cb42f890d1c453b4971126b5a55e74bca5814915c7ae68d627d2337",
}
UVG_YUV_STEM = {"ReadySetGo": "ReadySteadyGo"}   # the file inside the ReadySetGo archive
DAVIS = {  # key: (DAVIS sequence, frames used, GoP length)
    "bmx": ("bmx-trees", 80, 10), "boat": ("boat", 75, 10), "train": ("train", 80, 10),
    "camel": ("camel", 90, 10), "cows": ("cows", 103, 10), "swan": ("blackswan", 50, 6),
}
W720, H720 = 1280, 720

SCENES = {"uvg": list(UVG), "davis": list(DAVIS)}   # --scenes: the names each dataset takes
# verify: set -> (checksum file, directory its GoP paths are relative to; None: --out, scenes)
CHECKSUMS = {
    "uvg_davis": (DATA / "checksums" / "uvg_davis.sha256", None, list(UVG) + list(DAVIS)),
    "bunny": (DATA / "checksums" / "bunny.sha256", BUNNY.parent, ["bunny"]),
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found on PATH")

def resize_720(job: tuple[str, str]) -> None:
    from PIL import Image
    src, dst = job
    try:
        bilinear = Image.Resampling.BILINEAR
    except AttributeError:          # Pillow < 9.1
        bilinear = Image.BILINEAR
    Image.open(src).convert("RGB").resize((W720, H720), bilinear).save(dst)

def run_jobs(fn, jobs: list, workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, jobs, chunksize=4))

def split_gops(frames: list[Path], gop_len: int, scene: str, root_full: Path,
               root_720: Path, keep_partial: bool, workers: int) -> int:
    """Group full-resolution frames into GoP folders under root_full and write their 720p copies."""
    jobs, n_gops = [], 0
    for g, start in enumerate(range(0, len(frames), gop_len), start=1):
        chunk = frames[start:start + gop_len]
        if len(chunk) < gop_len and not keep_partial:
            print(f"  {scene}_{g}: partial GoP ({len(chunk)} frames) skipped")
            continue
        full = root_full / scene / f"{scene}_{g}"
        small = root_720 / scene / f"{scene}_{g}"
        full.mkdir(parents=True, exist_ok=True)
        small.mkdir(parents=True, exist_ok=True)
        for j, f in enumerate(chunk, start=1):
            dst = full / f"f{j:05d}.png"
            shutil.move(str(f), dst)
            jobs.append((str(dst), str(small / f"f{j:05d}.png")))
        n_gops += 1
    run_jobs(resize_720, jobs, workers)
    return n_gops

def extract_7z(archive: Path, dest: Path) -> None:
    if shutil.which("7z"):
        run(["7z", "x", "-y", f"-o{dest}", str(archive)])
        return
    try:
        import py7zr
    except ImportError:
        sys.exit(f"cannot extract {archive.name}: install p7zip (7z) or the py7zr package")
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(dest)

def find_uvg_yuv(src: Path, scene: str, tmp: Path) -> Path:
    stem = UVG_YUV_STEM.get(scene, scene)
    for d in (src, tmp):
        hits = sorted(d.glob(f"{stem}_1920x1080_120fps_420_8bit_YUV*.yuv"))
        if hits:
            return hits[0]
    archive = src / f"{scene}_1920x1080_120fps_420_8bit_YUV_RAW.7z"
    if not archive.is_file():
        sys.exit(f"UVG {scene}: neither the .yuv nor {archive.name} is in {src}")
    print(f"  extracting {archive.name}")
    extract_7z(archive, tmp)
    hits = sorted(tmp.rglob(f"{stem}_1920x1080_120fps_420_8bit_YUV*.yuv"))
    if not hits:
        sys.exit(f"UVG {scene}: no .yuv found in {archive.name}")
    return hits[0]

def checked_uvg_yuv(src: Path, scene: str, tmp: Path) -> Path:
    """The scene's source .yuv (extracted into tmp if only the archive is there). Exits on a wrong
    size and warns on a SHA-256 other than UVG_SHA256."""
    yuv = find_uvg_yuv(src, scene, tmp)
    n_src = UVG[scene]
    size = yuv.stat().st_size
    if size != n_src * 1920 * 1080 * 3 // 2:
        sys.exit(f"UVG {scene}: {yuv.name} has {size} bytes, not {n_src} 1080p yuv420p frames")
    digest = sha256_file(yuv)
    if digest != UVG_SHA256[scene]:
        print(f"  warning: {yuv.name} has sha256 {digest}, not that of the paper's source "
              f"({UVG_SHA256[scene]}); its frames will not match the paper's")
    return yuv

def drop_extracted(yuv: Path, tmp: Path) -> None:
    """Delete a .yuv that this script extracted into its scratch folder (never a source file)."""
    if yuv.parent == tmp or tmp in yuv.parents:
        yuv.unlink()

def yuv_every_nth_to_pngs(yuv: Path, step: int, out: Path) -> list[Path]:
    """Frames 0, step, 2 step, ... of a 1080p yuv420p clip as PNGs (ffmpeg's select filter)."""
    out.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", "1920x1080", "-framerate", "120", "-i", str(yuv),
         "-vf", f"select=not(mod(n\\,{step}))", "-vsync", "0", str(out / "s%05d.png")])
    return sorted(out.glob("s*.png"))

def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

def prep_uvg(args) -> None:
    need("ffmpeg")
    root = args.out / "uvg_proto"
    with tempfile.TemporaryDirectory(dir=args.tmp) as t:
        tmp = Path(t)
        for scene, n_src in UVG.items():
            if args.scenes and scene not in args.scenes:
                continue
            yuv = checked_uvg_yuv(args.src, scene, tmp)
            frames = yuv_every_nth_to_pngs(yuv, UVG_STEP, tmp / f"{scene}_frames")
            want = len(range(0, n_src, UVG_STEP))
            if len(frames) != want:
                sys.exit(f"UVG {scene}: expected {want} protocol frames, decoded {len(frames)}")
            n = split_gops(frames, UVG_GOP, scene, tmp / "full", root / "uvg720", True, args.workers)
            shutil.rmtree(tmp / "full" / scene)
            drop_extracted(yuv, tmp)
            print(f"UVG {scene}: every {UVG_STEP}th of {n_src} frames = {want} frames -> {n} GoPs of up to {UVG_GOP}")

def davis_jpeg_root(src: Path, sequences: list[str], tmp: Path) -> Path:
    """The folder holding DAVIS's JPEGImages/Full-Resolution/<sequence>/ folders. From the zip,
    only the given sequences are extracted, into tmp."""
    if src.is_file() and src.suffix == ".zip":
        names = [f"DAVIS/JPEGImages/Full-Resolution/{seq}/" for seq in sequences]
        with zipfile.ZipFile(src) as z:
            members = [m for m in z.namelist() if any(m.startswith(p) for p in names)]
            z.extractall(tmp, members)
        src = tmp
    root = next((p for p in (src / "DAVIS" / "JPEGImages" / "Full-Resolution",
                             src / "JPEGImages" / "Full-Resolution", src)
                 if all((p / seq).is_dir() for seq in sequences)), None)
    if root is None:
        sys.exit(f"no DAVIS JPEGImages/Full-Resolution/<sequence>/ folders for {', '.join(sequences)} "
                 f"under {src} (the DAVIS 2017 TrainVal Full-Resolution archive holds them all)")
    return root

def prep_davis(args) -> None:
    need("ffmpeg")
    keys = [k for k in DAVIS if not args.scenes or k in args.scenes]
    with tempfile.TemporaryDirectory(dir=args.tmp) as t:
        tmp = Path(t)
        jpeg_root = davis_jpeg_root(args.src, [DAVIS[k][0] for k in keys], tmp)
        for key in keys:
            seq, n_frames, gop = DAVIS[key]
            jpgs = sorted((jpeg_root / seq).glob("*.jpg"))[:n_frames]
            if len(jpgs) != n_frames:
                sys.exit(f"DAVIS {seq}: expected {n_frames} frames, found {len(jpgs)}")
            out = tmp / f"{key}_frames"
            out.mkdir()
            frames = []
            for i, jpg in enumerate(jpgs, start=1):
                png = out / f"f{i:05d}.png"
                run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(jpg), str(png)])
                frames.append(png)
            n = split_gops(frames, gop, key, tmp / "full", args.out / "davis720", True, args.workers)
            shutil.rmtree(tmp / "full" / key)
            print(f"DAVIS {key} ({seq}): {n_frames} frames -> {n} GoPs of up to {gop}")

def gop_pixel_hash(gop: Path) -> tuple[str, int, tuple[int, int]]:
    """(sha256, frames, (W, H)) of a GoP directory: the hash of its frames' RGB pixels (uint8,
    row-major), concatenated in sorted file order. PNG files can differ in their compression
    while holding the same pixels; this hash depends on the pixels only."""
    import hashlib
    import numpy as np
    from PIL import Image
    h, sizes = hashlib.sha256(), set()
    files = sorted(f for f in gop.iterdir() if f.is_file())
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        sizes.add((a.shape[1], a.shape[0]))
        h.update(np.ascontiguousarray(a).tobytes())
    if len(sizes) != 1:
        raise ValueError(f"{gop}: frames of different sizes {sorted(sizes)}")
    return h.hexdigest(), len(files), sizes.pop()

def read_checksums(path: Path) -> list[list[str]]:
    """The lines of a checksum file: [sha256, frames, WxH, GoP directory]."""
    return [ln.split() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

def selects(scene: str, rel: str) -> bool:
    """A --scenes name selects a GoP of a checksum file: a folder of its path, or its GoP name's prefix."""
    return scene in Path(rel).parts or Path(rel).name.startswith(f"{scene}_")

def verify_set(name: str, data_root: Path, scenes: list[str] | None) -> tuple[int, int, int, int]:
    """(GoPs listed, checked, different, not prepared) of one set; prints every difference."""
    path, base, _ = CHECKSUMS[name]
    root = base or data_root
    lines = read_checksums(path)
    if scenes:
        lines = [ln for ln in lines if any(selects(s, ln[3]) for s in scenes)]
    bad = missing = checked = 0
    for digest, nf, size, rel in lines:
        gop = root / rel
        if not gop.is_dir():
            missing += 1
            continue
        checked += 1
        try:
            got, n, (w, h) = gop_pixel_hash(gop)
        except (OSError, ValueError) as e:      # a file that is not an image, frames of different sizes
            bad += 1
            print(f"DIFFERENT {rel}: {e}")
            continue
        ok = got == digest and n == int(nf) and f"{w}x{h}" == size
        bad += not ok
        if not ok:
            print(f"DIFFERENT {rel}: {n} frames {w}x{h} sha256 {got[:16]}..., the paper's: {nf} frames {size} "
                  f"{digest[:16]}...")
    print(f"{name}: verified {checked} of the {len(lines)} GoPs of {path.relative_to(REPO)}"
          + (" (selected scenes)" if scenes else "") + f": {checked - bad} identical, {bad} different"
          + (f", {missing} not prepared" if missing else ""))
    return len(lines), checked, bad, missing

def verify(args) -> None:
    """Check GoP folders against the pixel checksums of the paper's frames. The exit status is 1
    if any listed GoP is different or not prepared."""
    names = list(CHECKSUMS) if "all" in args.set else args.set
    unknown = [n for n in names if n not in CHECKSUMS]
    if unknown:
        sys.exit(f"unknown set {unknown[0]}; the sets are {', '.join(CHECKSUMS)} (or all)")
    if args.scenes:
        rels = [ln[3] for n in names for ln in read_checksums(CHECKSUMS[n][0])]
        bad = [s for s in args.scenes if not any(selects(s, r) for r in rels)]
        if bad:
            valid = list(dict.fromkeys(s for n in names for s in CHECKSUMS[n][2]))
            sys.exit(f"--scenes {' '.join(bad)}: no GoP of {', '.join(names)}; their scenes are {', '.join(valid)}")
    failed = 0
    for name in names:
        listed, checked, bad, missing = verify_set(name, args.out, args.scenes)
        failed += bool(bad or missing or not listed)
    sys.exit(1 if failed else 0)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["uvg", "davis", "verify"])
    ap.add_argument("--src", type=Path, help="downloaded sources (see data/README.md)")
    ap.add_argument("--out", type=Path, default=DATA, help="the data root (default: data/)")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="subset of scenes, named as in the output folders")
    ap.add_argument("--set", nargs="+", default=["uvg_davis"],
                    help=f"verify: the sets to check ({', '.join(CHECKSUMS)}, or all; default uvg_davis)")
    ap.add_argument("--workers", type=int, default=4, help="processes for resizing")
    ap.add_argument("--tmp", type=Path, default=None,
                    help="scratch directory for decoded clips (default: inside --out)")
    args = ap.parse_args()
    if args.scenes and args.dataset in SCENES:
        bad = [s for s in args.scenes if s not in SCENES[args.dataset]]
        if bad:   # e.g. a lower-case UVG name, or a DAVIS sequence named by its full name instead of its key
            seq = {v[0]: k for k, v in DAVIS.items()} if args.dataset == "davis" else {}
            hint = "".join(f" ({s} is {seq[s]} here)" for s in bad if s in seq)
            sys.exit(f"--scenes {' '.join(bad)}: not a scene of {args.dataset}{hint}; its scenes are "
                     f"{', '.join(SCENES[args.dataset])}")
    if args.dataset == "verify":
        return verify(args)
    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp = args.tmp or args.out
    if not args.tmp.is_dir():
        sys.exit(f"--tmp {args.tmp}: not a directory")
    if args.src is None or not args.src.exists():
        sys.exit("--src is required and must exist (see data/README.md)")
    {"uvg": prep_uvg, "davis": prep_davis}[args.dataset](args)


if __name__ == "__main__":
    main()
