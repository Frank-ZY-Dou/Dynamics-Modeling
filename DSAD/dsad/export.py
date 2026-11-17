#!/usr/bin/env python
"""Write the stored form of a DSAD GoP (--appearance sad): 9 numbers per site and the ROM.

Training keeps each site's anisotropy axis as an unnormalized 2-vector v, so
gaussian.pth holds 10 numbers per site. The metric depends only on the
direction of v up to sign, which is stored as one float32 angle

    theta = atan2(v_y, v_x) mod pi        (float64 arithmetic)

and decoded as u = (cos theta, sin theta). The paper scores DSAD on frames
decoded from this form. export_run refuses a checkpoint whose angles would not
be lossless: an axis of norm < 1e-4, or a rebuilt axis more than |sin| = 1e-6
away from v / |v|.

<run>/stored.npz (numpy .npz, no pickled objects):

    sites       [N, 9] float32, one row per site:
                z_x z_y log_tau radius r g b theta log_aniso
                (z is the pre-tanh position: pixel position = (tanh(z) + 1) / 2 * [W, H])
    rom/<name>  the ROM tensors (anchors.*, coef_xyz.*, coef_features.*), float32
    meta        JSON header: the model's structure (N, M, K, anchor top-K and mode,
                residual channels), the GoP (frame count, size, path, pixel checksum)
                and the checkpoint files it was exported from (size and sha256)

Stored parameters per GoP: 9 N + ROM (meta["stored_params"]), with
ROM = 7 M + 9 M K for the paper's anchors with Δxyz and Δfeatures knots. The
site opacity is the constant 1 and is not stored.

usage:
    python dsad/export.py --run checkpoints/<tag>/SAD3_<it>_<N>/<gop> [--out <file>] [--force]
                          [--gop_path <path>]

dsad/decode.py and scripts/score_decode.py export a run themselves when its
stored.npz is missing or was exported from other checkpoint files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from . import _paths, runinfo
except ImportError:
    import _paths
    import runinfo

FORMAT = "dsad-stored"
VERSION = 1
STORED_NAME = "stored.npz"
SITE_COLUMNS = ("z_x", "z_y", "log_tau", "radius", "r", "g", "b", "theta", "log_aniso")
PER_SITE = len(SITE_COLUMNS)        # 9
AXIS_MIN_NORM = 1e-4                # below it the metric's + 1e-8 is not negligible
AXIS_MAX_SIN = 1e-6                 # rebuilt axis vs v / |v|
SITE_KEYS = ("_xyz", "_features_dc", "_log_tau", "_radius", "_aniso_dir", "_log_aniso")
# Candidate-update settings of the paper's configuration. Decoding with exact candidates
# does not use them; they matter only for VPT passes (tools/exact_candidates_check.py).
CAND_KEYS = ("cand_freq", "cand_warmup_passes", "cand_flood_every", "cand_inject_count")
PAPER_CAND = {"cand_freq": 2, "cand_warmup_passes": 16, "cand_flood_every": 1000,
              "cand_inject_count": 4}


class RunError(Exception):
    """A run that cannot be exported or decoded: not an --appearance sad run, or files
    or settings missing."""


def anchor_mode(deform_sd: dict) -> str:
    """The anchor mode of a ROM state dict, from the anchor parameters it holds."""
    keys = set(deform_sd)
    if "anchors.log_aspect" in keys:
        return "apollonius_aniso"
    if "anchors.radius" in keys:
        return "apollonius"
    if "anchors.weight" in keys:
        return "power"
    return "voronoi"


def read_run_info(run: Path) -> dict:
    """Run settings from run_config.json and train.txt.
    `gop` is the GoP directory as given to the trainer (-d)."""
    run = Path(run)
    info = {"gop": None}
    cfg_path = run / "run_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        info.update(anchor_top_k=cfg.get("anchor_top_k"), frames=cfg.get("frames"),
                    height=cfg.get("height"), width=cfg.get("width"),
                    iterations=cfg.get("iterations"), appearance=cfg.get("appearance"),
                    gop=cfg.get("dataset"),
                    cand={k: cfg[k] for k in CAND_KEYS if k in cfg})
    txt = run / "train.txt"
    log = txt.read_text(errors="replace") if txt.is_file() else ""
    head = log[:6000]
    if info.get("anchor_top_k") is None:
        m = re.search(r"anchor top-K: (\d+)", head)
        info["anchor_top_k"] = int(m.group(1)) if m else None
    if info.get("frames") is None:
        m = re.search(r"frames=(\d+)\s+HxW=(\d+)x(\d+)", head)
        if m:
            info.update(frames=int(m.group(1)), height=int(m.group(2)), width=int(m.group(3)))
    if info["gop"] is None or info.get("appearance") is None:
        m = re.search(r"^\[provenance\] argv: (.*)$", head, re.M)
        argv = m.group(1) if m else ""
        g = re.search(r"(?:^| )(?:-d|--dataset) (\S+)", argv)
        if info["gop"] is None and g:
            info["gop"] = g.group(1)
        a = re.findall(r"--appearance (\S+)", argv)
        if info.get("appearance") is None and a:
            info["appearance"] = a[-1]
    done = re.findall(r"^\[fine\] DONE.*?PSNR=([\d.]+)", log, re.M)
    info["recorded_psnr"] = float(done[-1]) if done else None
    info["finished"] = bool(done)
    return info


def ckpt_files(run: Path, iterations: int | None = None) -> tuple[Path, Path]:
    """(gaussian.pth, deform_sad.pth) of a run; RunError if either is missing."""
    run = Path(run)
    if not (run / "gaussian.pth").is_file():
        raise RunError(f"{run}: no gaussian.pth (the run has not saved its checkpoint)")
    it = iterations
    if it is None:
        m = re.match(r"SAD3_(\d+)_\d+$", run.parent.name)
        it = int(m.group(1)) if m else None
    dpath = run / "deform_sad" / f"iteration_{it}" / "deform_sad.pth"
    if not dpath.is_file():
        hits = sorted((run / "deform_sad").glob("iteration_*/deform_sad.pth"))
        if len(hits) != 1:
            raise RunError(f"{run}: cannot find the ROM weights (deform_sad/iteration_*/)")
        dpath = hits[0]
    return run / "gaussian.pth", dpath


def ckpt_source(run: Path, iterations: int | None = None) -> dict:
    """{path relative to the run: {"size", "sha256"}} of the run's checkpoint files: read
    from finished.json if the run has one, else computed from the files. A stored form or
    a decode score whose `source` equals it belongs to the run's current checkpoint."""
    marker = runinfo.read_finished(run)
    if marker is not None:
        return marker["files"]
    return runinfo.files_entry(run, ckpt_files(run, iterations))


def find_gop(gop_path: str | None, data_root: Path | None = None) -> Path | None:
    """The run's GoP directory on this machine, or None if it is not found.

    Without `data_root`: the recorded path if it is a directory (a relative path is
    tried from the working directory, then from the DSAD directory). With
    `data_root`: the longest trailing part of the recorded path that is a directory
    under it, e.g. <data_root>/uvg_proto/uvg720/Jockey/Jockey_1 for a run trained on
    data/uvg_proto/uvg720/Jockey/Jockey_1. The frame count and size are not checked."""
    if not gop_path:
        return None
    p = Path(gop_path)
    if data_root is not None:
        root = Path(data_root)
        parts = p.parts[1:] if p.is_absolute() else p.parts
        for k in range(len(parts)):
            tail = parts[k:]
            if ".." in tail or "." in tail:
                continue
            if root.joinpath(*tail).is_dir():
                return root.joinpath(*tail)
        return None
    for cand in ((p,) if p.is_absolute() else (p, _paths.REPO / p)):
        if cand.is_dir():
            return cand
    return None


def axis_angle(v: torch.Tensor) -> torch.Tensor:
    """[N] float32 angles of the [N, 2] axis vectors v: atan2 mod pi in float64, in [0, pi),
    rounded to float32. The rounding can give float32(pi), which is harmless: the metric
    does not depend on the sign of the axis."""
    d = v.double()
    return torch.remainder(torch.atan2(d[:, 1], d[:, 0]), math.pi).float()


def axis_from_angle(theta: torch.Tensor) -> torch.Tensor:
    """[N, 2] float32 unit axes (cos theta, sin theta) of [N] float32 angles (float64 arithmetic)."""
    th = theta.double()
    return torch.stack([torch.cos(th), torch.sin(th)], 1).float()


def axis_check(v: torch.Tensor, theta: torch.Tensor) -> float:
    """Largest |sin| of the angle between a rebuilt axis and v / |v| (sign-free, as the metric)."""
    d = v.double()
    r = axis_from_angle(theta).double()
    u = d / d.norm(dim=1, keepdim=True)
    return float((r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]).abs().max()) if len(d) else 0.0


def gop_pixel_hash(gop: Path) -> str:
    """sha256 of a GoP directory's frames as RGB uint8 pixels, concatenated in sorted file order
    (the hash of scripts/prepare_data.py and data/checksums/)."""
    from PIL import Image
    h = hashlib.sha256()
    for f in sorted(x for x in Path(gop).iterdir() if x.is_file()):
        h.update(np.ascontiguousarray(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)).tobytes())
    return h.hexdigest()


def stored_path(run: Path) -> Path:
    return Path(run) / STORED_NAME


def is_current(stored: Path, run: Path) -> bool:
    """stored.npz exists, reads, and was exported from the run's current checkpoint files."""
    stored = Path(stored)
    if not stored.is_file():
        return False
    try:
        meta = read_meta(stored)
        return meta.get("source") == ckpt_source(run, meta.get("iterations"))
    except Exception:   # unreadable (empty, truncated) or no checkpoint: export again
        return False


def _read_checkpoint(run: Path, dev) -> tuple[dict, int, dict, dict, dict]:
    """(info, iterations, sites, ROM, source) of a DSAD run: its settings
    (read_run_info), the fine-stage iteration count, the site and ROM state dicts on
    `dev`, and the checkpoint files as finished.json names them, computed from the
    files read. Raises RunError if the run's finished.json names other files."""
    run = Path(run)
    info = read_run_info(run)
    if info.get("appearance") == "gaussian":
        raise RunError(f"{run}: a DSAD-GS (--appearance gaussian) run; only SAD sites have a stored form here")
    gpath, dpath = ckpt_files(run, info.get("iterations"))
    it = int(dpath.parent.name.split("_")[1])
    source = runinfo.files_entry(run, (gpath, dpath))
    marker = runinfo.read_finished(run)
    if marker is not None and marker["files"] != source:
        raise RunError(f"{run}: the checkpoint files are not the ones its finished.json names (they changed "
                       "after the run finished); train the run again")
    state = torch.load(gpath, map_location=dev)
    if "_log_tau" not in state:
        raise RunError(f"{gpath} holds no SAD sites (not an --appearance sad run)")
    missing = [k for k in SITE_KEYS if k not in state]
    if missing:
        raise RunError(f"{gpath}: site tensors missing: {missing}")
    dsd = torch.load(dpath, map_location=dev)
    if info.get("anchor_top_k") is None:
        raise RunError(f"{run}: anchor top-K unknown (no run_config.json or train.txt)")
    if None in (info.get("frames"), info.get("height"), info.get("width")):
        raise RunError(f"{run}: GoP shape unknown (no run_config.json or train.txt)")
    return info, it, state, dsd, source


def _structure(info: dict, it: int, dsd: dict, N: int, per_site: int) -> dict:
    """Header fields that describe the model: N, M, K, anchor top-K and mode, the Δ stacks,
    the GoP's shape, the counts, the candidate settings and the PSNR the run logged."""
    rom_n = int(sum(t.numel() for t in dsd.values()))
    vxyz = dsd["coef_xyz.v"]
    return {"N": N, "M": int(vxyz.shape[0]), "K": int(vxyz.shape[1]), "anchor_top_k": int(info["anchor_top_k"]),
            "anchor_mode": anchor_mode(dsd),
            "features": any(k.startswith("coef_features.") for k in dsd),
            "opacity": any(k.startswith("coef_opacity.") for k in dsd),
            "frames": int(info["frames"]), "height": int(info["height"]), "width": int(info["width"]),
            "iterations": it, "per_site": per_site, "rom_params": rom_n, "stored_params": per_site * N + rom_n,
            "recorded_psnr": info.get("recorded_psnr"), "cand": {**PAPER_CAND, **(info.get("cand") or {})}}


@torch.no_grad()
def export_run(run: Path, out: Path | None = None, device: str | torch.device | None = None,
               quiet: bool = False, gop_path: str | None = None) -> Path:
    """Write the stored form of a finished run to `out` (default <run>/stored.npz).

    The header records the GoP as `gop_path` if given (e.g. its path relative to the data
    root), else as the trainer was given it. The pixel checksum is that of the frames the
    run was trained on, when they are found."""
    run = Path(run)
    # The angle arithmetic runs on the GPU when there is one; meta["axis_device"] records it.
    dev = torch.device(device) if device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    info, it, state, dsd, source = _read_checkpoint(run, dev)
    if "_opacity" in state and not bool((state["_opacity"] == 1).all()):
        raise RunError(f"{run}: the site opacity is not the constant 1; it would have to be stored")

    v = state["_aniso_dir"]
    n_min = float(v.double().norm(dim=1).min()) if len(v) else 1.0
    if n_min < AXIS_MIN_NORM:
        raise RunError(f"{run}: a site axis has norm {n_min:.3g} < {AXIS_MIN_NORM}: the angle would not be lossless")
    theta = axis_angle(v)
    max_sin = axis_check(v, theta)
    if not max_sin <= AXIS_MAX_SIN:
        raise RunError(f"{run}: a rebuilt axis is |sin| = {max_sin:.3g} away from the trained one")

    N = int(state["_xyz"].shape[0])
    cols = [state["_xyz"].float(), state["_log_tau"].float().reshape(N, 1),
            state["_radius"].float().reshape(N, 1), state["_features_dc"].float(),
            theta.reshape(N, 1), state["_log_aniso"].float().reshape(N, 1)]
    sites = torch.cat(cols, 1).cpu().numpy()
    assert sites.shape == (N, PER_SITE) and sites.dtype == np.float32
    rom = {k: t.detach().float().cpu().numpy() for k, t in dsd.items()}
    gop = find_gop(info["gop"])
    gop_sha = None
    if gop is not None:   # the frames the run was trained on, if they are where it recorded them
        files = [x for x in gop.iterdir() if x.is_file()]
        if len(files) == int(info["frames"]):
            gop_sha = gop_pixel_hash(gop)
    meta = {
        "format": FORMAT, "version": VERSION, "site_columns": list(SITE_COLUMNS),
        **_structure(info, it, dsd, N, PER_SITE),
        "axis_max_sin": max_sin, "axis_device": str(dev),
        "gop": gop_path if gop_path is not None else info["gop"], "gop_sha256": gop_sha,
        "source": source,
    }
    out = Path(out) if out is not None else stored_path(run)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp.npz")   # one per process: concurrent exports
    np.savez(tmp, sites=sites, meta=np.array(json.dumps(meta, sort_keys=True)),
             **{f"rom/{k}": a for k, a in rom.items()})
    tmp.replace(out)
    if not quiet:
        print(f"{run}: wrote {out.name}: {N:,} sites x {PER_SITE} + ROM {meta['rom_params']:,} = "
              f"{meta['stored_params']:,} numbers (axis check |sin| <= {max_sin:.2e})")
    return out


def ensure_stored(run: Path, device=None, quiet: bool = False) -> Path:
    """<run>/stored.npz, exported first if it is missing or not from the current checkpoint."""
    p = stored_path(run)
    if not is_current(p, run):
        export_run(run, p, device=device, quiet=quiet)
    return p


def read_meta(path: Path) -> dict:
    """The JSON header of a stored.npz; ValueError if the file has another format."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError(f"{path}: not a {FORMAT} v{VERSION} file")
    return meta


@torch.no_grad()
def build_model(state: dict, rom: dict, meta: dict, device, cand: dict | None = None,
                size: tuple[int, int] | None = None, rescale_radius: bool = True):
    """(SADAppearance_V3, DeformModelSAD) from site and ROM state dicts, for decoding.

    size: decode onto another (W, H) grid. The site positions (tanh of the stored
    coordinates, scaled to the frame), log_tau and the anisotropy are resolution-free:
    the metric measures distances in units of max(W, H). Only the additive radius is in
    pixels of the fitting grid, so it is scaled by max(W', H') / max(W, H) unless
    rescale_radius is False."""
    from deformation_core import DeformModelSAD
    from sad_appearance_v3 import SADAppearance_V3
    W, H = (int(meta["width"]), int(meta["height"])) if size is None else (int(size[0]), int(size[1]))
    fit = max(int(meta["width"]), int(meta["height"]))
    if rescale_radius and max(W, H) != fit:
        state = dict(state, _radius=state["_radius"] * (max(W, H) / fit))
    c = {**PAPER_CAND, **(meta.get("cand") or {}), **(cand or {})}
    model = SADAppearance_V3(num_points=meta["N"], H=H, W=W, device=device,
                             num_frames=meta["frames"], **c).to(device)
    model.load_state_dict(state)
    model.eval()
    deform = DeformModelSAD(num_anchors=meta["M"], num_knots=meta["K"], anchor_top_k=meta["anchor_top_k"],
                            anchor_mode=meta["anchor_mode"], enable_features=meta["features"],
                            enable_opacity=meta["opacity"], max_steps=meta["iterations"], device=str(device))
    deform.deform.load_state_dict(rom)
    return model, deform


@torch.no_grad()
def load_stored(path: Path, device, cand: dict | None = None, size: tuple[int, int] | None = None,
                rescale_radius: bool = True):
    """(model, deform, meta) rebuilt from a stored.npz alone: SADAppearance_V3 with the sites
    (axes rebuilt from their angles) and DeformModelSAD with the ROM; `size` as in build_model."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != FORMAT or meta.get("version") != VERSION:
            raise ValueError(f"{path}: not a {FORMAT} v{VERSION} file")
        sites = torch.from_numpy(z["sites"]).to(device)
        rom = {k[len("rom/"):]: torch.from_numpy(z[k]).to(device) for k in z.files if k.startswith("rom/")}
    N = meta["N"]
    if tuple(sites.shape) != (N, PER_SITE) or sites.dtype != torch.float32:
        raise ValueError(f"{path}: sites are {tuple(sites.shape)} {sites.dtype}, expected ({N}, {PER_SITE}) float32")
    state = {"_xyz": sites[:, 0:2].contiguous(), "_log_tau": sites[:, 2].contiguous(),
             "_radius": sites[:, 3].contiguous(), "_features_dc": sites[:, 4:7].contiguous(),
             "_aniso_dir": axis_from_angle(sites[:, 7]), "_log_aniso": sites[:, 8].contiguous(),
             "_opacity": torch.ones(N, 1, device=device)}
    model, deform = build_model(state, rom, meta, device, cand, size, rescale_radius)
    return model, deform, meta


@torch.no_grad()
def load_checkpoint(run: Path, device, cand: dict | None = None, size: tuple[int, int] | None = None,
                    rescale_radius: bool = True):
    """(model, deform, meta) from a run's training checkpoint, with every site's axis as the
    trained 2-vector (10 numbers per site) instead of the stored angle, for comparing the
    two decodes. `meta` has the stored form's structure fields, with per_site 10 and
    stored_params counting the checkpoint's numbers; `size` as in build_model."""
    run = Path(run)
    info, it, state, dsd, source = _read_checkpoint(run, device)
    N = int(state["_xyz"].shape[0])
    meta = {**_structure(info, it, dsd, N, 10), "axis": "vector", "gop": info["gop"], "gop_sha256": None,
            "source": source}
    model, deform = build_model(state, dsd, meta, device, cand, size, rescale_radius)
    return model, deform, meta


def stored_angles(path: Path, device) -> torch.Tensor:
    """The [N] float32 stored angles of a stored.npz (for decoders that time the axis rebuild)."""
    with np.load(path, allow_pickle=False) as z:
        return torch.from_numpy(z["sites"][:, SITE_COLUMNS.index("theta")].copy()).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True,
                    help="run directories (checkpoints/<tag>/SAD3_<it>_<N>/<gop>)")
    ap.add_argument("--out", type=Path, default=None, help="output file (one run only; default <run>/stored.npz)")
    ap.add_argument("--force", action="store_true", help="export again even if stored.npz is current")
    ap.add_argument("--gop_path", default=None,
                    help="GoP path to record in the stored form (one run only; default: as given to the "
                         "trainer)")
    args = ap.parse_args()
    if (args.out is not None or args.gop_path is not None) and len(args.run) != 1:
        ap.error("--out and --gop_path take one --run")
    _paths.setup_imports()
    bad = 0
    for run in args.run:
        try:
            if args.out is not None or args.force or args.gop_path is not None \
                    or not is_current(stored_path(run), run):
                export_run(run, args.out, gop_path=args.gop_path)
            else:
                print(f"{run}: {STORED_NAME} is current")
        except RunError as e:
            print(f"ERROR {e}", file=sys.stderr)
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
