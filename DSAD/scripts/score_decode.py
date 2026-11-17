#!/usr/bin/env python
"""Score finished DSAD runs by decoding them from their stored form.

    CUDA_VISIBLE_DEVICES=0 python scripts/score_decode.py --runs 'checkpoints/dsad_*'

Every finished run (one with a finished.json) is exported to <run>/stored.npz and decoded from it
with the eager decoder of dsad/decode.py. PSNR and MS-SSIM against the GoP's frames go to
<run>/decode.json, which scripts/summarize.py reads. These are the scores the paper reports.

A run that already has the decode.json of its checkpoint is skipped unless --force. The exit
status is 1 if a run could not be scored.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dsad"))
import _paths  # noqa: E402

_paths.setup()
import export  # noqa: E402
import runinfo  # noqa: E402


def load_decoder():
    """Import dsad/decode.py after the arguments are parsed, so that --help works before
    setup.sh has built SAD's CUDA extension."""
    try:
        import decode
        decode.require_sad()
    except ImportError as e:
        raise SystemExit(f"cannot load the decoder: {e}\n(build the CUDA extensions first: bash setup.sh)") from None
    return decode


def logged_runs(pattern: str):
    """DSAD run directories (holding a train.txt) under the tag directories."""
    for tag_dir in sorted(glob.glob(pattern)):
        for txt in sorted(Path(tag_dir).glob("SAD3_*/*/train.txt")):
            yield txt.parent


def has_score(run: Path) -> bool:
    """decode.json exists and was computed from the run's current checkpoint."""
    try:
        d = json.loads((run / "decode.json").read_text())
        return "ms_ssim" in d and d.get("source") == export.ckpt_source(run)
    except (OSError, ValueError, export.RunError):
        return False


def score_run(decode, run: Path, dev, data_root: Path | None) -> dict | None:
    """Decoded PSNR and MS-SSIM of one run from its stored form (None if its GoP is not found).
    decode is the module that load_decoder() returns."""
    stored = export.ensure_stored(run, device=dev, quiet=True)
    model, deform, info, frames = decode.load(stored, dev, data_root=data_root)
    if frames is None:
        recorded = info.get("gop") or "(none recorded)"
        print(f"SKIP {run}: its GoP was not found ({recorded}"
              + (f" under {data_root}" if data_root else "") + "); pass --data_root", flush=True)
        return None
    res = decode.scores(decode.decode_gop(model, deform, info["T"]), frames)
    meta = export.read_meta(stored)
    res.update(method="exact top-8, eager decoder", direction="angle", stored_params=meta["stored_params"],
               per_site=meta["per_site"], axis_max_sin=meta["axis_max_sin"], source=meta["source"],
               recorded_psnr=export.read_run_info(run).get("recorded_psnr"),
               config={k: info[k] for k in ("N", "M", "K", "anchor_top_k", "mode", "features", "opacity",
                                            "T", "H", "W", "gop")})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, nargs="+",
                    help="glob(s) of run tag directories, e.g. 'checkpoints/dsad_*'")
    ap.add_argument("--force", action="store_true",
                    help="score again runs whose decode.json matches their checkpoint")
    ap.add_argument("--data_root", type=Path, default=None,
                    help="look each run's recorded GoP path up under this directory")
    args = ap.parse_args()
    decode = load_decoder()
    dev = torch.device("cuda:0")
    n_done = n_kept = n_gs = n_skip = n_seen = n_unfinished = 0
    unmarked, failed = [], []
    for pattern in args.runs:
        for run in logged_runs(pattern):
            n_seen += 1
            if not runinfo.check_finished(run):
                n_unfinished += 1
                if export.read_run_info(run)["finished"]:
                    unmarked.append(str(run))     # DONE logged, but no finished.json
                continue
            if has_score(run) and not args.force:
                n_kept += 1
                continue
            if export.read_run_info(run).get("appearance") == "gaussian":
                n_gs += 1
                continue
            try:
                res = score_run(decode, run, dev, args.data_root)
            except ValueError as e:           # the GoP found does not match the run
                print(f"SKIP {e}", flush=True)
                res = None
            except Exception as e:            # one bad run must not end the pass
                msg = str(e).strip().splitlines()
                print(f"ERROR {run}: {type(e).__name__}: {msg[0] if msg else ''}", flush=True)
                failed.append(str(run))
                continue
            finally:
                torch.cuda.empty_cache()
            if res is None:
                n_skip += 1
                continue
            tmp = run / "decode.json.tmp"
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(run / "decode.json")
            n_done += 1
            print(f"{run}: decoded PSNR {res['psnr']:.4f}  MS-SSIM {res['ms_ssim']:.6f}  "
                  f"({res['stored_params']:,} stored parameters; train.txt: {res['recorded_psnr']})", flush=True)
    if not n_seen:
        sys.exit(f"no run folder matches {' '.join(args.runs)}")
    print(f"scored {n_done} runs; {n_kept} already scored"
          + (f"; {n_unfinished} unfinished" if n_unfinished else "")
          + (f"; {n_gs} DSAD-GS runs need no decoding" if n_gs else "")
          + (f"; {n_skip} skipped (GoP not found or not matching)" if n_skip else "")
          + (f"; {len(failed)} failed" if failed else ""))
    if unmarked:
        print(f"{len(unmarked)} runs logged '[fine] DONE' but have no finished.json (the checkpoint was not "
              "saved; fit them again): " + ", ".join(unmarked[:10])
              + (" ..." if len(unmarked) > 10 else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
