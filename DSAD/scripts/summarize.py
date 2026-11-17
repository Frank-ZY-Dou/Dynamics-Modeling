#!/usr/bin/env python
"""PSNR and MS-SSIM of a benchmark from the decoded scores of its GoPs.

    python scripts/score_decode.py --runs 'checkpoints/dsad_tab2_*'       # first: decode your fits
    python scripts/summarize.py                                           # UVG and DAVIS, both budgets
    python scripts/summarize.py bunny bunny3m
    python scripts/summarize.py --records records                         # from the released logs

Scores are read from the decode.json that scripts/score_decode.py writes into a run folder
(--runs, default checkpoints) or from the released logs (--records). For every scene the script
prints the frame average over the scene's GoPs. A benchmark's value is the mean over its videos
(Tab. 2), and the mean and standard deviation over GoPs are printed beside it (the Bunny tables).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_jobs  # noqa: E402


def score_file(run: dict, root: Path, records: Path | None) -> Path | None:
    """The decode.json of a run: in the released logs (<records>/<study>/<tag>/) or in its run folder."""
    if records is not None:
        hits = sorted(records.glob(f"*/{run['tag']}/decode.json"))
        return hits[0] if hits else None
    f = Path(make_jobs.run_dir(run, str(root))) / "decode.json"
    return f if f.is_file() else None


def summarize(name: str, runs: list[dict], root: Path, records: Path | None) -> bool:
    """Print one benchmark; False if a GoP has no score."""
    scores: dict[str, list[dict]] = {}
    for r in runs:
        f = score_file(r, root, records)
        scores.setdefault(r["scene"], [])
        if f is not None:
            scores[r["scene"]].append(json.loads(f.read_text()))
    total = {s: sum(r["scene"] == s for r in runs) for s in scores}
    print(f"\n{name}")
    print(f"  {'scene':12s} {'GoPs':>5s} {'frames':>6s} {'PSNR':>7s} {'MS-SSIM':>8s}")
    videos = []
    for scene, ds in scores.items():
        if len(ds) < total[scene]:
            print(f"  {scene:12s} {len(ds):2d}/{total[scene]:<2d}  incomplete")
            continue
        psnr = statistics.fmean(x for d in ds for x in d["per_frame"])
        ms = statistics.fmean(x for d in ds for x in d["per_frame_ms_ssim"])
        videos.append((psnr, ms))
        print(f"  {scene:12s} {len(ds):5d} {sum(len(d['per_frame']) for d in ds):6d} {psnr:7.2f} {ms:8.4f}")
    gops = [d for ds in scores.values() for d in ds]
    if len(gops) < len(runs):
        print(f"  {len(runs) - len(gops)} of {len(runs)} GoPs have no score; no mean is printed")
        return False
    sd = (statistics.stdev(d["psnr"] for d in gops), statistics.stdev(d["ms_ssim"] for d in gops)) \
        if len(gops) > 1 else (0.0, 0.0)
    print(f"  {'mean over videos':25s} {statistics.fmean(v[0] for v in videos):7.2f} "
          f"{statistics.fmean(v[1] for v in videos):8.4f}")
    print(f"  {'mean over GoPs':25s} {statistics.fmean(d['psnr'] for d in gops):7.2f} "
          f"{statistics.fmean(d['ms_ssim'] for d in gops):8.4f}   (sd {sd[0]:.2f} / {sd[1]:.4f}, {len(gops)} GoPs)")
    return True


def main() -> None:
    benchmarks, config = make_jobs.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmarks", nargs="*", default=list(make_jobs.MAIN),
                    help=f"any of {', '.join(benchmarks)} (default: {' '.join(make_jobs.MAIN)})")
    ap.add_argument("--runs", type=Path, default=Path("checkpoints"), help="root of the run folders")
    ap.add_argument("--records", type=Path, default=None, help="read the released logs unpacked here instead")
    args = ap.parse_args()
    unknown = [b for b in args.benchmarks if b not in benchmarks]
    if unknown:
        sys.exit(f"unknown benchmark {unknown[0]}; choose from {', '.join(benchmarks)}")
    if args.records is not None and not args.records.is_dir():
        sys.exit(f"{args.records}: not a directory")
    complete = [summarize(b, make_jobs.runs(benchmarks, config, [b]), args.runs, args.records)
                for b in args.benchmarks]
    sys.exit(0 if all(complete) else 1)


if __name__ == "__main__":
    main()
