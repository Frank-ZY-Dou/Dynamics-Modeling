#!/usr/bin/env python
"""Write the fits of a benchmark as shell lines, one GoP per line.

    python scripts/make_jobs.py > jobs/main.txt                 # UVG and DAVIS, both budgets (Tab. 2)
    python scripts/make_jobs.py bunny > jobs/bunny.txt
    python scripts/make_jobs.py uvgS --scenes Beauty --gops 1 2
    python scripts/make_jobs.py --counts                        # sites, anchors and parameters per GoP
    bash scripts/run_jobs.sh jobs/main.txt 0 1 2 3              # one worker per GPU

The benchmarks, their GoPs and the site and anchor counts are in configs/benchmarks.yaml. A line
skips its fit when the run has finished (finished.json), so a job file can be run again after an
interruption; the files of an unfinished attempt are moved to <run>/stale/. A fit writes to
checkpoints/<tag>/SAD3_<iterations>_<N>/<gop>/ and logs to logs/<tag>.log.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/dsad.yaml"
MAIN = ("uvgS", "uvgL", "davisS", "davisL")
STALE = "train.txt stored.npz decode.json"


def load() -> tuple[dict, dict]:
    with open(REPO / "configs" / "benchmarks.yaml") as f:
        benchmarks = yaml.safe_load(f)
    with open(REPO / CONFIG) as f:
        config = yaml.safe_load(f)
    return benchmarks, config


def runs(benchmarks: dict, config: dict, names, scenes=None, gops=None, data_root: str = "data") -> list[dict]:
    """One dict per GoP: tag, GoP folder, N, M, fine iterations, stored parameters, trainer options."""
    out = []
    for name in names:
        b = benchmarks[name]
        options = shlex.split(b.get("options", ""))
        iterations = int(options[options.index("--iterations") + 1]) if "--iterations" in options \
            else int(config["iterations"])
        for scene, s in b["scenes"].items():
            if scenes and scene not in scenes:
                continue
            n, m = int(s["num_points"]), s.get("num_anchors")
            anchors = int(m) if m is not None else int(config["num_anchors"])
            for g in range(1, int(s["gops"]) + 1):
                if gops and g not in gops:
                    continue
                out.append(dict(
                    benchmark=name, scene=scene, g=g, n=n, m=m, anchors=anchors, iterations=iterations,
                    options=options,
                    tag=b["tag"].format(scene=scene, g=g),
                    gop=b["gop"].format(data=data_root, scene=scene, g=g),
                    stored=9 * n + 7 * anchors + 9 * anchors * int(config["num_knots"])))
    return out


def command(r: dict) -> str:
    """The trainer call of a run, from the DSAD directory, with $PY the interpreter."""
    args = [f"--config {CONFIG}", f"-d {shlex.quote(r['gop'])}", f"--data_name {r['tag']}",
            f"--num_points {r['n']}"]
    if r["m"] is not None:
        args.append(f"--num_anchors {r['m']}")
    return "$PY -u dsad/train_sad_video.py " + " ".join(args + [shlex.quote(x) for x in r["options"]])


def run_dir(r: dict, root: str = "checkpoints") -> str:
    return f"{root}/{r['tag']}/SAD3_{r['iterations']}_{r['n']}/{Path(r['gop']).name}"


def job_line(r: dict) -> str:
    aside = (f's="$d/stale/$(date -u +%Y%m%dT%H%M%SZ)"; for f in {STALE}; do '
             'if [ -e "$d/$f" ]; then mkdir -p "$s" && mv "$d/$f" "$s/"; fi; done')
    return (f'd={shlex.quote(run_dir(r))}; [ -f "$d"/finished.json ] || '
            f"{{ {aside}; {command(r)} > logs/{r['tag']}.log 2>&1; }}")


def print_counts(rs: list[dict]) -> None:
    print(f"{'benchmark':10s} {'scene':11s} {'sites N':>8s} {'anchors M':>9s} {'per GoP':>9s} {'GoPs':>5s} {'per video':>10s}")
    seen = {}
    for r in rs:
        seen.setdefault((r["benchmark"], r["scene"]), []).append(r)
    for (name, scene), group in seen.items():
        r = group[0]
        print(f"{name:10s} {scene:11s} {r['n']:8,d} {r['anchors']:9d} {r['stored']:9,d} {len(group):5d} "
              f"{sum(x['stored'] for x in group):10,d}")


def main() -> None:
    benchmarks, config = load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmarks", nargs="*", default=list(MAIN),
                    help=f"any of {', '.join(benchmarks)} (default: {' '.join(MAIN)})")
    ap.add_argument("--scenes", nargs="+", default=None, help="only these scenes")
    ap.add_argument("--gops", nargs="+", type=int, default=None, help="only these GoP indices (from 1)")
    ap.add_argument("--data_root", default="data",
                    help="prepared frames of UVG and DAVIS, absolute or relative to the DSAD directory (default: data)")
    ap.add_argument("--counts", action="store_true", help="print the parameter counts instead of job lines")
    args = ap.parse_args()
    unknown = [b for b in args.benchmarks if b not in benchmarks]
    if unknown:
        sys.exit(f"unknown benchmark {unknown[0]}; choose from {', '.join(benchmarks)}")
    known = {s for b in args.benchmarks for s in benchmarks[b]["scenes"]}
    unknown = [s for s in args.scenes or [] if s not in known]
    if unknown:
        sys.exit(f"unknown scene {unknown[0]}; choose from {', '.join(sorted(known))}")
    rs = runs(benchmarks, config, args.benchmarks, args.scenes, args.gops, args.data_root)
    if not rs:
        sys.exit("no GoP selected")
    if args.counts:
        return print_counts(rs)
    missing = sorted({r["gop"] for r in rs if not (REPO / r["gop"]).is_dir()})
    if missing:
        print(f"warning: {len(missing)} GoP folders are missing (first: {missing[0]}); see data/README.md",
              file=sys.stderr)
    print(f"# {len(rs)} fits: {' '.join(args.benchmarks)}")
    for r in rs:
        print(job_line(r))


if __name__ == "__main__":
    main()
