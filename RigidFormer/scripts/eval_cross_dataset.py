"""
Evaluation runner.

Loads a base training config, points data.dataset_path, data.split_file and (when the base config uses one)
data.object_scale_file at each target dataset, runs src/train.py --inference and collects the per-target eval.txt
files into one output file. --position_estimator overrides the base config's object-position estimator; it changes no
model setting.

Usage:
    python scripts/eval_cross_dataset.py \
        --base_config configs/movi_b.yaml \
        --checkpoint checkpoints/rigidformer_movib.pth \
        --output_file eval_results.txt \
        --targets B,A,S \
        --gpus 0
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import math
import subprocess
import sys
from pathlib import Path

import yaml


# Per-target data: dataset folder, scene lists, object sizes, scene cameras (partial point clouds)
TARGET_TABLE = {
    "B": {"name": "MoviB", "dataset_path": "./datasets/MOVi-B", "split_file": "dataset_split_MoviB_1200.json",
          "scales": "data_generation/object_scales_MoviB.json", "cameras": "data_generation/scene_cameras_MoviB.json"},
    "A": {"name": "MoviA", "dataset_path": "./datasets/MOVi-A", "split_file": "dataset_split_MoviA_1200.json",
          "scales": "data_generation/object_scales_MoviA.json", "cameras": "data_generation/scene_cameras_MoviA.json"},
    "S": {"name": "MoviSphere", "dataset_path": "./datasets/MOVi-Sphere", "split_file": "dataset_split_MoviSphere_1200.json",
          "scales": "data_generation/object_scales_MoviSphere.json",
          "cameras": "data_generation/scene_cameras_MoviSphere.json"},
}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_target_config(base_cfg, target_key, run_tag, render=0, floor="light"):
    """Inference config for one target dataset. render > 0: after the metrics, videos of the model's predictions (no
    ground truth) on the first `render` test scenes at every step size, with the fast renderer
    (src/utils/rollout_render.py) on a light or dark floor."""
    cfg = copy.deepcopy(base_cfg)
    t = TARGET_TABLE[target_key]
    cfg["data"]["dataset_path"] = t["dataset_path"]
    cfg["data"]["split_file"] = t["split_file"]
    if cfg["data"].get("object_scale_file"):
        # the object sizes are per dataset (scene names repeat across MOVi-A/B/Sphere)
        cfg["data"]["object_scale_file"] = t["scales"]
    if cfg["data"].get("partial_view", {}).get("enable"):
        # so are the scene cameras that place the depth camera
        cfg["data"]["partial_view"]["camera_file"] = t["cameras"]
    cfg["experiment_tag"] = run_tag
    cfg.setdefault("inference", {})
    cfg["inference"]["mode"] = True
    cfg["inference"]["generate_videos"] = render > 0
    cfg["inference"]["compute_metrics"] = True
    # the evaluator computes the rollout metrics only with training.metrics_test.enable, which a training config may
    # switch off (no rollouts on the test scenes during training)
    cfg.setdefault("training", {}).setdefault("metrics_test", {})["enable"] = True
    ck = cfg.setdefault("training", {}).setdefault("checkpoint", {})
    ck["save_video"] = render > 0
    if render > 0:
        ck.update(video_renderer="filament", video_floor=floor, if_overlay=False, video_every_n_epochs=1,
                  num_videos_per_split={"train": 0, "val": 0, "test": int(render)}, video_fps=60)
    # Disable validation entirely (we only need test split)
    cfg.setdefault("training", {})
    cfg["training"]["enable_validation"] = False
    return cfg


def parse_eval_txt(eval_txt_path):
    """Parse evaluator's eval.txt into structured dict.

    Returns: list of section dicts with keys:
      step_size: int
      pos_rmse: dict{frame:int -> rmse:float (m)}
      rot_err: dict{frame:int -> deg:float}
      rot_mae: dict{frame:int -> deg:float}
      not_evaluated: sorted list of frames reported as "/" (off the step-size grid or at a warm-up frame)
    """
    sections = []
    cur = None
    mode = None  # "pos" | "rot" | None
    with open(eval_txt_path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if s.startswith("Epoch ") and "step_size=" in s:
                if cur is not None:
                    sections.append(cur)
                ss = int(s.split("step_size=")[1].split("|")[0].strip())
                cur = {"step_size": ss, "pos_rmse": {}, "rot_err": {}, "rot_mae": {}, "not_evaluated": []}
                mode = None
                continue
            if "[Position RMSE]" in s:
                mode = "pos"
                continue
            if "[Rotation Error]" in s:
                mode = "rot"
                continue
            if "[Rotation MAE]" in s:
                mode = "mae"
                continue
            if s.startswith("mapping:") and cur is not None:
                cur["mapping"] = s[len("mapping:"):].strip()
                continue
            if s.startswith("protocol:") and cur is not None:
                cur["protocol"] = s[len("protocol:"):].strip()
                continue
            if cur is None or mode is None:
                continue
            # Lines like:  "  frame  50: 0.029552 m"   /  "  frame  50: 12.3456°"
            if s.startswith("frame "):
                try:
                    parts = s.split(":", 1)
                    frame = int(parts[0].split()[1])
                    val = parts[1].strip()
                    if val == "/":
                        if frame not in cur["not_evaluated"]:
                            cur["not_evaluated"].append(frame)
                        continue
                    val = val.replace("m", "").replace("°", "").strip()
                    val = float(val)
                    if mode == "pos":
                        cur["pos_rmse"][frame] = val
                    elif mode == "rot":
                        cur["rot_err"][frame] = val
                    else:
                        cur["rot_mae"][frame] = val
                except Exception:
                    pass
    if cur is not None:
        sections.append(cur)
    return sections


def format_section(target_key, target_name, ckpt_path, sections):
    lines = []
    lines.append("=" * 80)
    lines.append(f"Target dataset: {target_name}  (key={target_key})")
    lines.append(f"Checkpoint: {ckpt_path}")
    lines.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    if not sections:
        lines.append("(no metrics parsed)")
        return "\n".join(lines) + "\n"
    # Sort step sizes from high to low (matches the eval order in the config: [10, 5, 1])
    sections_sorted = sorted(sections, key=lambda x: -x["step_size"])
    for sec in sections_sorted:
        ss = sec["step_size"]
        lines.append(f"\n--- step_size = {ss} ---")
        # Position RMSE row
        na = set(sec.get("not_evaluated", []))
        for label, key, fmt in (("Pos RMSE (m) ", "pos_rmse", ".4f"), ("Rot Err (deg)", "rot_err", ">6.3f"),
                                ("Rot MAE (deg)", "rot_mae", ">6.3f")):
            vals = sec.get(key, {})
            frames = sorted(set(vals) | na)
            if vals:
                # "/": a checkpoint off the step-size grid or at a warm-up frame, not evaluated
                lines.append(f"  {label} | " + " | ".join(f"@{f:>3d}" for f in frames))
                lines.append("                | " + " | ".join(format(vals[f], fmt) if f in vals else f"{'/':>6}"
                                                            for f in frames))
        if sec.get("mapping"):
            lines.append(f"  mapping: {sec['mapping']}")
        if sec.get("protocol"):
            lines.append(f"  protocol: {sec['protocol']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_one(target_key, base_cfg, base_cfg_path, checkpoint, num_gpus, project_dir, master_port_offset, compile_flag=False, cuda_graph_flag=False, split="test", render=0, floor="light", video_dir=None):
    target = TARGET_TABLE[target_key]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"  # unique per process: parallel evaluations must not share a log dir
    run_tag = f"eval_{target_key}_{ts}" + ("_val" if split == "val" else "")
    cfg = build_target_config(base_cfg, target_key, run_tag, render=render, floor=floor)
    if split == "val":
        # Evaluate the held-out validation scenes with the exact test protocol (for checkpoint
        # selection without touching the test split): write a split file whose test list is the
        # validation list. The training list is untouched.
        src_split = Path(project_dir) / cfg["data"]["split_file"]
        with open(src_split, "r") as f:
            sp = json.load(f)
        sp["test_datasets"] = list(sp["val_datasets"])
        sp["evaluated_partition"] = "val"   # reported by the evaluator's protocol line
        sp.setdefault("metadata", {})["note"] = f"validation scenes evaluated as the test split (derived from {src_split.name})"
        tmp_split_path = Path(project_dir) / "configs" / f"_tmp_{run_tag}_valsplit.json"
        with open(tmp_split_path, "w") as f:
            json.dump(sp, f, indent=1)
        cfg["data"]["split_file"] = str(tmp_split_path)
    tmp_cfg_path = Path(project_dir) / "configs" / f"_tmp_{run_tag}.yaml"
    dump_yaml(cfg, tmp_cfg_path)

    log_dir = Path(project_dir) / "exps" / "logs_multi" / run_tag / f"run_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["EXP_TIMESTAMP"] = ts
    env["DISABLE_NORMALIZERS"] = "1"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["MPLBACKEND"] = "Agg"
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_TIMEOUT"] = "28800"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(num_gpus)  # caller passes the device list string
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT_BASE", "29500")) + master_port_offset)

    nproc = len([x for x in num_gpus.split(",") if x.strip()])
    # Resolve torchrun next to the current Python interpreter, falling back to PATH.
    torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
    if not os.path.isfile(torchrun):
        torchrun = "torchrun"
    cmd = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={nproc}",
        "src/train.py",
        "--config",
        str(tmp_cfg_path),
        "--checkpoint",
        str(checkpoint),
        "--use_multi_dataset",
        "--inference",
    ]
    if compile_flag:
        cmd.append("--compile")
    if cuda_graph_flag:
        cmd.append("--cuda-graph")

    log_file = log_dir / "stdout.log"
    print(f"\n[eval] target={target_key} ({target['name']}), log_dir={log_dir}")
    print(f"[eval] cmd: {' '.join(cmd)}")
    print(f"[eval] CUDA_VISIBLE_DEVICES={num_gpus}, MASTER_PORT={env['MASTER_PORT']}")
    sys.stdout.flush()

    with open(log_file, "w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"[eval][ERROR] target {target_key} failed (rc={proc.returncode})")
        print(f"[eval] last 30 lines of {log_file}:")
        try:
            with open(log_file, "r") as lf:
                tail = lf.readlines()[-30:]
                sys.stdout.write("".join(tail))
        except Exception:
            pass
        return None

    eval_txt = log_dir / "eval.txt"
    if not eval_txt.exists():
        print(f"[eval][ERROR] eval.txt not found for {target_key}: {eval_txt}")
        return None
    sections = parse_eval_txt(eval_txt)
    # complete results only: every configured step size, every checkpoint, finite position and rotation values
    want_ss = [int(x) for x in cfg["data"].get("eval_step_sizes", [cfg["data"].get("skip_step_size", 1)])]
    want_cp = [int(x) for x in cfg["training"].get("metrics_test", {}).get("hopnet_checkpoints", [])]
    by_ss = {sec["step_size"]: sec for sec in sections}
    problems = []
    for ss in want_ss:
        sec = by_ss.get(ss)
        if sec is None:
            problems.append(f"step size {ss} missing")
            continue
        warmup = int(cfg["training"].get("metrics_test", {}).get("eval_warmup_frames", 2))   # the evaluator's default
        for cp in want_cp:
            if cp % ss or cp // ss < warmup:          # off the step-size grid or a warm-up frame: "/"
                if cp not in sec["not_evaluated"]:
                    problems.append(f"step size {ss} @{cp}: expected '/', found a value")
                continue
            for key in ("pos_rmse", "rot_err"):
                v = sec[key].get(cp)
                if v is None or not math.isfinite(v):
                    problems.append(f"step size {ss} {key} @{cp}: {v}")
    if problems:
        print(f"[eval][ERROR] incomplete results for {target_key}: {problems[:6]}")
        return None
    if render > 0:
        # inference_epoch_0000_dataset_<n>_<split>_size_<s>.mp4 -> <video_dir>/<target>/scene_<n>_step_<s>.mp4
        dest = Path(video_dir) / target["name"]
        dest.mkdir(parents=True, exist_ok=True)
        videos = sorted(log_dir.glob("inference_epoch_*_dataset_*_size_*.mp4"))
        for v in videos:
            parts = v.stem.split("_")
            shutil.move(str(v), str(dest / f"scene_{parts[parts.index('dataset') + 1]}_step_{parts[-1]}.mp4"))
        want = int(render) * len(want_ss)
        if len(videos) != want:
            print(f"[eval][ERROR] {len(videos)} of {want} videos rendered for {target_key} (see {log_file})")
            return None
        print(f"[eval] {len(videos)} videos -> {dest}")
    return {
        "target_key": target_key,
        "target_name": target["name"],
        "log_dir": str(log_dir),
        "eval_txt": str(eval_txt),
        "sections": sections,
        "tmp_config": str(tmp_cfg_path),
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-dataset evaluation of a trained checkpoint")
    ap.add_argument("--base_config", default="configs/movi_b.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/rigidformer_movib.pth")
    ap.add_argument("--output_file", default="eval_results.txt")
    ap.add_argument("--targets", default="B,A,S")
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                    help="CUDA_VISIBLE_DEVICES string, e.g. '0' or '0,1,2,3' (default: the one already set, e.g. by Slurm, else 0)")
    ap.add_argument("--compile", action="store_true", help="Forward --compile to the evaluation processes")
    ap.add_argument("--cuda-graph", action="store_true", help="Forward --cuda-graph to the evaluation processes")
    ap.add_argument("--position_estimator", choices=["config", "area_centroid", "hopnet"], default="config",
                    help="Object-position estimator: the base config's (default), the area-weighted surface centroid "
                         "of the predicted mesh, or HopNet's shape matching (exact for a perfect prediction)")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="Scenes to evaluate: the test split (default, reported numbers) or the validation split "
                         "(for selecting checkpoints / hyper-parameters without using the test scenes)")
    ap.add_argument("--render", type=int, default=0,
                    help="After the metrics, render the model's predictions on this many test scenes per step size "
                         "(0: no videos)")
    ap.add_argument("--floor", choices=["light", "dark"], default="light", help="Floor of the rendered videos")
    ap.add_argument("--video_dir", default="videos", help="Rendered videos: <video_dir>/<dataset>/scene_<n>_step_<s>.mp4")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    base_cfg = load_yaml(args.base_config)
    if args.position_estimator != "config":
        base_cfg.setdefault("training", {}).setdefault("metrics_test", {})["position_estimator"] = args.position_estimator

    targets = [t.strip().upper() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in TARGET_TABLE:
            raise SystemExit(f"Unknown target {t}, expected one of {list(TARGET_TABLE.keys())}")

    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    results = []
    failed = []
    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = project_dir / out_path

    # Header (overwrite)
    header = []
    header.append("=" * 80)
    header.append("Cross-Dataset Generalization Evaluation")
    header.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.append(f"Base config: {args.base_config}")
    header.append(f"Checkpoint:  {args.checkpoint}")
    header.append(f"Checkpoint sha256: {sha256(args.checkpoint)}")
    header.append(f"Config sha256:     {sha256(args.base_config)}")
    header.append(f"Targets:     {targets}")
    header.append(f"Split:       {args.split}")
    header.append(f"Estimator:   {base_cfg.get('training', {}).get('metrics_test', {}).get('position_estimator', 'area_centroid')}")
    header.append("=" * 80)
    header.append("")
    out_path.write_text("\n".join(header) + "\n")

    for i, target_key in enumerate(targets):
        result = run_one(
            target_key=target_key,
            base_cfg=base_cfg,
            base_cfg_path=args.base_config,
            checkpoint=args.checkpoint,
            num_gpus=args.gpus,
            project_dir=str(project_dir),
            master_port_offset=i,
            compile_flag=args.compile,
            cuda_graph_flag=args.cuda_graph,
            split=args.split,
            render=args.render,
            floor=args.floor,
            video_dir=args.video_dir if Path(args.video_dir).is_absolute() else project_dir / args.video_dir,
        )
        if result is None:
            with open(out_path, "a") as fh:
                fh.write(f"\n[FAILED] target {target_key} (see logs)\n\n")
            failed.append(target_key)
            continue
        section_text = format_section(
            target_key=result["target_key"],
            target_name=result["target_name"],
            ckpt_path=args.checkpoint,
            sections=result["sections"],
        )
        with open(out_path, "a") as fh:
            fh.write(section_text)
        results.append(result)

    # Summary appendix: machine-readable JSON dump
    with open(out_path, "a") as fh:
        fh.write("\n" + "=" * 80 + "\n")
        fh.write("Machine-readable JSON dump\n")
        fh.write("=" * 80 + "\n")
        compact = []
        for r in results:
            compact.append(
                {
                    "target": r["target_key"],
                    "name": r["target_name"],
                    "log_dir": os.path.relpath(r["log_dir"], project_dir),
                    "metrics": r["sections"],
                }
            )
        fh.write(json.dumps(compact, indent=2) + "\n")

    if failed:
        # a failed target must fail the driver (and eval.sh), not only leave a note in the output file
        raise SystemExit(f"[eval] targets failed: {failed}; output written to {out_path}")
    print(f"\n[eval] All targets completed. Output written to {out_path}")


if __name__ == "__main__":
    main()
