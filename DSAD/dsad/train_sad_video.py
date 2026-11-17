"""Fit one GoP with DSAD: SAD sites (or Gaussians) and the anchor ROM.

The loop follows D2GV's train_deform.py (itself adapted from GaussianImage's
train.py) in two stages: `coarse` fits the
canonical appearance without deformation for --coarse_iterations, then `fine`
optimizes the appearance and the ROM jointly for --iterations. Frame i of T has
the timestamp t = (i + 1) / T, in training and in evaluation.

Outputs, in <output_dir>/<data_name>/SAD3_<iterations>_<N>/<gop>/:
    train.txt                                          the log
    run_config.json                                    the resolved options
    gaussian.pth                                       the sites (or Gaussians)
    deform_sad/iteration_<iterations>/deform_sad.pth   the ROM
    finished.json                                      completion marker (dsad/runinfo.py)

The checkpoint keeps each site's axis as a 2-vector (10 numbers per site).
dsad/export.py writes the stored form (9 numbers per site), which
dsad/decode.py decodes.

Usage, with the paper's configuration (configs/benchmarks.yaml lists N and M of
every benchmark GoP, and scripts/make_jobs.py writes the command lines):
    python dsad/train_sad_video.py --config configs/dsad.yaml \\
        -d data/uvg_proto/uvg720/Beauty/Beauty_1 --num_points 23714 --num_anchors 256 --data_name tab2

The command-line defaults are not the paper's configuration: use a config file
or pass every option. Options on the command line override the config file.
"""
from __future__ import annotations

import math
import os
import sys
import time
import random
from pathlib import Path

import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pytorch_msssim import ms_ssim
from tqdm import tqdm

# D2GV's modules (Adan for the sites; with --appearance gaussian also GaussianImage_Cholesky,
# gsplat and the loss) and the DSAD modules are imported as top-level modules.
try:
    from . import _paths
except ImportError:
    import _paths
_paths.setup()

try:
    from sad_appearance_v3 import SADAppearance_V3 as SADAppearance
except Exception as _e:
    SADAppearance = None
    _sad_import_err = _e
_paths.require_d2gv()

from deformation_core import DeformModelSAD
import losses_motion
import diagnostic
import runinfo


class LogWriter:
    """The run's log, <run>/train.txt. write() prints the text and appends it to the file, so
    a run repeated in the same directory adds its log to the existing one."""

    def __init__(self, run_dir: str | Path):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(run_dir) / "train.txt"

    def write(self, text: str) -> None:
        print(text)
        with open(self.path, "a") as f:
            f.write(text + "\n")


def _parse_lr_mults(spec: str):
    """Parse 'xyz=0.1,color=2' into {"xyz": 0.1, "color": 2.0}; an empty string gives None."""
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        key, val = item.split("=")
        out[key.strip()] = float(val)
    return out


class SimpleTrainer:
    """The coarse and fine fit of one GoP, with its evaluation, log and checkpoint."""

    def __init__(self, image_path: Path, num_points: int, iterations: int, args):
        self.device = torch.device("cuda:0")
        self.image_list = self._load_images(image_path)
        self.original_image_list = self.image_list.copy()
        self.gt_image = self.image_list[0]

        self.num_points = num_points
        self.image_name = image_path.stem
        # Frames that supervise the fit (--train_frames): all of them, or the even-indexed ones
        # only. The odd frames then keep their timestamps t_i = (i + 1) / T and are evaluated,
        # never trained on (the temporal-interpolation study).
        T0 = len(self.original_image_list)
        self._train_idx = list(range(0, T0, 2)) if args.train_frames == "even" else list(range(T0))
        self._train_pos = 0
        # --frame_order shuffle: the frame indices left in the current pass, and a generator
        # of its own, so that the other random streams do not depend on the frame order.
        self._order_queue: list[int] = []
        self._order_gen = torch.Generator().manual_seed(int(getattr(args, "seed", 0) or 0) + 104729)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.args = args

        out_root = Path(getattr(args, "output_dir", None) or "checkpoints")
        self.log_dir = out_root / args.data_name / f"SAD3_{iterations}_{num_points}" / self.image_name
        self.logwriter = LogWriter(self.log_dir)

        BLOCK_H, BLOCK_W = 16, 16
        appearance = getattr(args, "appearance", "gaussian")
        # Combinations that would otherwise be silently ignored.
        if getattr(args, "sad_step_size", 20000) < 1:
            raise ValueError("--sad_step_size must be >= 1")
        if appearance != "sad" and (getattr(args, "sad_lr_mults", "") or getattr(args, "sad_init", "random") != "random"
                                    or getattr(args, "sad_step_size", 20000) != 20000
                                    or getattr(args, "cand_exact_tail", 0)):
            raise ValueError("--sad_* flags and --cand_exact_tail apply to --appearance sad only")
        if int(getattr(args, "cand_exact_tail", 0) or 0) > iterations:
            print(f"WARNING: --cand_exact_tail {args.cand_exact_tail} exceeds --iterations {iterations}: "
                  f"the exact-candidate tail never starts (it covers the last N fine iterations).")
        if appearance == "sad":
            if SADAppearance is None:
                raise RuntimeError(f"SADAppearance unavailable: {_sad_import_err}")
            self.gaussian_model = SADAppearance(
                loss_type="L2", opt_type="adan",
                lr_mults=_parse_lr_mults(getattr(args, "sad_lr_mults", "")),
                num_points=num_points, H=self.H, W=self.W,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                device=self.device, lr=args.lr, quantize=False,
                cand_freq=args.cand_freq,
                cand_passes_per_update=args.cand_passes_per_update,
                cand_warmup_passes=args.cand_warmup_passes,
                cand_inject_count=args.cand_inject_count,
                cand_flood_every=args.cand_flood_every,
                num_frames=len(self.original_image_list),
                # --sad_init gradient: placement on the temporal mean of the supervising frames
                init_image=(torch.cat([self.original_image_list[i] for i in self._train_idx]).mean(0)
                            .permute(1, 2, 0)
                            if args.sad_init == "gradient" else None),
                step_size=getattr(args, "sad_step_size", 20000),
                cand_method="vpt",
            ).to(self.device)
        else:
            # DSAD-GS ablation: D2GV's Gaussian primitive and its loss. Imported here, so the SAD
            # path does not need gsplat's CUDA extension.
            from gaussianimage_cholesky import GaussianImage_Cholesky
            from utils import loss_fn   # D2GV's utils.py
            self._gs_loss = loss_fn
            self.gaussian_model = GaussianImage_Cholesky(
                loss_type="L2", opt_type="adan",
                num_points=num_points, H=self.H, W=self.W,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                device=self.device, lr=args.lr, quantize=False,
            ).to(self.device)
        self._is_sad_appearance = (appearance == "sad")

        self.deform = DeformModelSAD(
            num_anchors=args.num_anchors,
            num_knots=args.num_knots,
            anchor_init_sigma=args.anchor_init_sigma,
            time_init_radius=0.15,
            anchor_top_k=args.anchor_top_k,
            enable_opacity=args.enable_opacity,
            enable_features=args.enable_features,
            anchor_mode=args.anchor_mode,
            lr_anchor=args.lr_anchor,
            lr_coef=args.lr_coef,
            max_steps=iterations,
            device="cuda",
        )
        self.deform.train_setting()

        self.lambda_move = args.lambda_move
        self.lambda_temporal = args.lambda_temporal
        self.lambda_arap = args.lambda_arap
        self.lambda_spatial = args.lambda_spatial
        self.diag_every = getattr(args, "diag_every", 1000)

        self.knn_idx = None
        self.knn_rebuild_every = getattr(args, "knn_rebuild_every", 5000)
        self._gt_hwc_cache = None

        # The fused training step (--fast, dsad/fast_step.py).
        self._fast = None
        self._step_kind = "eager"
        if getattr(args, "fast", False):
            import fast_step
            reason = fast_step.unsupported_reason(self)
            if reason is None:
                self._fast = fast_step.FusedStep(self, fused_vpt=True)
                self._step_kind = "fused"
            else:
                self._step_kind = f"eager (--fast fell back: {reason})"
                print(f"--fast: not available for this configuration ({reason}); "
                      f"using the eager step.")
        # The fused decoder's rate is logged as FPS_fused.
        self._fps_key = "FPS_fused" if self._fast is not None else "FPS"

        self._log_init_report()
        self._write_run_config()

    def _write_run_config(self):
        """run_config.json: every resolved option plus the GoP's shape, so that a
        decoder can rebuild the model without parsing the log."""
        cfg = dict(vars(self.args))
        cfg.update(argv=sys.argv,
                   frames=len(self.original_image_list), height=self.H, width=self.W,
                   num_points=self.num_points, iterations=self.iterations,
                   appearance=getattr(self.args, "appearance", "gaussian"),
                   step=self._step_kind)
        with open(self.log_dir / "run_config.json", "w") as f:
            json.dump(cfg, f, indent=1, sort_keys=True, default=str)

    def _log_init_report(self):
        """Log the [provenance] lines and the start-of-run report (sizes, parameter counts)."""
        g_total = diagnostic.count_trainable(self.gaussian_model)
        core = self.deform.deform
        breakdown = diagnostic.rom_module_breakdown(core)
        rom_total = breakdown.pop("_total")

        # Provenance: the command line, config, git commit, seed, GPU and the step that runs.
        import subprocess
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        def _git(*a):
            try:
                return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                                      text=True, timeout=20).stdout.strip()
            except Exception:
                return "?"
        self.logwriter.write(
            f"[provenance] argv: {' '.join(sys.argv)}\n"
            f"[provenance] config: {getattr(self.args, 'config', None)}\n"
            f"[provenance] git: {_git('rev-parse', 'HEAD')}  "
            f"dirty_files: {len(_git('status', '--porcelain', '--untracked-files=no', '--', '.').splitlines())}\n"
            f"[provenance] seed: {getattr(self.args, 'seed', None)}  "
            f"gpu: {torch.cuda.get_device_name(0)}\n"
            f"[provenance] step: {self._step_kind}  optimizers: "
            f"appearance={type(self.gaussian_model.optimizer).__name__} "
            f"rom={type(self.deform.optimizer).__name__}"
            + ("\n[provenance] decode: fused, anchor basis once per GoP pass (rom_forward on the first "
               "frame, rom_apply after); FPS_fused includes it once per pass"
               if self._fast is not None else ""))

        lines = [
            "=" * 72,
            "DSAD anchor ROM: start-of-run report",
            "=" * 72,
            f"  GoP: {self.image_name}   frames={len(self.original_image_list)}   "
            f"HxW={self.H}x{self.W}",
            f"  Appearance primitives (N): {self.num_points}   "
            f"trainable params: {g_total:,}",
            f"  ROM rank M (anchors): {core.anchors.M}   "
            f"knots K: {core.coef_xyz.K}   "
            f"anchor top-K: {core.anchors.top_k}",
            f"  anchor σ init: {core.anchors.log_sigma.detach().exp()[0].item():.3f}",
            f"  ROM enable_opacity={core.enable_opacity}  "
            f"enable_features={core.enable_features}",
            f"  ROM params: {rom_total:,}  (breakdown: "
            f"{', '.join(f'{k}={v}' for k,v in breakdown.items())})",
        ]
        if self._is_sad_appearance:
            # Training keeps each site's axis as an unnormalized 2-vector (10 numbers per
            # site); the stored form keeps its angle (9), see dsad/export.py.
            lines.append(f"  Stored per GoP (dsad/export.py): 9 N + ROM = "
                         f"{9 * self.num_points + rom_total:,}  (trainable: 10 N + ROM = "
                         f"{10 * self.num_points + rom_total:,})")
        lines += [
            f"  Regularizer λs: move={self.lambda_move}  temporal={self.lambda_temporal}  "
            f"arap={self.lambda_arap}  spatial={self.lambda_spatial}",
            "=" * 72,
        ]
        self.logwriter.write("\n".join(lines))

    def _gt_hwc(self, frame_idx: int) -> torch.Tensor:
        """Frame `frame_idx` as a contiguous [H, W, 3] tensor (the SAD renderer's layout), cached."""
        if self._gt_hwc_cache is None:
            self._gt_hwc_cache = [img.squeeze(0).permute(1, 2, 0).contiguous()
                                  for img in self.original_image_list]
        return self._gt_hwc_cache[frame_idx]

    def _load_images(self, image_path):
        """Every file of the GoP directory, in sorted order, as [1, 3, H, W] tensors on the GPU."""
        imgs = []
        for f in sorted(os.listdir(image_path)):
            full = os.path.join(image_path, f)
            t = transforms.ToTensor()(Image.open(full)).unsqueeze(0).to(self.device)
            imgs.append(t)
        return imgs

    def _next_in_pass(self, frames: list[int]) -> int:
        """--frame_order shuffle: the next frame of the current pass over `frames`, starting a
        new pass (a fresh seeded permutation) when the last one is used up. Every frame is
        visited once per pass; the passes run on through both stages."""
        if not self._order_queue:
            perm = torch.randperm(len(frames), generator=self._order_gen).tolist()
            self._order_queue = [frames[k] for k in perm]
        return self._order_queue.pop(0)

    def _pop_image(self):
        """(frame, timestamp, frame index) of the next training iteration: frame i of T at
        t = (i + 1) / T, in cyclic order 0..T-1 or, with --frame_order shuffle, in a fresh
        seeded random order per pass over the GoP. With --train_frames even, the passes run
        over the supervising frames only."""
        if len(self._train_idx) != len(self.original_image_list):
            T = len(self.original_image_list)
            if getattr(self.args, "frame_order", "cyclic") == "shuffle":
                i = self._next_in_pass(self._train_idx)
            else:
                i = self._train_idx[self._train_pos % len(self._train_idx)]
                self._train_pos += 1
            return self.original_image_list[i], (i + 1) / T, i
        if getattr(self.args, "frame_order", "cyclic") == "shuffle":
            T = len(self.original_image_list)
            i = self._next_in_pass(list(range(T)))
            # The cyclic path's expression, so every frame keeps bit-identical timestamps.
            return self.original_image_list[i], 1.0 - (T - 1 - i) / T, i
        if len(self.image_list) == 0:
            self.image_list = list(self.original_image_list)
        T = len(self.original_image_list)
        # Index (0..T-1) of the frame popped next.
        frame_idx = T - len(self.image_list)
        img = self.image_list.pop(0)
        ts = 1.0 - len(self.image_list) / T
        return img, ts, frame_idx

    def train(self, stage):
        """Run the "coarse" or the "fine" stage, evaluate, and log the '[<stage>] DONE' line.
        Returns (PSNR, MS-SSIM)."""
        iterations = (getattr(self.args, "coarse_iterations", 10000)
                      if stage == "coarse" else self.iterations)
        pbar = tqdm(range(1, iterations + 1), desc=f"Training [{stage}]", disable=None)
        self.gaussian_model.train()
        start = time.time()

        # Running sums of the loss terms for the diagnostic line. They stay on the GPU in
        # float64: adding the float32 terms there gives the same sums as adding Python
        # floats, without a host synchronization per iteration.
        term_names = ("rec", "move", "temporal", "arap", "spatial")
        running = torch.zeros(len(term_names), dtype=torch.float64, device=self.device)
        running_n = 0
        zero = torch.zeros((), device=self.device)
        fast = self._fast
        if fast is not None:
            # The fused step keeps its own sums (same layout), updated on the GPU.
            running = fast.running
            running.zero_()
            fast.invalidate_basis()  # drop the anchor weights cached by a decode
        eb = getattr(self.gaussian_model, "exact_builder", None)
        if eb is not None:
            eb.reset()  # choose the exact builder's cap again (affects speed only)

        tail = int(getattr(self.args, "cand_exact_tail", 0) or 0)
        for it in range(1, iterations + 1):
            if stage == "fine" and tail and it == iterations - tail + 1 and self._is_sad_appearance:
                # The last `tail` fine iterations (and the final evaluation) use the
                # decoder's exact top-8 candidates, rebuilt on each frame's next use.
                gm = self.gaussian_model
                gm.cand_method = "exact"
                gm._frame_init = [False] * gm.num_frames
                gm.exact_builder.reset()
            image, timestamp, frame_idx = self._pop_image()
            gt = image
            if self._is_sad_appearance:
                # Same frame, already in the renderer's [H, W, 3] layout.
                gt = self._gt_hwc(frame_idx)

            if fast is not None:
                # Forward, backward and both optimizer updates in fused kernels.
                fast.step(stage, it, frame_idx, timestamp, gt)
                self.gaussian_model.scheduler.step()
                self.deform.update_learning_rate(it)
                running_n += 1
                d_xyz, d_op, d_feat = fast.deltas() if stage == "fine" else (None, None, None)
                if it % 10 == 0:
                    loss_v, mse_v = fast.cur[[4, 0]].tolist()
                    psnr = 10 * math.log10(1.0 / max(mse_v, 1e-12))
                    pbar.set_postfix({"Loss": f"{loss_v:.6f}", "PSNR": f"{psnr:.2f}"})
                    pbar.update(10)
                self._periodic_logs(stage, it, running, running_n, d_xyz, d_op, d_feat)
                if stage == "fine" and self.diag_every and it % self.diag_every == 0:
                    running_n = 0
                continue

            # Per-term losses (detached) for logging.
            term = dict.fromkeys(term_names, zero)

            if stage == "coarse":
                d_xyz = d_op = d_feat = None
                if self._is_sad_appearance:
                    rec_loss = self.gaussian_model.forward_mse_loss(
                        gt, frame_idx=frame_idx)
                    pred = None
                else:
                    render = self.gaussian_model.forward()
                    pred = render["render"]
                    rec_loss = self._gs_loss(pred, gt, self.gaussian_model.loss_type,
                                             lambda_value=0.8)
            else:
                xyz = self.gaussian_model.get_xyz
                d_xyz, d_op, d_feat = self.deform.step(xyz, timestamp)
                if self._is_sad_appearance:
                    rec_loss = self.gaussian_model.forward_dynamic_mse_loss(
                        d_xyz, d_op, d_feat, gt, frame_idx=frame_idx)
                    pred = None
                else:
                    render = self.gaussian_model.forward_dynamic(d_xyz, d_op, d_feat)
                    pred = render["render"]
                    rec_loss = self._gs_loss(pred, gt, self.gaussian_model.loss_type,
                                             lambda_value=0.8)
            loss = rec_loss
            term["rec"] = rec_loss.detach()

            if stage == "fine":
                # Motion regularizers (losses_motion.py).
                if self.lambda_move > 0:
                    move_l = losses_motion.movement_penalty(d_xyz)
                    loss = loss + self.lambda_move * move_l
                    term["move"] = move_l.detach()
                if self.lambda_temporal > 0:
                    temporal_l = losses_motion.temporal_smoothness(
                        self.deform.deform.coef_xyz)
                    loss = loss + self.lambda_temporal * temporal_l
                    term["temporal"] = temporal_l.detach()
                if self.lambda_arap > 0:
                    # The canonical positions move during the fine stage, so the KNN
                    # graph is rebuilt periodically.
                    if (self.knn_idx is None
                            or (self.knn_rebuild_every
                                and it % self.knn_rebuild_every == 0)):
                        self.knn_idx = losses_motion.build_knn_graph(
                            self.gaussian_model.get_xyz.detach(), k=6)
                    deformed = torch.tanh(self.gaussian_model._xyz + d_xyz)
                    arap_l = losses_motion.arap(
                        self.gaussian_model.get_xyz.detach(), deformed, self.knn_idx)
                    loss = loss + self.lambda_arap * arap_l
                    term["arap"] = arap_l.detach()
                if self.lambda_spatial > 0:
                    a_t = self.deform.deform.coef_xyz(
                        self.deform.timestamp_tensor(timestamp, self.device, torch.float32))
                    spatial_l = losses_motion.spatial_smoothness(
                        self.deform.deform.anchors, a_t)
                    loss = loss + self.lambda_spatial * spatial_l
                    term["spatial"] = spatial_l.detach()

            loss.backward()

            self.gaussian_model.optimizer.step()
            self.gaussian_model.optimizer.zero_grad(set_to_none=True)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.gaussian_model.scheduler.step()
            self.deform.update_learning_rate(it)

            running += torch.stack([term[k] for k in term_names]).double()
            running_n += 1

            if it % 10 == 0:
                # Progress display: the one host synchronization per 10 iterations.
                with torch.no_grad():
                    mse_t = (rec_loss.detach() if pred is None
                             else F.mse_loss(pred, image))
                    loss_v, mse_v = torch.stack([loss.detach(), mse_t]).tolist()
                psnr = 10 * math.log10(1.0 / max(mse_v, 1e-12))
                pbar.set_postfix({"Loss": f"{loss_v:.6f}", "PSNR": f"{psnr:.2f}"})
                pbar.update(10)

            self._periodic_logs(stage, it, running, running_n, d_xyz, d_op, d_feat)
            if stage == "fine" and self.diag_every and it % self.diag_every == 0:
                running_n = 0

        pbar.close()
        elapsed = time.time() - start
        avg_psnr, avg_ssim = self.test(log_per_frame=True)
        # One untimed pass, then the timed one.
        _ = self.test_fps()
        fps = self.test_fps()
        total_params = sum(p.numel() for p in self.gaussian_model.parameters()) + self.deform.num_params()
        done = (f"[{stage}] DONE  time={elapsed:.1f}s  PSNR={avg_psnr:.4f}  MS-SSIM={avg_ssim:.6f}  "
                f"{self._fps_key}={fps:.1f}  total_params={total_params}  rom_params={self.deform.num_params()}")
        self.logwriter.write(done)
        if stage == "fine":
            self._fine_done = done
        return avg_psnr, avg_ssim

    def _periodic_logs(self, stage, it, running, running_n, d_xyz, d_op, d_feat):
        """Log the diagnostic lines every diag_every fine iterations (resetting `running`)
        and an evaluation every 5,000 iterations."""
        if stage == "fine" and self.diag_every and it % self.diag_every == 0:
            names = ("rec", "move", "temporal", "arap", "spatial")
            self._log_diag(it, dict(zip(names, running.tolist())), running_n,
                           d_xyz, d_op, d_feat)
            running.zero_()
        if it % 5000 == 0:
            avg_psnr, avg_ssim = self.test()
            fps = self.test_fps()
            self.logwriter.write(
                f"[{stage}] iter={it}  PSNR={avg_psnr:.4f}  MS-SSIM={avg_ssim:.6f}  {self._fps_key}={fps:.1f}  "
                f"rom_params={self.deform.num_params()}")

    def _log_diag(self, it, running, running_n, d_xyz, d_op, d_feat):
        """Log the [diag] lines: the mean of every loss term over the last diag_every
        iterations with its share of the total, the learning rates, and ROM statistics
        (displacement magnitude, effective anchors per site, coefficient energy and the
        fraction of anchors that move)."""
        core = self.deform.deform
        xyz = self.gaussian_model.get_xyz
        # Loss terms.
        avg = {k: v / max(running_n, 1) for k, v in running.items()}
        weighted = {
            "rec":      avg["rec"],
            "move":     self.lambda_move     * avg["move"],
            "temporal": self.lambda_temporal * avg["temporal"],
            "arap":     self.lambda_arap     * avg["arap"],
            "spatial":  self.lambda_spatial  * avg["spatial"],
        }
        total = sum(weighted.values())
        pct = {k: (100.0 * v / total if total > 0 else 0.0)
               for k, v in weighted.items()}
        loss_line = (f"[diag iter={it}] loss(avg): "
                     f"rec={avg['rec']:.3e}({pct['rec']:.1f}%)  "
                     f"move(λ)={weighted['move']:.3e}({pct['move']:.2f}%)  "
                     f"tmp(λ)={weighted['temporal']:.3e}({pct['temporal']:.2f}%)  "
                     f"arap(λ)={weighted['arap']:.3e}({pct['arap']:.2f}%)  "
                     f"sp(λ)={weighted['spatial']:.3e}({pct['spatial']:.2f}%)")
        self.logwriter.write(loss_line)

        # Learning rates.
        gauss_lr = self.gaussian_model.optimizer.param_groups[0]["lr"]
        gauss_groups = self.gaussian_model.optimizer.param_groups
        gauss_rest = "".join(f"  gauss.{pg.get('name', f'g{i}')}={pg['lr']:.3e}"
                             for i, pg in enumerate(gauss_groups) if len(gauss_groups) > 1)
        deform_lrs = {pg.get("name", f"g{i}"): pg["lr"]
                      for i, pg in enumerate(self.deform.optimizer.param_groups)}
        self.logwriter.write(
            f"[diag iter={it}] lr: gauss={gauss_lr:.3e}{gauss_rest}  "
            + "  ".join(f"deform.{k}={v:.3e}" for k, v in deform_lrs.items())
        )

        # ROM statistics.
        with torch.no_grad():
            delt = diagnostic.delta_stats(
                d_xyz, d_op, d_feat, core.enable_opacity, core.enable_features)
            b = diagnostic.basis_stats(core.anchors, xyz)
            a_stats = diagnostic.anchor_stats(core.anchors)
            c_xyz = diagnostic.coef_stats(core.coef_xyz, "coef_xyz")
        rom_line = (f"[diag iter={it}] "
                    f"Δxyz_rms={delt['delta.xyz.rms']:.4f} "
                    f"Δxyz_max={delt['delta.xyz.abs_max']:.4f}  "
                    f"basis.eff_k={b['basis.eff_k']:.2f} "
                    f"basis.phi_max={b['basis.phi.max']:.3f}  "
                    f"μ.abs_max={a_stats['anchors.mu.abs_max']:.3f} "
                    f"σ.mean={a_stats['anchors.sigma.mean']:.3f}  "
                    f"coef.energy.mean={c_xyz['coef_xyz.energy.mean']:.4f} "
                    f"active_frac={c_xyz['coef_xyz.active_frac']:.2f}")
        self.logwriter.write(rom_line)

        if core.enable_features:
            with torch.no_grad():
                c_f = diagnostic.coef_stats(core.coef_features, "coef_feat")
            self.logwriter.write(
                f"[diag iter={it}] feat: Δfeat_rms={delt['delta.feat.rms']:.4f} "
                f"feat_max={delt['delta.feat.abs_max']:.4f} "
                f"energy.mean={c_f['coef_feat.energy.mean']:.4f} "
                f"active_frac={c_f['coef_feat.active_frac']:.2f}")
        if core.enable_opacity:
            with torch.no_grad():
                c_o = diagnostic.coef_stats(core.coef_opacity, "coef_op")
            self.logwriter.write(
                f"[diag iter={it}] op(IGNORED by renderer): "
                f"Δop_rms={delt['delta.op.rms']:.4f} "
                f"energy.mean={c_o['coef_op.energy.mean']:.4f}")

    @torch.no_grad()
    def _render_eval(self, i: int, ts: float) -> torch.Tensor:
        """Frame i at time ts as [1, 3, H, W]. SAD appearance renders with the model's
        evaluation candidates (SADAppearance_V3.eval_candidates)."""
        if self._fast is not None:
            return self._fast.decode(i, ts).permute(2, 0, 1).unsqueeze(0).contiguous()
        d_xyz, d_op, d_feat = self.deform.step(self.gaussian_model.get_xyz, ts)
        if self._is_sad_appearance:
            return self.gaussian_model.forward_dynamic(d_xyz, d_op, d_feat, frame_idx=i)["render"]
        # GaussianImage_Cholesky.forward_dynamic takes no frame index.
        return self.gaussian_model.forward_dynamic(d_xyz, d_op, d_feat)["render"]

    def test(self, log_per_frame: bool = False):
        """Mean PSNR and MS-SSIM over the T frames of the GoP, frame i at t = (i + 1) / T as in
        training. log_per_frame also logs the statistics of the per-frame PSNR."""
        self.gaussian_model.eval()
        imgs = list(self.original_image_list)
        n = len(imgs)
        per_frame_psnr = []
        per_frame_ssim = []
        with torch.no_grad():
            for i in range(n):
                img = imgs[i]
                ts = (i + 1) / n
                pred = self._render_eval(i, ts)
                mse = F.mse_loss(pred.float(), img.float())
                per_frame_psnr.append(10 * math.log10(1.0 / mse.item()))
                per_frame_ssim.append(
                    ms_ssim(pred.float(), img.float(), data_range=1, size_average=True).item())
        self.gaussian_model.train()
        avg_p = sum(per_frame_psnr) / n
        avg_s = sum(per_frame_ssim) / n
        if log_per_frame:
            import numpy as _np
            pf = _np.asarray(per_frame_psnr)
            self.logwriter.write(
                f"[eval] per-frame PSNR: mean={pf.mean():.3f} std={pf.std():.3f} "
                f"min={pf.min():.3f}@f{int(pf.argmin())} "
                f"max={pf.max():.3f}@f{int(pf.argmax())}")
        return avg_p, avg_s

    def test_fps(self):
        """Frames per second of one evaluation pass over the GoP (CUDA-synchronized)."""
        self.gaussian_model.eval()
        imgs = self.original_image_list
        n = len(imgs)
        if self._fast is not None:
            self._fast.invalidate_basis()  # timed: the anchor basis once per pass
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for i in range(n):
                ts = (i + 1) / n
                _ = self._render_eval(i, ts)
        torch.cuda.synchronize()
        return n / (time.time() - t0)

    def save_model(self):
        """Save the checkpoint (the ROM, then the sites), then, after a finished fine stage,
        finished.json (dsad/runinfo.py), which names both files with their sizes and sha256."""
        path = str(self.log_dir)
        self.deform.save_weights(path, self.iterations)
        ckpt = os.path.join(path, "gaussian.pth")
        torch.save(self.gaussian_model.state_dict(), ckpt)
        done = getattr(self, "_fine_done", None)
        if done is not None:
            rom = os.path.join(path, "deform_sad", f"iteration_{self.iterations}", "deform_sad.pth")
            runinfo.write_finished(self.log_dir, done, [rom, ckpt])

    def dump_images(self):
        """Write the evaluation renders of all frames to <run>/frames/ (--save_imgs)."""
        self.gaussian_model.eval()
        outdir = os.path.join(str(self.log_dir), "frames")
        os.makedirs(outdir, exist_ok=True)
        n = len(self.original_image_list)
        with torch.no_grad():
            for i in range(n):
                ts = (i + 1) / n
                pred = self._render_eval(i, ts)
                img = transforms.ToPILImage()(pred.float().squeeze(0))
                img.save(os.path.join(outdir, f"frame_{i + 1:04d}.png"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DSAD: fit one GoP (coarse canonical stage, then the fine stage).")
    p.add_argument("--config", type=str, default=None,
                   help="YAML file of option defaults, keyed by flag name without the dashes "
                        "(configs/dsad.yaml is the paper's configuration). Command-line flags "
                        "override it.")
    p.add_argument("-d", "--dataset", type=str, required=True,
                   help="GoP directory. Every file in it is a frame, loaded in sorted order.")
    p.add_argument("--data_name", type=str, default="bunny",
                   help="Run tag: outputs go to <output_dir>/<data_name>/SAD3_<iterations>_<N>/<gop>/.")
    p.add_argument("--output_dir", type=str, default="checkpoints",
                   help="Root of the run directories (relative to the working directory).")
    p.add_argument("--iterations", type=int, default=60000,
                   help="Fine-stage iterations (the paper: 120000 in Tab. 2, 60000 elsewhere).")
    p.add_argument("--coarse_iterations", type=int, default=10000,
                   help="Coarse-stage (canonical, no deformation) iterations. The paper uses 10000.")
    p.add_argument("--frame_order", type=str, default="cyclic", choices=["cyclic", "shuffle"],
                   help="Order of the frames within each pass over the GoP: cyclic 0..T-1, or a "
                        "fresh seeded permutation per pass (the paper's configuration). Every "
                        "frame is visited once per pass; the passes run on through both stages. "
                        "With --train_frames even the passes run over the supervising frames only.")
    p.add_argument("--train_frames", type=str, default="all", choices=["all", "even"],
                   help="Frames that supervise the fit: all (the paper's configuration), or the "
                        "even-indexed ones only (the odd frames keep their timestamps and are "
                        "evaluated, never trained on).")
    p.add_argument("--num_points", type=int, default=20000,
                   help="Appearance primitives N (SAD sites, or Gaussians with --appearance gaussian).")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_imgs", action=argparse.BooleanOptionalAction, default=False,
                   help="After training, write the decoded frames to <run>/frames/.")
    # ---- ROM ----
    p.add_argument("--num_anchors", type=int, default=128,
                   help="ROM rank M: number of motion anchors.")
    p.add_argument("--num_knots", type=int, default=8,
                   help="Temporal knots K per anchor coefficient a_j(t).")
    p.add_argument("--anchor_top_k", type=int, default=8,
                   help="Local support of the anchor basis phi_j(x): softmax over the top-K anchors.")
    p.add_argument("--enable_opacity", action=argparse.BooleanOptionalAction, default=False,
                   help="Also learn ROM coefficients for Delta-opacity. Neither renderer uses "
                        "Delta-opacity, so this only adds parameters.")
    p.add_argument("--enable_features", action=argparse.BooleanOptionalAction, default=False,
                   help="Also learn ROM coefficients for Delta-features (RGB). The paper uses it.")
    p.add_argument("--lr_anchor", type=float, default=5e-3,
                   help="LR of the anchor parameters (positions, bandwidths, mode parameters).")
    p.add_argument("--lr_coef", type=float, default=5e-3,
                   help="LR of the temporal coefficients (knot times, widths, values). "
                        "The paper uses 2e-3.")
    # Regularizer weights (losses_motion.py). The paper's configuration sets the
    # movement, temporal and ARAP weights to 0.
    p.add_argument("--lambda_move", type=float, default=0.01)
    p.add_argument("--lambda_temporal", type=float, default=0.001)
    p.add_argument("--lambda_arap", type=float, default=0.005)
    p.add_argument("--lambda_spatial", type=float, default=0.0)
    p.add_argument("--anchor_mode", type=str, default="voronoi",
                   choices=["voronoi", "apollonius", "power", "apollonius_aniso"],
                   help="Tessellation of the anchor basis. voronoi: Euclidean. apollonius: "
                        "additive radius r_j. power: Laguerre (w_j on squared distance). "
                        "apollonius_aniso: the SAD metric (anisotropic + r_j), as in the paper.")
    p.add_argument("--anchor_init_sigma", type=float, default=0.25,
                   help="Initial anchor bandwidth sigma.")
    p.add_argument("--diag_every", type=int, default=1000,
                   help="Write ROM diagnostic lines every N fine iterations (0 disables).")
    p.add_argument("--knn_rebuild_every", type=int, default=5000,
                   help="Rebuild the canonical KNN graph of the ARAP term every N iterations.")
    # ---- Appearance ----
    p.add_argument("--appearance", type=str, default="gaussian",
                   choices=["gaussian", "sad"],
                   help="'sad': SAD sites (SADAppearance_V3), the full method. 'gaussian': "
                        "D2GV's GaussianImage_Cholesky, the DSAD-GS ablation. Default 'gaussian'; "
                        "configs/dsad.yaml sets 'sad'.")
    p.add_argument("--sad_lr_mults", type=str, default="",
                   help="(SAD) Per-attribute multipliers on --lr, e.g. 'color=2,dir=0.5'. "
                        "Keys: xyz, color, tau, radius, dir, aniso. Empty (default): one "
                        "shared LR. The paper: color=2,tau=2,radius=2,dir=0.5,aniso=5.")
    p.add_argument("--sad_step_size", type=int, default=20000,
                   help="(SAD) StepLR period of the site LR: halve it every this many coarse + "
                        "fine steps. The paper: 40000 at 120k fine iterations (Tab. 2), 20000 "
                        "at 60k.")
    p.add_argument("--sad_init", type=str, default="random", choices=["random", "gradient"],
                   help="(SAD) Site placement: uniform random, or gradient-weighted on the "
                        "GoP's temporal-mean frame (the paper's configuration).")
    # ---- SAD candidates (per-frame buffers) ----
    p.add_argument("--cand_freq", type=int, default=8,
                   help="(SAD) Run a VPT candidate pass every N training renders of a frame. "
                        "The paper uses 2.")
    p.add_argument("--cand_passes_per_update", type=int, default=1,
                   help="(SAD) VPT passes per cand_freq tick.")
    p.add_argument("--cand_warmup_passes", type=int, default=16,
                   help="(SAD) VPT passes after each JFA seed + flood (first use of a frame, "
                        "and every periodic re-flood).")
    p.add_argument("--cand_inject_count", type=int, default=4,
                   help="(SAD) Random sites injected per VPT pass.")
    p.add_argument("--cand_flood_every", type=int, default=1000,
                   help="(SAD) Full JFA re-flood every N training renders of a frame (0 disables).")
    p.add_argument("--cand_exact_tail", type=int, default=0,
                   help="(SAD) Use the decoder's exact top-8 candidates (rebuilt every "
                        "--cand_freq renders of a frame) for the last N fine iterations and the "
                        "final evaluation; 0 = never. The paper's configuration uses 6000.")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False,
                   help="Fused CUDA training step and decoder (dsad/fast_step.py). Unsupported "
                        "configurations fall back to the eager PyTorch step.")
    p.add_argument("--fast_sad_render", action=argparse.BooleanOptionalAction, default=False,
                   help="With --fast: render with SAD's render_sad / render_sad_backward ops "
                        "instead of the fused render + MSE + backward kernel.")
    return p


def config_defaults(parser: argparse.ArgumentParser, path: str) -> dict:
    """Option defaults from a YAML config, {dest: value}.

    A key is a long flag name without the dashes. Values are checked against the flag's
    type, so '5e-3' (which YAML reads as a string) becomes a float. On the command line, a
    switch that the config turns on is turned off with --no-<flag>, e.g. --no-fast."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: expected a mapping of flag names to values")
    by_flag = {opt.lstrip("-"): a for a in parser._actions for opt in a.option_strings
               if opt.startswith("--") and not opt.startswith("--no-")}
    out = {}
    for key, value in cfg.items():
        action = by_flag.get(str(key))
        if action is None or action.dest in ("help", "config", "dataset"):
            raise SystemExit(f"{path}: unknown or per-run option {key!r}")
        value = _config_value(path, key, action, value)
        if action.choices is not None and value not in action.choices:
            raise SystemExit(f"{path}: {key}={value!r} is not one of {list(action.choices)}")
        out[action.dest] = value
    return out


def _config_value(path: str, key: str, action: argparse.Action, value):
    """A config value as its flag's type. YAML reads `yes` as true, `1.2e5` as a string and
    `[1, 2]` as a list, so values are checked, not cast. A switch takes true/false, an
    integer option an integral number, a float option a finite number (not .nan or .inf),
    and a string option a string."""
    if isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            raise SystemExit(f"{path}: {key} takes true or false, got {value!r}")
        return value
    if value is None:
        raise SystemExit(f"{path}: {key} has no value (leave the key out to keep the default)")
    kind = {int: "an integer", float: "a number"}.get(action.type, "a string")
    if isinstance(value, (bool, list, dict)):
        raise SystemExit(f"{path}: {key} takes {kind}, got {value!r}")
    if action.type in (int, float):
        try:
            number = float(value)
        except (ValueError, OverflowError):
            raise SystemExit(f"{path}: {key} takes {kind}, got {value!r}") from None
        if action.type is int:
            if not number.is_integer():
                raise SystemExit(f"{path}: {key} takes {kind}, got {value!r}")
            return int(number)
        if not math.isfinite(number):
            raise SystemExit(f"{path}: {key} takes a finite number, got {value!r}")
        return number
    if not isinstance(value, str):
        raise SystemExit(f"{path}: {key} takes {kind}, got {value!r}")
    return value


def parse_args(argv):
    p = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        p.set_defaults(**config_defaults(p, known.config))
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("DSAD trains on a CUDA GPU and PyTorch sees none: check CUDA_VISIBLE_DEVICES "
                         "and that the installed PyTorch has CUDA support.")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)

    if args.enable_opacity:
        print("WARNING: --enable_opacity only adds parameters: neither renderer uses "
              "the ROM's Delta-opacity.")
    trainer = SimpleTrainer(
        image_path=Path(args.dataset),
        num_points=args.num_points,
        iterations=args.iterations,
        args=args,
    )

    print("=== Stage 1: Coarse (canonical appearance) ===")
    trainer.train(stage="coarse")

    print("=== Stage 2: Fine (anchor-ROM deformation) ===")
    psnr, ssim_v = trainer.train(stage="fine")
    print(f"Final: PSNR={psnr:.4f}  MS-SSIM={ssim_v:.6f}  deform_params={trainer.deform.num_params()}")

    trainer.save_model()
    if args.save_imgs:
        trainer.dump_images()

    return psnr, ssim_v


if __name__ == "__main__":
    main(sys.argv[1:])
