<div align="center">
<h1>S4R: Scaling for Rigid-Body Interpenetration Resolution</h1>

<!-- <a href="PROJECT_PAGE_URL"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> -->
<!-- <a href="ARXIV_URL"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" alt="arXiv"></a> -->
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT"></a>
<a href="data/"><img src="https://img.shields.io/badge/Data-Kubric_%2B_HY3D-orange" alt="Bundled datasets"></a>

**SIGGRAPH Asia 2026 (Journal Track)**

Zhiyang Dou<sup>1*</sup>, Ang Zhao<sup>2*</sup>, Chen Peng<sup>3</sup>, Minghao Guo<sup>1</sup>, Haixu Wu<sup>1</sup>, Cheng Lin<sup>4</sup>,<br>
Yuan Liu<sup>5</sup>, Junfeng Yao<sup>2</sup>, Xiaohu Guo<sup>6</sup>, Wenping Wang<sup>7</sup> and Wojciech Matusik<sup>1&dagger;</sup>

<sup>1</sup>MIT CSAIL&emsp;<sup>2</sup>Xiamen University&emsp;<sup>3</sup>The University of Hong Kong&emsp;<sup>4</sup>Macau University of Science and Technology<br>
<sup>5</sup>The Hong Kong University of Science and Technology&emsp;<sup>6</sup>The University of Texas at Dallas&emsp;<sup>7</sup>Texas A&amp;M University

<sub><sup>*</sup>Equal contribution.&emsp;<sup>&dagger;</sup>Corresponding author.</sub>

<img src="assets/teaser.png" alt="S4R teaser" width="100%">
<p align="left"><sub><b>S4R resolves severe rigid-body interpenetrations at scale and produces
simulation-ready scenes.</b> <b>Left:</b> progressive scaling of 1,000 Kubric assets arranged to
spell &ldquo;SIGGRAPH ASIA,&rdquo; from an interpenetrating initialization to a penetration-free
configuration. <b>Right:</b> S4R efficiently converts cluttered 3D assets into simulation-ready
scenes for downstream robot policy training.</sub></p>
</div>

## Overview

Penetration-free configurations are a prerequisite for physical
simulation and any downstream contact-dependent task; generated or
roughly placed 3D scenes routinely contain mesh-level overlaps that
break those pipelines.

S4R resolves rigid-body interpenetration by progressive scaling: every
body is first shrunk about its own reference center to a small initial
scale, then restored to full size along a monotone scale schedule. Each
scale increment solves one small minimum-norm convex contact QP that
keeps the linearized separation margin; a conservative scale-event bound
and frozen-witness distance caching keep the number of exact mesh
queries low, and a bounded tail-refinement pass corrects the residual
contacts at full scale. Every returned pose is scored by a shared
mesh-level evaluator.

The clips below show the process from interpenetrating layouts to a
penetration-free state. Each clip pauses 2 s on the initial overlap, then
plays the optimization.

<table>
<tr>
<td><img src="gifs/seed1.gif"  width="240" /></td>
<td><img src="gifs/seed2.gif"  width="240" /></td>
<td><img src="gifs/seed3.gif"  width="240" /></td>
<td><img src="gifs/seed4.gif"  width="240" /></td>
</tr>
<tr>
<td><img src="gifs/seed5.gif"  width="240" /></td>
<td><img src="gifs/seed6.gif"  width="240" /></td>
<td><img src="gifs/seed7.gif"  width="240" /></td>
<td><img src="gifs/seed8.gif"  width="240" /></td>
</tr>
<tr>
<td><img src="gifs/seed9.gif"  width="240" /></td>
<td><img src="gifs/seed10.gif" width="240" /></td>
<td><img src="gifs/seed11.gif" width="240" /></td>
<td><img src="gifs/seed12.gif" width="240" /></td>
</tr>
<tr>
<td><img src="gifs/seed13.gif" width="240" /></td>
<td><img src="gifs/seed14.gif" width="240" /></td>
<td><img src="gifs/seed15.gif" width="240" /></td>
<td><img src="gifs/seed16.gif" width="240" /></td>
</tr>
</table>

## Layout

```
Penetration_Solving/
├── s4r/            CPU solver (progressive scaling + contact QP + FCL oracle)
│   ├── s4r_qp.py            solver: schedule, QP, SOI events, cache, tail
│   ├── mesh_collision.py    scene objects + shared mesh-level evaluator
│   ├── box_collision.py     rotation / OBB utilities
│   └── common.py            shared dataclasses
├── s4r_gpu/        GPU-native solver (NVIDIA Warp)
│   ├── s4r_gpu_native.py    full-GPU pipeline (detection + dual APGD QP)
│   ├── dual_apgd.py         dual accelerated projected-gradient QP solver
│   ├── warp_pair_contact_v3.py  Warp mesh contact oracle
│   └── oracle_v4.py         batched all-GPU oracle
├── scenes/         benchmark scene generation (fixed seeds)
│   ├── make_scene.py        entry point: kubric / hy3d / thingi
│   ├── generate_hy3d_scene.py
│   └── generate_thingi10k_scene.py
├── data/
│   ├── kubric_pool/         41 watertight household meshes (19 MB)
│   └── hy3d_processed/      300 generated volume meshes, decimated to ≤1500 faces (175 MB)
├── examples/            end-to-end demos (run_kubric.py, run_upright.py)
└── requirements.txt
```

## Getting started

```bash
pip install -r requirements.txt
python examples/run_kubric.py --N 40 --seed 42
```

Expected output on the bundled Kubric pool:

```
[init]  N=40 seed=42  penetrating pairs = 30
[final] penetrating pairs = 0   RMSD = 0.0379   solve = <1s
```

Scenes are deterministic in `(dataset, N, seed)`. The benchmark seeds in
the paper are 42, 123 and 456; `--dataset hy3d` uses the bundled
generated-mesh pool, and `--dataset thingi` streams meshes through the
`thingi10k` package on first use.

All timings below were measured on an Intel Xeon E5-2680 v4 CPU
(2.40 GHz) and an NVIDIA GeForce RTX 2080 Ti GPU.

### CPU solver at more sizes

```bash
python examples/run_kubric.py --N 100  --seed 42
python examples/run_kubric.py --N 500  --seed 42
python examples/run_kubric.py --N 1000 --seed 42
```

| N | init pen. | final pen. | RMSD | solve |
|---|---|---|---|---|
| 40 | 30 | 0 | 0.0379 | 0.27 s |
| 100 | 80 | 0 | 0.0333 | 0.62 s |
| 500 | 371 | 0 | 0.0344 | 4.5 s |
| 1000 | 759 | 0 | 0.0348 | 13.1 s |
| 2000 | 1587 | 0 | 0.0371 | 32.5 s |
| 3000 | 2423 | 0 | 0.0372 | 66.3 s |

### GPU solver

Requires an NVIDIA GPU and `warp-lang`; the first call in a process pays
a one-time kernel-compilation and allocation cost, included in the times
below.

```bash
python examples/run_kubric.py --N 40   --seed 42 --solver gpu
python examples/run_kubric.py --N 1000 --seed 42 --solver gpu
```

| N | init pen. | final pen. | RMSD | total |
|---|---|---|---|---|
| 40 | 30 | 0 | 0.0372 | 6.4 s |
| 100 | 80 | 0 | 0.0347 | 6.7 s |
| 500 | 371 | 0 | 0.0343 | 9.6 s |
| 1000 | 759 | 0 | 0.0341 | 15.7 s |
| 2000 | 1587 | 0 | 0.0354 | 34.6 s |
| 3000 | 2423 | 0 | 0.0359 | 65.8 s |

The GPU pipeline is also callable directly:

```python
from s4r_gpu.s4r_gpu_native import solve_s4r_gpu
```

### Upright-on-plane repair with rotation

The tabletop variant keeps every object standing on a common support
plane: roll/pitch tilt is driven to zero along the scale path while each
step optimizes in-plane translation and yaw. Snapshots at 30/60/90%
inflation and the final state are written to a JSON file.

```bash
python examples/run_upright.py --output upright_seed42.json  --N 12 --seed 42  --min-init-pen 8
python examples/run_upright.py --output upright_seed123.json --N 12 --seed 123 --min-init-pen 8
```

Both runs end with `final pen=0` and all objects upright on the plane;
the JSON holds the poses of every stage (`init`, `s30`, `s60`, `s90`,
`final`) for rendering.

## Data

- `data/kubric_pool/` — the 41-mesh household-object pool used by the
  Kubric benchmarks. Meshes originate from Google Scanned Objects
  (CC-BY 4.0); attribution to the original dataset applies.
- `data/hy3d_processed/` — 300 watertight volume meshes generated with
  [Tencent Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) and
  processed for collision use (decimated to ≤1500 faces, per-mesh visual
  and collision geometry). The generator is governed by the Tencent
  Hunyuan3D Community License; the generated meshes are distributed here
  for research use.
- Thingi10K scenes download on demand via the `thingi10k` package;
  per-model licenses are preserved by upstream and nothing is
  redistributed here.

## Citation

```bibtex
@article{dou2026s4r,
  title   = {S4R: Scaling for Rigid-Body Interpenetration Resolution},
  author  = {Dou, Zhiyang and Zhao, Ang and Peng, Chen and Guo, Minghao and
             Wu, Haixu and Lin, Cheng and Liu, Yuan and Yao, Junfeng and
             Guo, Xiaohu and Wang, Wenping and Matusik, Wojciech},
  journal = {ACM Transactions on Graphics (SIGGRAPH Asia)},
  year    = {2026},
}
```

## License

Code is released under the [MIT License](LICENSE). Bundled meshes keep
their upstream licenses (see [Data](#data)).
