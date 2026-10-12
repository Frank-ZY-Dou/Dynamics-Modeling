---
license:
- cc-by-4.0
- cc-by-sa-4.0
pretty_name: RigidFormer MOVi
task_categories:
- other
tags:
- physics
- rigid-body-dynamics
- simulation
- learned-simulator
- point-cloud
- mesh
- video
- kubric
- movi
size_categories:
- 1K<n<10K
configs:
- config_name: movi_sphere
  data_files:
  - split: train
    path: data/movi_sphere/train-*.parquet
  - split: validation
    path: data/movi_sphere/validation-*.parquet
  - split: test
    path: data/movi_sphere/test-*.parquet
- config_name: movi_a
  data_files:
  - split: train
    path: data/movi_a/train-*.parquet
  - split: validation
    path: data/movi_a/validation-*.parquet
  - split: test
    path: data/movi_a/test-*.parquet
- config_name: movi_b
  data_files:
  - split: train
    path: data/movi_b/train-*.parquet
  - split: validation
    path: data/movi_b/validation-*.parquet
  - split: test
    path: data/movi_b/test-*.parquet
- config_name: meshes
  data_files:
  - split: train
    path: data/meshes/meshes.parquet
- config_name: point_clouds
  data_files:
  - split: train
    path: data/point_clouds/point_clouds.parquet
---

# RigidFormer MOVi

MOVi-Sphere, MOVi-A and MOVi-B with every simulated object: multi-object rigid-body dynamics for learning and
evaluating simulators, as used by RigidFormer ([paper](https://arxiv.org/abs/2605.09196),
[project page](https://people.csail.mit.edu/frankzydou/projects/RigidFormer/index.html)).

Each dataset has 1,200 scenes. In each scene, 3 to 10 objects are placed at random above a floor with random initial
horizontal velocities biased toward the center, then fall and collide. Kubric simulates them with PyBullet for 2 s and
records 480 frames at 240 Hz.

| config | objects | scenes (train / validation / test) |
|---|---|---|
| `movi_sphere` | spheres, size 0.7 or 1.4 | 960 / 120 / 120 |
| `movi_a` | cube, cylinder or sphere, size 0.7 or 1.4 | 960 / 120 / 120 |
| `movi_b` | eleven KuBasic shapes, size uniform in [0.7, 1.4] | 960 / 120 / 120 |

Each object is either rubber (friction 0.8, restitution 0.7) or metal (friction 0.4, restitution 0.3). The floor has
friction 0.3 and restitution 0.5; gravity is 10 m/s² along -z.

## Contents

| part | content | location |
|---|---|---|
| scenes | one row per scene: settings, physical properties, the trajectory of every object, camera, image annotations | `data/movi_sphere`, `data/movi_a`, `data/movi_b` |
| videos | the rendered video of every scene, 256 x 256, 480 frames, H.265 | `videos/<config>/<scene>.mp4` |
| meshes | the 12 KuBasic collision meshes the simulation used, as `.obj` files and as arrays | `assets/kubasic/`, `data/meshes` |
| point clouds | per object shape: the mesh vertices, and surface samples of 128 to 4,096 points with normals | `data/point_clouds` |
| checksums | SHA-256 of every file; scene and object counts of every split | `manifest.json` |

## Source and completeness

The scenes are HopNet's MOVi datasets (Wei and Fink, Nature Communications 2025), split into train / validation / test
by HopNet's procedure (960 / 120 / 120). HopNet's generator and Kubric fork are on
[GitHub](https://github.com/AmauryWEI/hopnet); its raw Kubric files and rendered videos are on
[Zenodo](https://doi.org/10.5281/zenodo.15800434).

**Every simulated object is included.** The generator simulates every object it samples, but it writes into a
scene's metadata only the objects that appear in at least one rendered frame. In 1 MOVi-Sphere scene and 48 MOVi-B
scenes, one or two objects never come into view, so their trajectories were missing. Where such an object strikes a
recorded one (detected in 1 MOVi-Sphere and 18 MOVi-B scenes, with velocity jumps of up to 7.6 m/s), the recorded object
changes velocity with no visible cause.

We recovered these objects by replaying each scene's simulation from its seed and recorded settings, using HopNet's
generator, its Kubric fork and PyBullet 3.2.6. The replay reproduces every recorded object in all 3,600 scenes bit for
bit, so the recovered trajectories are the simulated ones. The `recovered` field marks them: 1 object in MOVi-Sphere and
52 in MOVi-B. Recovered objects come after the recorded ones in each scene. They are never in view, so the videos and
the image annotations of the recorded objects are HopNet's, unchanged.

## Geometry

`assets/kubasic/<shape>/collision_geometry.obj` are the collision meshes of Kubric's KuBasic assets, the files the
simulation used; `data/meshes` holds the same meshes as arrays. The mesh origin is the center of mass of the simulated
body. At frame t, object i is its mesh scaled by `size[i]`, rotated by `quaternions[t, i]` and moved to
`positions[t, i]`: a mesh vertex v is at `positions[t, i] + size[i] * R(quaternions[t, i]) v`. Points of
`data/point_clouds` move the same way, and their normals rotate with the object.

`data/point_clouds` has, for each of the 11 object shapes, a row with the mesh vertices (`sampling` = `vertices`),
which are the points RigidFormer takes as input, and rows with 128, 256, 512, 768, 1,024, 2,048 and 4,096 points drawn
uniformly by area on the mesh surface (`sampling` = `surface`, drawn with `numpy.random.default_rng(num_points)`). The
normals are the area-weighted vertex normals for the vertices, and the normals of the sampled faces for the surface
points.

The floor is the static KuBasic `dome` at scale 2. Its collision mesh is a slab whose top face is the plane z = 0 out to
a radius of 79 m.

## Camera, videos and image annotations

The camera is static and renders 256 x 256 images. `camera` holds Kubric's camera record: the intrinsics `K` (in
pixels, in Kubric's convention: the camera looks along its -z axis), the camera-to-world matrix `R`, the focal length
and sensor width (mm), the field of view (rad), and the position and orientation in every frame. Per object and frame,
`visibility` is the number of pixels the object covers in the rendered frame and `image_positions` is its center
projected into the image (x, y in pixels, also for frames where it is out of view). `bboxes` are the 2D boxes
(ymin, xmin, ymax, xmax, as fractions of the image height and width) in the frames listed in `bbox_frames`, the frames
in which the object is visible.

## Fields

### Scenes (`movi_sphere`, `movi_a`, `movi_b`)

One row per scene. Object `i` is the same object in every per-object field.

| field | type | content |
|---|---|---|
| `scene` | string | `movi_<seed>` |
| `seed` | int | Kubric random seed |
| `split` | string | `train`, `validation` or `test` |
| `num_frames`, `frame_rate` | int | 480 frames at 240 Hz |
| `step_rate` | int | simulation steps per second: 1200 for MOVi-Sphere, 2400 for MOVi-A and MOVi-B |
| `gravity` | float[3] | (0, 0, -10) m/s² |
| `floor_friction`, `floor_restitution` | float | 0.3, 0.5 |
| `generator` | struct | generator settings: `objects_set`, `camera`, `background`, `min_num_objects`, `max_num_objects` |
| `num_objects` | int | N |
| `shape`, `material` | string[N] | KuBasic shape, `rubber` or `metal` |
| `size`, `mass`, `friction`, `restitution` | float[N] | uniform scale of the mesh, kg, -, - |
| `color` | float[N][3] | RGB in [0, 1] |
| `size_label`, `color_label` | string[N] | Kubric's labels (`small` / `large`, colour names) in MOVi-Sphere and MOVi-A; empty in MOVi-B |
| `recovered` | bool[N] | recovered by replay (see above) |
| `positions` | float32 [T][N][3] | mesh origin (the center of mass), m |
| `quaternions` | float32 [T][N][4] | orientation, (w, x, y, z) |
| `velocities` | float32 [T][N][3] | m/s |
| `angular_velocities` | float32 [T][N][3] | rad/s, world frame |
| `resolution` | int[2] | (256, 256) |
| `background_color` | string | `clevr` (Kubric's CLEVR backdrop) in MOVi-Sphere and MOVi-A; the RGBA colour of the backdrop in MOVi-B |
| `camera` | struct | `K` float[3][3], `R` float[4][4], `focal_length`, `sensor_width`, `field_of_view`, `positions` float32 [T][3], `quaternions` float32 [T][4] |
| `visibility` | int32 [T][N] | visible pixels |
| `image_positions` | float32 [T][N][2] | projected center, pixels |
| `bbox_frames` | int32 [N][*] | frames in which the object is visible |
| `bboxes` | float32 [N][*][4] | 2D boxes in those frames |
| `video` | string | path of the scene's video in this repository |

The trajectories, the camera and the image annotations are the values Kubric recorded, unchanged.

### `meshes`

| field | type | content |
|---|---|---|
| `shape` | string | KuBasic shape (11 objects and the floor, `dome`) |
| `num_vertices`, `num_faces` | int | |
| `vertices` | float64 [V][3] | as in the `.obj` file, in its order |
| `faces` | int32 [F][3] | vertex indices |
| `vertex_normals` | float32 [V][3] | area-weighted vertex normals |

### `point_clouds`

| field | type | content |
|---|---|---|
| `shape` | string | KuBasic object shape |
| `sampling` | string | `vertices` or `surface` |
| `num_points` | int | number of points |
| `points` | float32 [P][3] | in the mesh frame |
| `normals` | float32 [P][3] | unit normals |

## Usage

```python
import numpy as np
from datasets import load_dataset

repo = "frankzydou/RigidFormer-MOVi"
scenes = load_dataset(repo, "movi_b", split="test")
clouds = load_dataset(repo, "point_clouds", split="train")


def rotation(q):
    """Rotation matrices of quaternions (..., 4) in (w, x, y, z) order."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


scene = scenes[0]
positions = np.asarray(scene["positions"], np.float64)              # (480, N, 3)
rotations = rotation(np.asarray(scene["quaternions"], np.float64))  # (480, N, 3, 3)
vertices = {r["shape"]: np.asarray(r["points"], np.float64) for r in clouds if r["sampling"] == "vertices"}

t = 100   # the objects' point clouds at frame 100
points = [positions[t, i] + scene["size"][i] * vertices[s] @ rotations[t, i].T for i, s in enumerate(scene["shape"])]
```

To download everything, including the videos and the `.obj` files:

```bash
hf download frankzydou/RigidFormer-MOVi --repo-type dataset --local-dir RigidFormer-MOVi
```

The videos are H.265; FFmpeg-based readers (PyAV, imageio-ffmpeg, torchvision) decode them.

## Evaluation in the RigidFormer paper

Models are trained on `train` and evaluated on the 120 `test` scenes. A rollout starts from the first four frames and
predicts the following ones autoregressively. Position error is the RMSE of the object positions (estimated from the
predicted mesh by HopNet's shape matching) and rotation error is the RMS angle over the objects of a scene, both
averaged over the scenes at frames 25, 50, 75, 100, 200, 300 and 400.

## License

The scenes, the videos and the image annotations are released under CC BY 4.0; they are derived from HopNet's MOVi
datasets, which are CC BY 4.0. The meshes in `assets/kubasic` are Kubric's KuBasic assets, licensed CC BY-SA 4.0, and
`data/meshes` and `data/point_clouds`, which are derived from them, are CC BY-SA 4.0 as well.

## Acknowledgments

The scenes, their renders and the generator come from HopNet by Amaury Wei and Olga Fink; the scenes were generated with
Kubric, and the object meshes are Kubric's KuBasic assets. We thank the authors for making them available.

## Citation

```bibtex
@inproceedings{dou2026rigidformer,
  title     = {RigidFormer: Learning Rigid Dynamics using Transformers},
  author    = {Dou, Zhiyang and Guo, Minghao and Wu, Haixu and Roble, Doug and Stuyck, Tuur and Matusik, Wojciech},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
@inproceedings{greff2022kubric,
  title     = {Kubric: A Scalable Dataset Generator},
  author    = {Greff, Klaus and Belletti, Francois and Beyer, Lucas and Doersch, Carl and Du, Yilun and
               Duckworth, Daniel and Fleet, David J. and Gnanapragasam, Dan and Golemo, Florian and Herrmann, Charles and
               Kipf, Thomas and Kundu, Abhijit and Lagun, Dmitry and Laradji, Issam and Liu, Hsueh-Ti (Derek) and
               Meyer, Henning and Miao, Yishu and Nowrouzezahrai, Derek and Oztireli, Cengiz and Pot, Etienne and
               Radwan, Noha and Rebain, Daniel and Sabour, Sara and Sajjadi, Mehdi S. M. and Sela, Matan and
               Sitzmann, Vincent and Stone, Austin and Sun, Deqing and Vora, Suhani and Wang, Ziyu and Wu, Tianhao and
               Yi, Kwang Moo and Zhong, Fangcheng and Tagliasacchi, Andrea},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
