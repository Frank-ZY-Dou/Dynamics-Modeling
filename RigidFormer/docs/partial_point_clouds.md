# Partial Point Clouds

This document describes how RigidFormer is trained and evaluated on partial point clouds from a single simulated depth camera on MOVi-Sphere, MOVi-A and MOVi-B.

Each object is represented by the points of its segment in one depth image, i.e. by the visible part of its surface, and a completion network adds points on the occluded side. The points are stored in the object frame and moved with the ground-truth poses, so every point keeps its identity over time: the training loss is on their ground-truth 3D positions, and nothing after the observation is read from the camera.

## Depth camera

`data.partial_view` ([`src/data/partial_view.py`](../src/data/partial_view.py)) places a pinhole depth camera in each scene, ray-casts the visual meshes at the observation frame and back-projects each object's segment to 3D.

| Setting | Value |
|---|---|
| Viewing direction | that of the scene's Kubric camera |
| Distance | 12 m from the arena centre |
| Image | 640 × 480 pixels, 58° vertical field of view |
| Observation | the last of the four warm-up frames |
| Points per object | at most 1,024, by farthest-point sampling |
| Unobserved objects | fewer than 16 points |

The original MOVi cameras frame only part of the arena (about 69% of the MOVi-B objects are visible at frame 3). At 12 m, the camera observes 751 of the 772 test objects of MOVi-A, 796 of 804 of MOVi-B and 756 of 772 of MOVi-Sphere. With Kubric's own camera, the ray cast reproduces the object pixel counts and image positions recorded by Kubric ([`scripts/check_depth_camera.py`](../scripts/check_depth_camera.py)). The camera poses of all scenes are in `data_generation/scene_cameras_{MoviA,MoviB,MoviSphere}.json`, written from the Hugging Face dataset by [`data_generation/export_scene_cameras.py`](../data_generation/export_scene_cameras.py).

During fine-tuning, each training sample is observed at the distance, field of view and resolution of the evaluation camera and at the elevation of its scene camera, from an azimuth drawn uniformly at random (`data.partial_view.train_azimuth_jitter_deg`).

## Completion network

The completion network ([`src/data/shape_completion.py`](../src/data/shape_completion.py), a PointNet encoder and an MLP decoder) reads up to 256 observed points of an object in a view-aligned frame, together with the camera elevation, the camera distance and the height of the points above the floor, and predicts 512 points on the object's surface. The observed and predicted points are merged and reduced to at most 1,024 points by farthest-point sampling; about 40-45% of the resulting points come from the network.

The network is trained on training scenes only: the 960 training scenes of MOVi-B and 2,400 additional scenes generated with MOVi-B's shapes, new seeds, continuous friction, restitution and density, and thrown, tossed and sliding objects ([`data_generation/augment_movi_scenes.py`](../data_generation/augment_movi_scenes.py), in the replay environment of [`data_generation/README.md`](../data_generation/README.md)). Training views are drawn at random (elevation 10-80°, distance 8-16 m, 320 × 240 or 640 × 480 pixels) and perturbed with depth noise, eroded segment boundaries and random occluders ([`configs/completion_data_movi_b_aug.yaml`](../configs/completion_data_movi_b_aug.yaml)). On the MOVi-B validation scenes, its Chamfer distance is 4.7 cm from the predicted points to the surface and 5.8 cm from the surface to the predicted points. The trained network is `completion/completion.pth` of [`frankzydou/RigidFormer`](https://huggingface.co/frankzydou/RigidFormer); the configs read it from `checkpoints/completion/completion.pth`. To train it yourself:

```bash
python data_generation/augment_movi_scenes.py --manifest KuBasic/KuBasic_local.json --out raw/MOVi-B-aug \
    --kinds movi:1200,toss:600,slide:600 --workers 32
python data_generation/build_augmented_dataset.py --raw raw/MOVi-B-aug --movi datasets/MOVi-B \
    --movi_scales data_generation/object_scales_MoviB.json --movi_split dataset_split_MoviB_1200.json \
    --assets RigidFormer-MOVi/assets/kubasic --out datasets/MOVi-B-aug
python scripts/make_completion_data.py --config configs/completion_data_movi_b_aug.yaml --split train --out completion/train.npz
python scripts/make_completion_data.py --config configs/completion_data_movi_b_aug.yaml --split val --out completion/val.npz
python scripts/train_completion.py --train completion/train.npz --val completion/val.npz \
    --out checkpoints/completion/completion.pth
```

## Fine-tuning

The released models are fine-tuned on these inputs for 60 epochs with all weights trained, at a learning rate of 4e-6 after a two-epoch warm-up with cosine decay (ten times that for the fusion layers of the acceleration head, as in the released models' training) and 21 scenes per step. The fine-tuned models are `rigidformer_movi{sphere,a,b}_partial.pth` of [`frankzydou/RigidFormer`](https://huggingface.co/frankzydou/RigidFormer).

```bash
CONFIG=configs/movi_a_partial.yaml bash scripts/train.sh --checkpoint=checkpoints/rigidformer_movia.pth   # likewise movi_b_partial.yaml, movi_sphere_partial.yaml
```

Without the completion network, the released models are fine-tuned on the camera's points alone with the acceleration head (`predictor`) and the encoder's BatchNorm statistics kept at their released values (`training.optimizer.frozen_params: [predictor]`, `model.encoder.freeze_bn_stats: true`): the encoder and the decoder adapt to partial point clouds, and the learned dynamics stay as they are. The schedule is the same, 60 epochs at a learning rate of 4e-6 with 21 scenes per step, with an additional per-step linear warm-up from 0 over the first epoch. The fine-tuned models are `rigidformer_movi{sphere,a,b}_partial_nocompletion.pth`.

```bash
CONFIG=configs/movi_a_partial_nocompletion.yaml bash scripts/train.sh --checkpoint=checkpoints/rigidformer_movia.pth   # likewise movi_b_partial_nocompletion.yaml, movi_sphere_partial_nocompletion.yaml
```

## Evaluation

```bash
CHECKPOINT=checkpoints/rigidformer_movia_partial.pth BASE_CONFIG=configs/movi_a_partial.yaml TARGETS=A CUDA_GRAPH=1 bash scripts/eval.sh
CHECKPOINT=checkpoints/rigidformer_movia_partial_nocompletion.pth BASE_CONFIG=configs/movi_a_partial_nocompletion.yaml TARGETS=A CUDA_GRAPH=1 bash scripts/eval.sh   # without the completion network
```

The model rolls out the input points. For every object, the rigid motion of its points moves its full collision mesh, which is scored with the evaluation protocol of the [README](../README.md#-evaluation); the metrics are averaged over the observed objects. Test scenes, step size 1; position RMSE (m, top) and orientation RMSE (&deg;, bottom):

<table align="center">
<tr><th rowspan="2"><sub>Input and model</sub></th><th colspan="3"><sub>MOVi-A</sub></th><th colspan="3"><sub>MOVi-B</sub></th><th colspan="3"><sub>MOVi-Sphere</sub></th></tr>
<tr><th><sub>50</sub></th><th><sub>75</sub></th><th><sub>100</sub></th><th><sub>50</sub></th><th><sub>75</sub></th><th><sub>100</sub></th><th><sub>50</sub></th><th><sub>75</sub></th><th><sub>100</sub></th></tr>
<tr><td><sub>Full point clouds, released model</sub></td><td><sub>0.040<br>4.81</sub></td><td><sub>0.086<br>10.64</sub></td><td><sub>0.145<br>17.57</sub></td><td><sub>0.030<br>3.29</sub></td><td><sub>0.061<br>7.16</sub></td><td><sub>0.108<br>12.65</sub></td><td><sub>0.022<br>2.93</sub></td><td><sub>0.045<br>6.28</sub></td><td><sub>0.078<br>10.74</sub></td></tr>
<tr><td><sub>Partial point clouds, released model</sub></td><td><sub>0.055<br>4.84</sub></td><td><sub>0.111<br>10.69</sub></td><td><sub>0.182<br>17.84</sub></td><td><sub>0.041<br>3.75</sub></td><td><sub>0.083<br>8.04</sub></td><td><sub>0.142<br>14.29</sub></td><td><sub>0.053<br>3.30</sub></td><td><sub>0.110<br>7.15</sub></td><td><sub>0.184<br>12.65</sub></td></tr>
<tr><td><sub>Partial point clouds, fine-tuned</sub></td><td><sub>0.055<br>4.82</sub></td><td><sub>0.110<br>10.52</sub></td><td><sub>0.182<br>17.53</sub></td><td><sub>0.040<br>3.67</sub></td><td><sub>0.080<br>7.90</sub></td><td><sub>0.136<br>13.90</sub></td><td><sub>0.053<br>3.26</sub></td><td><sub>0.108<br>7.11</sub></td><td><sub>0.180<br>12.60</sub></td></tr>
<tr><td><sub>Partial point clouds + completion, released model</sub></td><td><sub>0.052<br>5.24</sub></td><td><sub>0.097<br>10.73</sub></td><td><sub>0.162<br>18.02</sub></td><td><sub>0.036<br>3.75</sub></td><td><sub>0.075<br>7.88</sub></td><td><sub>0.129<br>13.48</sub></td><td><sub>0.023<br>3.08</sub></td><td><sub>0.048<br>6.54</sub></td><td><sub>0.083<br>11.10</sub></td></tr>
<tr><td><sub>Partial point clouds + completion, fine-tuned</sub></td><td><sub>0.047<br>4.71</sub></td><td><sub>0.092<br>9.78</sub></td><td><sub>0.156<br>16.44</sub></td><td><sub>0.034<br>3.54</sub></td><td><sub>0.069<br>7.54</sub></td><td><sub>0.120<br>13.21</sub></td><td><sub>0.026<br>2.92</sub></td><td><sub>0.054<br>6.23</sub></td><td><sub>0.093<br>10.63</sub></td></tr>
</table>

Without the completion network, fine-tuning reduces the released models' mean position error over 50, 75 and 100 frames by 0.3% (MOVi-A), 4.0% (MOVi-B) and 1.8% (MOVi-Sphere) and the mean orientation error by 1.5%, 2.3% and 0.6%. The completion network reduces the released models' mean position error by 10% (MOVi-A), 10% (MOVi-B) and 56% (MOVi-Sphere). With it, fine-tuning reduces the mean orientation error by a further 9%, 3% and 5% and the mean position error by 5% (MOVi-A) and 7% (MOVi-B); on MOVi-Sphere it raises the position error by 12%.

## Data

The partial point clouds of all scenes are part of the Hugging Face dataset [`frankzydou/RigidFormer-MOVi`](https://huggingface.co/datasets/frankzydou/RigidFormer-MOVi) (configs `movi_sphere_partial`, `movi_a_partial` and `movi_b_partial`). For each scene they hold the camera and, at the observation frame of each evaluation step size (frames 3, 15 and 30 for step sizes 1, 5 and 10), the depth image, the segmentation and the observed points of every object, without completion. [`data_generation/export_partial_views.py`](../data_generation/export_partial_views.py) writes them; the dataset card describes the fields.

```bash
hf download frankzydou/RigidFormer-MOVi --repo-type dataset --local-dir RigidFormer-MOVi --include "data/movi_*_partial/*"
```

## Images and videos

The depth images, segmentations and rollout videos of the README (likewise for MOVi-B and MOVi-Sphere):

```bash
python scripts/plot_partial_view.py --config configs/movi_a_partial.yaml --scene movi_423 --out partial_movi_423
python scripts/export_rollouts.py --checkpoint checkpoints/rigidformer_movia_partial.pth --config configs/movi_a_partial.yaml --target A \
    --out rollouts/movi_a_partial
python scripts/render_rollouts.py --rollouts rollouts/movi_a_partial --best 2 --data datasets/MOVi-A --steps 1 \
    --config configs/movi_a_partial.yaml --out videos/movi_a_partial
```
