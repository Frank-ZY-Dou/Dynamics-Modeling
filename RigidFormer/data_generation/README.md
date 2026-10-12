# Training data

RigidFormer trains on MOVi-Sphere, MOVi-A and MOVi-B (HopNet's MOVi datasets, 1,200 scenes each, split 960 / 120 / 120)
with every simulated object in every scene. The training layout is one folder per dataset (`datasets/MOVi-Sphere`,
`datasets/MOVi-A`, `datasets/MOVi-B`) with one folder per scene:

```
movi_<n>/frame_00000/obj_<i>_pos.npy   per-frame vertex positions (Nv, 3), R(q) v_asset + p in asset units
movi_<n>/meshes/obj_<i>_faces.npy      face indices (F, 3)
movi_<n>/anchors/obj_<i>_anchors_fps<K>.npy   deterministic FPS anchors, K in {1,2,3,4,6,8,12,16}
movi_<n>/physics_params.json           per-object mass, friction, restitution, shape, material, size
movi_<n>/vis_traj.npy                  object origin + quaternion (WXYZ) trajectory, the metadata poses
movi_<n>/conversion.json               manifest, written last: converter version, metadata and mesh hashes, object counts
```

The stored vertices are in asset units; training and evaluation apply each object's size at load time from
`object_scales_*.json` (the configs' `data.object_scale_file`), which also records each object's unscaled frame-0
radius so that the loader refuses data that does not match.

## From Hugging Face

[`frankzydou/RigidFormer-MOVi`](https://huggingface.co/datasets/frankzydou/RigidFormer-MOVi) holds the three
datasets as Parquet, with the meshes, point clouds, camera, image annotations and rendered videos (the dataset card
describes the fields). `materialize_hf_dataset.py` writes the training layout from the scenes and meshes with the
converter's own code, identical to the conversion of the raw scenes below:

```bash
hf download frankzydou/RigidFormer-MOVi --repo-type dataset --local-dir RigidFormer-MOVi \
    --include "data/movi_*/*" --include "assets/*" --include manifest.json
python data_generation/materialize_hf_dataset.py --dataset RigidFormer-MOVi --out datasets --workers 16
```

## From HopNet's raw scenes

### 1. Download

HopNet's MOVi-spheres, MOVi-A and MOVi-B are on Zenodo, https://doi.org/10.5281/zenodo.15800434 (CC BY 4.0; linked from
https://github.com/AmauryWEI/hopnet, README section 3.1):

```bash
mkdir -p raw && cd raw
for d in MOVi-spheres MOVi-A MOVi-B; do
  wget -O $d.tar.gz "https://zenodo.org/records/15800434/files/$d.tar.gz?download=1"
done
md5sum MOVi-*.tar.gz      # 2c15e4e1d0c8880c5d61ef9b16639168 MOVi-spheres, eabca8332b297c934ddf75fe4f61e998 MOVi-A,
                          # f881545451223965c925a1d1a4f04efb MOVi-B
for d in MOVi-spheres MOVi-A MOVi-B; do tar -xzf $d.tar.gz; done
cd ..
```

Each archive unpacks to `MOVi-<name>/<n>/` (n = 1..1200, the scene's random seed) with a Kubric `metadata.json` (object
shapes, sizes, poses, mass, friction, restitution, camera) and the rendered `rgba.mp4` (256x256, 480 frames at 240 Hz).

### 2. Recover the never-visible objects

HopNet's generator (`data/movi_worker.py`) simulates every sampled object but writes into `metadata.json` only the
ones visible in at least one rendered frame. In 1 MOVi-Sphere and 48 MOVi-B scenes (none in MOVi-A) one or two objects
never come into view and their trajectories are missing; where they strike recorded objects (detected in 1 MOVi-Sphere
and 18 MOVi-B scenes), those change velocity with no visible cause. Each scene is
one deterministic Kubric simulation of seed n with the settings recorded in its metadata, so `replay_movi_scene.py`
rebuilds it (HopNet's generator, its Kubric fork, PyBullet 3.2.6; no rendering) and adds the missing objects. A scene
is accepted only if every recorded object is reproduced exactly (positions, quaternions, velocities and angular
velocities equal in every frame); all 3 x 1,200 scenes are reproduced bit for bit.

The replay runs in its own environment (TensorFlow is only imported by Kubric):

```bash
conda create -n kubric_replay python=3.9 -y && conda activate kubric_replay
pip install numpy==1.26.4 pybullet==3.2.6 pyquaternion traitlets "etils[epath]" absl-py imageio pypng munch bidict \
    pandas trimesh tensorflow-cpu==2.15.1
git clone https://github.com/AmauryWEI/kubric.git && git -C kubric checkout 511ef27 && pip install --no-deps -e kubric
mkdir -p KuBasic && for a in KuBasic.json dome cube cylinder sphere cone torus gear torus_knot sponge spot teapot suzanne; do
  f=$a; [ $a = KuBasic.json ] || f=$a.tar.gz
  wget -q -P KuBasic https://storage.googleapis.com/kubric-public/assets/KuBasic/$f
done
python -c "import json, os; m = json.load(open('KuBasic/KuBasic.json')); m['data_dir'] = os.path.abspath('KuBasic'); \
json.dump(m, open('KuBasic/KuBasic_local.json', 'w'))"

for d in MOVi-spheres MOVi-A MOVi-B; do     # each prints "1200 scenes: 1200 reproduced exactly, 0 not; ..."
  python data_generation/replay_movi_scene.py --raw_root raw/$d --scenes 1-1200 --manifest KuBasic/KuBasic_local.json \
      --report replay_$d.json --write_root replay/$d --workers 32
done
```

### 3. Assemble and convert

With the training environment: assemble the complete raw scenes (links to HopNet's scenes plus the completed ones),
then convert them.

```bash
for d in MOVi-spheres MOVi-A MOVi-B; do
  python data_generation/assemble_complete_raw.py --raw raw/$d --replayed replay/$d --out raw/$d-complete
done
python data_generation/movi_to_sim_data.py --input raw/MOVi-spheres-complete --output datasets/MOVi-Sphere --workers 16
python data_generation/movi_to_sim_data.py --input raw/MOVi-A-complete --output datasets/MOVi-A --workers 16
python data_generation/movi_to_sim_data.py --input raw/MOVi-B-complete --output datasets/MOVi-B --workers 16
```

The geometry comes from the collision meshes in `objects/`; the FPS anchors are deterministic, so the conversion is
reproducible byte for byte. The converter accepts only the Kubric pose keys (`positions`, `quaternions`, one entry per
frame, unit quaternions) and requires mass, friction, restitution and size for every object; an existing output is
reused only when every file it would write is already there and identical.

### 4. Object sizes and splits

`object_scales_{MoviSphere,MoviA,MoviB}.json` and `dataset_split_{MoviSphere,MoviA,MoviB}_1200.json` (repository root)
ship with the code and fit the data converted as above; the loader checks every scene against the size file (folder
name, object count, frame-0 origin and unscaled radius). To rebuild a size file:

```bash
python data_generation/build_object_scales.py --raw raw/MOVi-B-complete --converted datasets/MOVi-B \
    --out data_generation/object_scales_MoviB.json
```

`prepare_dataset_split.py` builds a train / validation / test split over newly generated data.

### 5. Checks

```bash
python data_generation/check_scenes.py --raw raw/MOVi-B-complete --converted datasets/MOVi-B \
    --scales data_generation/object_scales_MoviB.json --split dataset_split_MoviB_1200.json
MOVI_RAW=raw python scripts/verify_all_frames.py --out checks/frames
python scripts/verify_caches.py --out checks/caches        # after the anchor caches exist (README, Training)
MOVI_RAW=raw pytest tests/
```

`check_scenes.py` checks that every scene of a split holds every simulated object and that no recorded object changes
velocity without contact; `verify_all_frames.py` compares every stored frame with the posed asset meshes;
`verify_caches.py` recomputes the anchor caches.

Visual check of a converted scene: `visualize_overlay.py` projects the converted vertices, with the object sizes, into
the scene's Kubric camera and draws each object as a translucent colour over `rgba.mp4`. Every colour must cover
exactly its rendered object at every frame; a wrong size, pose or frame shows as a coloured rim or an uncovered part:

```bash
python data_generation/visualize_overlay.py --raw raw/MOVi-B/7 --converted datasets/MOVi-B/movi_7 \
  --object_scale_file data_generation/object_scales_MoviB.json --frames 0 100 240 479 --out movi_7_overlay.png
# every frame as a video (rendering | overlay):  ... --video movi_7_overlay.mp4
```

## Hugging Face export

`export_hf_dataset.py` writes the Hugging Face dataset from the complete raw scenes (`--raw_root raw`), with
`hf_dataset_card.md` as its card; `scripts/verify_hf_roundtrip.py` compares a materialized copy with the conversion of
the raw scenes, file by file.

## Mesh assets

`objects/<shape>/collision_geometry.obj` holds the collision meshes of the twelve KuBasic shapes (cone, cube, cylinder,
dome, gear, sphere, sponge, spot, suzanne, teapot, torus, torus_knot): Kubric's public assets from
`https://storage.googleapis.com/kubric-public/assets/KuBasic/<shape>.tar.gz` (CC BY-SA 4.0). The converter reads them
via `--assets` (default: this folder's `objects/`). `objects/<shape>/visual_geometry.obj` (with its `.mtl`), from the same
archives, is the render mesh of each object shape (the dome is the scene's backdrop and has none), in the frame of the
collision mesh; the videos and the interactive viewer draw it.
