# Data

Datasets are not redistributed. Big Buck Bunny ships with the D2GV submodule. UVG and DAVIS are downloaded from their owners and prepared with `scripts/prepare_data.py`, run from the `DSAD/` directory. It needs `ffmpeg`, and `7z` or the `py7zr` package for the UVG archives.

| Dataset | Folder | GoPs | License |
|---|---|---:|---|
| UVG | `data/uvg_proto/uvg720` | 98 | CC BY-NC 3.0 (Digiturk) |
| DAVIS 2017 | `data/davis720` | 53 | CC BY-NC 4.0 |
| Big Buck Bunny | `baselines/D2GV/dataset/bunny` | 13 | CC BY 3.0 (Blender Foundation) |

Each GoP is a folder `<scene>/<scene>_<g>/` of 1280x720 frames `f00001.png`, `f00002.png`, ... and nothing else: the trainer loads every file in it.

## UVG

Seven sequences of <https://ultravideo.fi/dataset.html> in the 1920x1080, 120 fps, 8-bit YUV RAW format (5.3 GB):

```bash
wget https://ultravideo.fi/video/{Beauty,Bosphorus,HoneyBee,Jockey,ReadySetGo,ShakeNDry,YachtRide}_1920x1080_120fps_420_8bit_YUV_RAW.7z
python scripts/prepare_data.py uvg --src <folder with the archives or the extracted .yuv files>
```

As in D2GV-AR, every 4th frame is used (150 of 600, ShakeNDry 75 of 300), in GoPs of 10.

## DAVIS 2017

The TrainVal Full-Resolution archive (3.0 GB) of <https://davischallenge.org/davis2017/code.html>:

```bash
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip
python scripts/prepare_data.py davis --src DAVIS-2017-trainval-Full-Resolution.zip
```

As in D2GV-AR, six scenes are used: `bmx` (bmx-trees, 80 frames), `boat` (75), `camel` (90), `cows` (the first 103), `train` (80) in GoPs of 10, and `swan` (blackswan, 50) in GoPs of 6.

## Verify

```bash
python scripts/prepare_data.py verify               # UVG and DAVIS, 151 GoPs
python scripts/prepare_data.py verify --set bunny
```

`verify` compares the pixels of every GoP with `data/checksums/`. Identical frames need ffmpeg 4.4.2 and Pillow 12; other versions can differ in the last bit of a few pixels. UVG frames are ffmpeg's default yuv420p to RGB conversion, DAVIS frames `ffmpeg -i <frame>.jpg <frame>.png`, and both are resized to 720p with Pillow's bilinear filter.
