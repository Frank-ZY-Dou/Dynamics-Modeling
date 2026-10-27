#!/bin/bash
# A certified scene in MuJoCo, Genesis and Isaac Sim: export it, run each engine for 2 s with its report and pose
# recording, and render the side-by-side clip from the assets and camera of an existing shots.json.
#
#   bash viz/run_engines.sh <repaired scene> <program> <base shots.json> <out dir> <genesis python> <isaac python>
#
# The engines come from examples/install_engines.sh. Isaac Sim starts only once its license is accepted
# (OMNI_KIT_ACCEPT_EULA=YES). BLENDER names a Blender 4.5 binary, PYTHON the Python with SimReady Gate's requirements.
# Writes <out dir>/export, report_<engine>.json and poses_<engine>.json, and <out dir>/clip/engines.gif.
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=$1; PROG=$2; BASE=$3; OUT=$4; GPY=$5; IPY=$6
PY=${PYTHON:-python3}; B=${BLENDER:-blender}
mkdir -p "$OUT"
$PY -m simready.cli export "$SCENE" --out "$OUT/export" --program "$PROG"
$PY examples/run_mujoco.py "$OUT/export" --seconds 2 --record "$OUT/poses_mujoco.json" --report "$OUT/report_mujoco.json"
$GPY examples/run_genesis.py "$OUT/export" --seconds 2 --record "$OUT/poses_genesis.json" --report "$OUT/report_genesis.json"
$IPY examples/run_isaac.py "$OUT/export" --seconds 2 --record "$OUT/poses_isaac.json" --report "$OUT/report_isaac.json"
$PY viz/make_engine_video.py --base "$BASE" --seconds 2 --out "$OUT/clip" \
    --record mujoco="$OUT/poses_mujoco.json" genesis="$OUT/poses_genesis.json" isaac="$OUT/poses_isaac.json"
for k in 0 1 2; do
  $B -b --python viz/blender_render.py -- --shots "$OUT/clip/shots.json" --list settle --panel $k --out "$OUT/clip/blender_settle" \
     --width 960 --height 540 --samples 96 --red-mix 0.45
done
$PY viz/make_engine_video.py --out "$OUT/clip" --assemble
