#!/bin/bash
# usage: run_blender_hq.sh <shots.json> <list> <panel> <gpu> <out_dir> [extra]   (1920x1080, 160 samples)
cd "$(dirname "$0")/.."
${BLENDER:-blender} -b --python viz/blender_render.py -- --shots "$1" --list "$2" --panel "$3" --gpu "$4" --out "$5" --width 1920 --height 1080 --samples 160 --red-mix 0.5 "${@:6}" 2>&1 | grep --line-buffered "^\[blender\]\|Error\|Traceback" | tee -a "$5.log"
