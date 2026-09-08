#!/bin/bash
# usage: run_blender.sh <shots.json> <list gate|settle> <panel or -1> <gpu> <out_dir> [extra args]
cd "$(dirname "$0")/.."
${BLENDER:-blender} -b --python viz/blender_render.py -- --shots "$1" --list "$2" --panel "$3" --gpu "$4" --out "$5" --width 1600 --height 900 --samples 96 --red-mix 0.45 "${@:6}" 2>&1 | grep --line-buffered "^\[blender\]\|Error\|Traceback" | tee "$5.log"
