#!/bin/bash
# The engines the runners in this folder load, each in its own virtual environment, at the versions the README reports:
#
#   bash examples/install_engines.sh <python3.11> <envs dir>
#
#   <envs dir>/genesis    Genesis 1.2.3, with PyTorch 2.9.1 for CUDA 12.8      ->  <envs dir>/genesis/bin/python run_genesis.py
#   <envs dir>/isaacsim   Isaac Sim 5.1.0 from NVIDIA's package index           ->  <envs dir>/isaacsim/bin/python run_isaac.py
#
# MuJoCo is one of SimReady Gate's own requirements. Isaac Sim asks on its first start to accept the NVIDIA Omniverse
# license agreement; OMNI_KIT_ACCEPT_EULA=YES accepts it.
set -euo pipefail
PY=$1; ENVS=$2
mkdir -p "$ENVS"
"$PY" -m venv "$ENVS/genesis"
"$ENVS/genesis/bin/python" -m pip install --upgrade pip
"$ENVS/genesis/bin/python" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
"$ENVS/genesis/bin/python" -m pip install genesis-world==1.2.3
"$PY" -m venv "$ENVS/isaacsim"
"$ENVS/isaacsim/bin/python" -m pip install --upgrade pip
"$ENVS/isaacsim/bin/python" -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
"$ENVS/genesis/bin/python" -c "import genesis, torch; print('genesis', genesis.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$ENVS/isaacsim/bin/python" -c "from importlib.metadata import version; print('isaacsim', version('isaacsim'))"
