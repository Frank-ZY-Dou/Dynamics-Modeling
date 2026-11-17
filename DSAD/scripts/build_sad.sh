#!/usr/bin/env bash
# Build the SAD CUDA extension (references/SAD) against the active PyTorch and check that it loads.
# setup.sh calls this script.
#
#     PYTHON=/path/to/python MAX_JOBS=4 bash scripts/build_sad.sh
#
# This is SAD's own backends/pytorch/build_ext.sh with bounded build parallelism (MAX_JOBS, default
# 4) and a load check. Rebuild whenever PyTorch changes: a build against another torch fails to
# load with an "undefined symbol" error, and setup.sh then rebuilds. GPU architectures are those of
# the visible GPUs. Set TORCH_CUDA_ARCH_LIST (e.g. "7.5;8.6") to build for others. It is required
# when no GPU is visible.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAD="$REPO/references/SAD"
PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3 || true)}"
JOBS="${MAX_JOBS:-4}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$PYTHON_BIN" ] || fail "No python found. Set PYTHON=/path/to/python."
[ -d "$SAD/backends/pytorch/torch_ext" ] || fail \
  "references/SAD is empty. Run setup.sh first: it fetches the submodule."
# SAD needs CMake >= 3.26. Prefer the one installed next to the interpreter (pip install cmake)
# over an older system CMake.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
command -v cmake >/dev/null || fail "cmake not found (pip install 'cmake>=3.26')."
cmake_ver=$(cmake --version | awk 'NR == 1 {print $3}')
printf '%s\n%s\n' "3.26" "$cmake_ver" | sort -V -C || fail \
  "CMake $cmake_ver is older than 3.26, which SAD requires (pip install 'cmake>=3.26')."
command -v nvcc >/dev/null || fail \
  "nvcc not found. Install the CUDA toolkit and put its bin/ on PATH."
# Without a visible GPU, PyTorch's CMake config falls back to its own architecture list, which
# need not include yours (PyTorch's CUDA 12.8 builds leave out sm_75, the RTX 2080 Ti). The build
# would succeed and every kernel would fail at run time with "no kernel image is available for
# execution on the device".
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && \
   ! "$PYTHON_BIN" -c 'import sys, torch; sys.exit(0 if torch.cuda.device_count() else 1)' 2>/dev/null; then
  fail "No GPU is visible, so its architecture cannot be detected. Build where the GPU is visible, or set TORCH_CUDA_ARCH_LIST (e.g. TORCH_CUDA_ARCH_LIST=7.5 for an RTX 2080 Ti)."
fi

echo "python : $PYTHON_BIN ($("$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__)'))"
echo "cmake  : $cmake_ver"
echo "nvcc   : $(nvcc --version | tail -1)"

BUILD_DIR="$SAD/backends/pytorch/torch_ext/build"
OUT_DIR="$SAD/backends/pytorch/torch_ext/sad_ops"
rm -rf "$BUILD_DIR"
rm -f "$OUT_DIR"/_sad_ext*.so
cmake -S "$SAD/backends/pytorch/torch_ext" -B "$BUILD_DIR" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUT_DIR" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "$JOBS"

# A build can succeed and still leave an extension that does not load or does not register its
# operators, so load it and check the operators.
"$PYTHON_BIN" - "$SAD" <<'PY'
import os, sys
import torch
sys.path.insert(0, os.path.join(sys.argv[1], "backends", "pytorch"))
import torch_ext.sad_ops  # noqa: F401
need = ("jfa_seed", "jfa_flood", "pack_candidate_sites", "update_candidates_compact",
        "render_sad", "render_sad_backward")
missing = [op for op in need if not hasattr(torch.ops.sad, op)]
if missing:
    raise SystemExit(f"FAILED: the extension loads but these operators are missing: {missing}")
from modules import SADRenderer, CandidateUpdater  # noqa: F401
print("OK: torch.ops.sad registered, SAD modules importable")
PY
