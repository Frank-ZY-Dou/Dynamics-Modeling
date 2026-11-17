#!/usr/bin/env bash
# Fetch the SAD and D2GV submodules, build the CUDA extensions against the active PyTorch, and
# check the installation. Install the Python packages first (requirements.txt): this script
# installs none. It needs nvcc, a C++ compiler that nvcc supports, ninja, CMake >= 3.26 and
# Python.h. Running it again skips a SAD build that loads.
#
#     PYTHON=/path/to/python bash setup.sh        # default: python, else python3, on PATH
#
# Environment:
#     MAX_JOBS=4            parallel compile jobs (default 4)
#     FORCE_REBUILD=1       rebuild the SAD extension even if the current build loads
#     CXX                   the C++ compiler (default c++)
#     TORCH_EXTENSIONS_DIR  where PyTorch caches JIT builds (default:
#                           .torch_extensions/<python>_<cuda>_<torch>/ in this directory)
#     TORCH_CUDA_ARCH_LIST  GPU architectures of the SAD extension, the exact candidate builder and
#                           gsplat, e.g. "7.5;8.6" (default: those of the visible GPUs; the SAD
#                           build requires it when no GPU is visible)
#     DSAD_FAST_ARCH        the single architecture of the fused step, e.g. 7.5 (default: the
#                           visible GPU's; TORCH_CUDA_ARCH_LIST does not apply to it)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_JOBS="${MAX_JOBS:-4}"

say()  { printf '\n=== %s ===\n' "$*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3 || true)}"
[ -n "$PYTHON_BIN" ] || fail "No python found. Set PYTHON=/path/to/python."
PYTHON_BIN="$(command -v "$PYTHON_BIN")" || fail "cannot run ${PYTHON:-python}. Set PYTHON=/path/to/python."
case "$PYTHON_BIN" in /*) ;; *) PYTHON_BIN="$PWD/$PYTHON_BIN" ;; esac
cd "$REPO"
# ninja and cmake installed next to the interpreter (pip) take precedence.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
# Every package of requirements.txt that is imported (rich and jaxtyping: D2GV's gsplat).
missing=$("$PYTHON_BIN" - <<'PY'
import importlib
bad = []
for m in ("torch", "torchvision", "numpy", "PIL", "yaml", "tqdm", "pytorch_msssim", "rich",
          "jaxtyping"):
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m} ({type(e).__name__})")
print(", ".join(bad))
PY
) || fail "cannot run $PYTHON_BIN"
[ -z "$missing" ] || fail "missing Python packages for $PYTHON_BIN: $missing.
Install requirements.txt first (pip install -r requirements.txt)."
"$PYTHON_BIN" -c "import torch; assert torch.version.cuda" 2>/dev/null \
  || fail "this PyTorch has no CUDA support."
# Build tools of the CUDA extensions (the SAD CMake build and the JIT builds).
command -v ninja >/dev/null || fail "ninja not found (pip install ninja)."
command -v nvcc >/dev/null || fail \
  "nvcc not found. Install the CUDA toolkit and put its bin/ on PATH."
command -v "${CXX:-c++}" >/dev/null || fail \
  "no C++ compiler (${CXX:-c++}) found. Install one that your nvcc supports (e.g. g++), or set CXX."
"$PYTHON_BIN" -c "import os, sysconfig; raise SystemExit(not os.path.isfile(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))" \
  || fail "the Python development headers (Python.h) of $PYTHON_BIN are missing.
Install them (e.g. the python3-dev or python3-devel package; conda environments include them)."
echo "python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPUs visible:', torch.cuda.device_count())"

say "1/4  Submodules"
if [ -d .git ] || git rev-parse --git-dir >/dev/null 2>&1; then
  git submodule update --init --recursive references/SAD baselines/D2GV
else
  echo "not a git checkout; expecting references/SAD and baselines/D2GV to be present"
fi
[ -d references/SAD/backends/pytorch ] || fail "references/SAD is missing. It is a git submodule, fetched only in a git clone of the repository: check 'git -C $REPO status'."
[ -f baselines/D2GV/train_deform.py ] || fail "baselines/D2GV is missing. It is a git submodule, fetched only in a git clone of the repository: check 'git -C $REPO status'."

say "2/4  SAD CUDA extension"
sad_loads() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys, torch
sys.path.insert(0, "references/SAD/backends/pytorch")
import torch_ext.sad_ops  # noqa: F401
assert hasattr(torch.ops.sad, "update_candidates_compact")
PY
}
if [ "${FORCE_REBUILD:-0}" != "1" ] && sad_loads; then
  echo "an existing build loads under this PyTorch; skipping (FORCE_REBUILD=1 rebuilds)"
else
  PYTHON="$PYTHON_BIN" bash scripts/build_sad.sh
fi

say "3/4  JIT extensions (dsad_fast, dsad_exact_candidates, gsplat_cuda)"
# Built once here so that concurrent training jobs only load them. Building needs a visible GPU
# (its architecture). Without one they build on first use.
"$PYTHON_BIN" - <<'PY'
import sys, time
sys.path.insert(0, "dsad")
import _paths
_paths.setup()
import os, torch
print("cache:", os.environ["TORCH_EXTENSIONS_DIR"])
if not torch.cuda.is_available():
    print("no GPU visible: skipped (the extensions build on first use)")
    raise SystemExit(0)
for name, build in (("dsad_fast", lambda: __import__("fast_step").ext()),
                    ("dsad_exact_candidates", lambda: __import__("exact_candidates")._ext()),
                    ("gsplat_cuda", lambda: __import__("gsplat.cuda._backend", fromlist=["_C"])._C)):
    t0 = time.time()
    mod = build()
    print(f"  {name:22s} {'ok' if mod is not None else 'FAILED'}  ({time.time() - t0:.0f} s)  "
          f"{getattr(mod, '__file__', '')}")
    if mod is None:
        raise SystemExit(1)
PY

say "4/4  Verifying"
if "$PYTHON_BIN" -c "import sys, torch; sys.exit(not torch.cuda.is_available())"; then
  "$PYTHON_BIN" scripts/check_install.py
else
  echo "no GPU visible: skipped (run python scripts/check_install.py on a GPU)"
fi

printf '\nSetup complete. Data: data/README.md. Smoke test: bash scripts/smoke_test.sh\n'
