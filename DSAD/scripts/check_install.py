#!/usr/bin/env python
"""Check a DSAD installation on tiny inputs (a few seconds on one GPU).

    python scripts/check_install.py

setup.sh runs it as its last step. The exit status is 1 if a check fails. A fused VPT pass that
differs from SAD's is only a warning: the two are bit-identical with nvcc 12.4 on sm_75, and
another compiler or GPU can change that while DSAD still trains and decodes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dsad"))
import _paths  # noqa: E402

_paths.setup()
import math  # noqa: E402

import torch  # noqa: E402

results: list[bool] = []
warned: list[str] = []


class Warn(Exception):
    """A check that does not hold although DSAD works: a warning, not a failure."""


def check(name: str, fn) -> None:
    try:
        msg = fn()
        print(f"  PASS  {name}" + (f"  ({msg})" if msg else ""))
        results.append(True)
    except Warn as w:
        print(f"  WARN  {name}: {w}")
        warned.append(name)
    except Exception as e:  # report every failure, then exit non-zero
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        if "no kernel image" in str(e):
            print("        a CUDA extension was built for other GPU architectures: rebuild where this GPU is "
                  "visible (FORCE_REBUILD=1 bash setup.sh), or with TORCH_CUDA_ARCH_LIST set to its architecture")
        results.append(False)


def cuda():
    assert torch.cuda.is_available(), "no CUDA device visible"
    return torch.cuda.get_device_name(0)


def imports():
    import train_sad_video  # noqa: F401  (imports the SAD backend too)
    import decode  # noqa: F401
    return f"torch {torch.__version__}"


def sad_render_backward():
    from sad_appearance_v3 import SADAppearance_V3
    torch.manual_seed(0)
    m = SADAppearance_V3(num_points=500, H=48, W=64, num_frames=1, device="cuda")
    out = m.forward(frame_idx=0)["render"]
    assert tuple(out.shape) == (1, 3, 48, 64), out.shape
    m.forward_mse_loss(torch.rand(48, 64, 3, device="cuda"), frame_idx=0).backward()
    assert m._features_dc.grad is not None and m._features_dc.grad.abs().sum() > 0
    return "render + backward"


def exact_candidates():
    from exact_candidates import build_candidates, candidates_to_sets
    from sad_appearance_v3 import SADAppearance_V3
    torch.manual_seed(0)
    W, H = 96, 64
    m = SADAppearance_V3(num_points=300, H=H, W=W, num_frames=1, device="cuda")
    with torch.no_grad():
        m._log_aniso.normal_(0, 1)
        m._log_tau.uniform_(2, 6)
        pos = (m.get_xyz + 1.0) * 0.5 * m._wh()
        sites = m._pack_sites_grad(pos, m.get_features).detach()
    e0, e1 = build_candidates(sites, W, H, inv_sq=m.inv_sq)
    b0, b1 = build_candidates(sites, W, H, inv_sq=m.inv_sq, mode="brute")
    same = (candidates_to_sets(e0, e1) == candidates_to_sets(b0, b1)).all(dim=-1)
    assert bool(same.all()), f"{int((~same).sum())} of {same.numel()} pixels differ from brute force"
    # The cooperative rescan (default) against the per-thread one, at a cap that makes
    # many pixels rescan: identical candidates and statistics.
    r = [build_candidates(sites, W, H, inv_sq=m.inv_sq, cap_quantile=0.5, coop_fallback=c, return_info=True)
         for c in (True, False)]
    assert torch.equal(r[0][0], r[1][0]) and torch.equal(r[0][1], r[1][1]) and r[0][2] == r[1][2], \
        "cooperative and per-thread rescans differ"
    return (f"exact == brute force on {same.numel()} pixels; cooperative == per-thread rescan "
            f"({r[0][2]['fallback_pixels']} pixels rescanned)")


def stored_axes():
    import export
    torch.manual_seed(0)
    v = torch.randn(4096, 2, device="cuda") * torch.rand(4096, 1, device="cuda").add_(0.01)
    theta = export.axis_angle(v)
    sin = export.axis_check(v, theta)
    # [0, pi) in float64, rounded to float32 (which can give float32(pi))
    assert bool((theta >= 0).all() and (theta.double() <= math.pi + 1e-6).all()), "angles outside [0, pi]"
    assert sin <= export.AXIS_MAX_SIN, f"rebuilt axes off by |sin| {sin:.2e}"
    return f"axis -> angle -> axis on 4,096 axes: largest |sin| {sin:.1e}"


def fused_step():
    import fast_step
    E = fast_step.ext()
    assert E.rom_supported(256, 12, 5), "anchor top-K 12 with M=256 unsupported"
    return "dsad_fast loaded"


def fused_vpt():
    """The fused step's VPT candidate pass against SAD's update_candidates_compact, bit for bit. The two are
    compiled separately and are bit-identical with nvcc 12.4 on sm_75 (dsad/csrc/dsad_fast.cu). This checks
    the build at hand."""
    r = subprocess.run([sys.executable, str(REPO / "tools" / "check_fast_ops.py"), "--only", "vpt"],
                       cwd=REPO, capture_output=True, text=True)
    out = r.stdout.strip().splitlines()
    if r.returncode != 0 and out and out[-1].startswith("FAILED: "):   # it ran, and the candidates differ
        raise Warn("on this build (compiler, GPU) the fused VPT pass is not bit-identical to SAD's: DSAD trains "
                   "and decodes, but --fast fits do not follow the eager step's candidates bit for bit "
                   "(details: python tools/check_fast_ops.py --only vpt)")
    if r.returncode != 0:
        lines = (r.stdout + r.stderr).strip().splitlines()
        raise AssertionError(lines[-1] if lines else f"tools/check_fast_ops.py --only vpt exited {r.returncode}")
    return "bit for bit with SAD's op (tools/check_fast_ops.py --only vpt)"


def gaussian_backend():
    from gaussianimage_cholesky import GaussianImage_Cholesky
    g = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=200, H=64, W=64,
                               BLOCK_H=16, BLOCK_W=16, device=torch.device("cuda:0"), lr=5e-3,
                               quantize=False).to("cuda")
    out = g.forward()["render"]
    assert tuple(out.shape) == (1, 3, 64, 64), out.shape
    return "gsplat render (DSAD-GS / D2GV)"


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    check("CUDA device", cuda)
    check("DSAD imports", imports)
    if results[0]:
        check("SAD extension", sad_render_backward)
        check("exact candidate builder", exact_candidates)
        check("stored form (axis as an angle)", stored_axes)
        check("fused training step (--fast)", fused_step)
        check("fused VPT pass = SAD's", fused_vpt)
        check("D2GV Gaussian rasterizer", gaussian_backend)
    print(f"{sum(results)} passed, {len(results) - sum(results)} failed"
          + (f", {len(warned)} warning{'s' if len(warned) > 1 else ''} ({', '.join(warned)})" if warned else ""))
    sys.exit(0 if all(results) else 1)
