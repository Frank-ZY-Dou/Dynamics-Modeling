"""Step time of MuJoCo Warp on the simplified hand versus world count, eager and as a captured CUDA graph.

    python diffsim/tests/bench_step.py

RTX 2080 Ti, 10 substeps per control step: 16 worlds 2.9 ms, 1040 worlds 4.1 ms, 4160 worlds 6.4 ms with
graphs; about 48 ms eager at any size (launch overhead).
"""
import os
import time

import mujoco
import numpy as np
import warp as wp

wp.init()
import mujoco_warp as mjw  # noqa: E402

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "serial_hand", "models", "AH_Right", "serial_hand.xml")


def main():
    mjm = mujoco.MjModel.from_xml_path(XML)
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mjm.actuator_gainprm[:, :] = 0
    mjm.actuator_biasprm[:, :] = 0
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    for W in (4, 16, 260, 1040, 2080, 4160):
        d = mjw.put_data(mjm, mjd, nworld=W)
        n_sub, reps = 10, 20

        def body():
            for _ in range(n_sub):
                mjw.step(m, d)

        body()
        wp.synchronize()
        t0 = time.time(); body(); wp.synchronize(); eager = time.time() - t0
        with wp.ScopedCapture() as cap:
            body()
        wp.capture_launch(cap.graph); wp.synchronize()
        t0 = time.time()
        for _ in range(reps):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        graph = (time.time() - t0) / reps
        print(f"W={W:5d} worlds, {n_sub} substeps: eager {eager * 1e3:6.1f} ms, CUDA graph {graph * 1e3:5.1f} ms per control step")


if __name__ == "__main__":
    main()
