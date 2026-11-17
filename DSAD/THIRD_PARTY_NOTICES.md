# Third-party notices

The DSAD code is released under the MIT License (see `LICENSE`). It builds on
the third-party work below. The datasets and the terms of the released models
follow the code.

## SAD: Soft Anisotropic Diagrams (MIT)

<https://github.com/LuckyIYI/SAD>

- **As a dependency.** SAD is an unmodified git submodule (`references/SAD`,
  pinned at commit `361b382`). The parts of SAD's code that DSAD contains or
  follows are listed below.
- **Build script.** `scripts/build_sad.sh` is SAD's
  `backends/pytorch/build_ext.sh` with bounded build parallelism and a load
  check.
- **Code from SAD in DSAD.** Two blocks of `dsad/csrc/dsad_fast.cu` contain
  SAD's device code:
  - the block that starts with the comment "Render + clamp/MSE + render
    backward in one pass" (namespace `sadref`):
    - copied: the `Site` layout and the helpers `dot`, `site_color`,
      `site_aniso_dir` and `voronoi_dmix2` (`backends/cuda/src/sad_common.cuh`),
      and the per-candidate body of `render_sad_backward_kernel`
      (`backends/pytorch/torch_ext/cuda/sad_bindings.cu`) as
      `candidate_grad_terms`;
    - merged: `render_weights` computes the color of `renderVoronoiKernel`
      (`backends/cuda/src/render.cu`) and the weights and prediction of
      `compute_candidate_weights_raw` (`sad_bindings.cu`) in one pass over the
      candidates, with their expressions and each function's validity rules, so
      that every value equals the one SAD's function returns; `render_color`,
      which the decoder's render uses, is the color part of `render_weights`;
    - changed: the accumulation of the gradients (summed within a warp first,
      then one atomic addition per value).
  - the block that starts with the comment "VPT candidate update" (`k_vpt_prep`,
    `k_vpt_update` and their helpers) reimplements SAD's `updateCandidatesKernel`,
    `candidatePackedDistance` and `insertClosest8Packed`
    (`backends/cuda/src/jfa.cu`) and `packCandidateSitesKernel`
    (`backends/cuda/src/candidate_pack.cu`), with the same expressions and the
    same pseudo-random injection, so that a pass gives the candidates of SAD's
    `update_candidates_compact` bit for bit; it computes the key's per-site
    constants once per pass and inserts into the sorted list with a
    branch-free network.
- **Code derived from SAD.** `dsad/csrc/exact_candidates_cuda.cu` evaluates
  SAD's candidate key with the expression sequence of SAD's `voronoi_dmix2` and
  `candidatePackedDistance` (`backends/cuda/src/jfa.cu`), so that its rankings
  match SAD's bit for bit. `_jump_for_step` in `dsad/sad_appearance_v3.py` is
  adapted from `_jump_step_for_index` (`backends/pytorch/train.py`), the jump
  distance of a VPT pass; it cycles through the distances where SAD's holds
  the last.
- **SAD's site layout.** The stored form of a GoP (`dsad/export.py`) keeps SAD's
  per-site fields with the axis as one angle; DSAD's decoders rebuild SAD's
  10-field sites from it. The eager decoder, which scores a fit, renders them
  with SAD's unmodified `render_sad`; the fused one (`decode.py`'s default for
  the decode rate) with `render_decode` of `dsad/csrc/dsad_fast.cu`, whose
  color is `render_color`, above.

SAD's license:

```
MIT License

Copyright (c) 2026 LuckyIYI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## D2GV

<https://github.com/Evan-sudo/D2GV>

D2GV is an unmodified git submodule (`baselines/D2GV`, pinned at commit
`7806b3b`). Its repository carries no license file, so no D2GV file or module
is included here: DSAD imports D2GV from the submodule. What runs of D2GV's
code:

- **the full method** (SAD appearance): one D2GV file, the Adan optimizer of
  the sites (`optimizer.py`, which carries its own Apache License 2.0 notice;
  see Adan below). Its training log is written by DSAD's own `LogWriter`
  (`dsad/train_sad_video.py`), and nothing else of D2GV is imported on this
  path;
- **the DSAD-GS ablation** (Gaussian appearance, `configs/dsad_gs.yaml`): also
  D2GV's Gaussian model (`gaussianimage_cholesky.py`, rasterized by its gsplat
  fork) and its loss (`utils.py`, `loss_fn`), imported only on that path.

Two parts of DSAD follow D2GV's code. `dsad/train_sad_video.py` follows the
two-stage training loop of D2GV's `train_deform.py` (the trainer class, the
frame queue, the optimizer steps and the seeding), which D2GV adapted from
GaussianImage's `train.py` (MIT, below). `_expon_lr` re-implements the
learning-rate decay that D2GV uses (see the learning-rate schedule below).
D2GV contains a fork of [gsplat](https://github.com/nerfstudio-project/gsplat)
(Apache-2.0) with [GLM](https://github.com/g-truc/glm), the Adan optimizer
(`optimizer.py`, see below), and the Bunny frames used in the paper (from
*Big Buck Bunny*, © Blender Foundation, CC BY 3.0); these are covered by their
own licenses.

## GaussianImage (MIT)

<https://github.com/Xinjie-Q/GaussianImage>

D2GV builds on GaussianImage: its Gaussian model (`gaussianimage_cholesky.py`),
its gsplat fork and the skeleton of its training script. Through D2GV, the
training loop of `dsad/train_sad_video.py` follows GaussianImage's `train.py`.
GaussianImage's license:

```
MIT License

Copyright (c) 2024 Xinjie Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Adan (Apache-2.0)

<https://github.com/sail-sg/Adan>, X. Xie et al., "Adan: Adaptive Nesterov
Momentum Algorithm for Faster Optimizing Deep Models".

DSAD optimizes the SAD sites with Adan, imported from the D2GV submodule's
`optimizer.py` (Copyright 2022 Garena Online Private Limited, Apache License
2.0, whose text is at the end of this file). For the fused training step,
`dsad/csrc/dsad_fast.cu` (`k_adan`) and `dsad/fast_step.py` (`_adan_step`)
implement the same update rule again, following the arithmetic of that file's
multi-tensor path step by step so that the updates match it bit for bit. They
are new code (one CUDA kernel and one PyTorch function; no text of the
file is copied), but since they follow its arithmetic they are treated as
derived from it: that file's copyright and license notice above, and the
Apache License 2.0 at the end of this file, apply to these two functions.

## Adam: PyTorch (BSD-3-Clause)

<https://github.com/pytorch/pytorch>. The fused training step updates the ROM's
parameters with Adam: `dsad/csrc/dsad_fast.cu` (`k_adam`) and
`dsad/fast_step.py` (`_adam_step`) implement `torch.optim.Adam`'s update again,
following the arithmetic of its multi-tensor path step by step so that the
updates match it bit for bit. They are new code (no text of PyTorch is
copied), but since they follow its arithmetic they are treated as derived from
it: PyTorch's copyright notices and BSD-3-Clause license, reproduced at the end
of this file, apply to these two functions.

## Learning-rate schedule: Plenoxels (BSD 2-Clause) and JaxNeRF (Apache-2.0)

`_expon_lr` in `dsad/deformation_core.py` re-implements, with Python's `math`
in place of numpy, the exponential learning-rate decay of D2GV's
`utils.get_expon_lr_func`. D2GV marks that function as copied from Plenoxels,
whose `get_expon_lr_func` (<https://github.com/sxyu/svox2>, `opt/util/util.py`)
is in turn adapted from JaxNeRF's `learning_rate_decay`
(<https://github.com/google-research/google-research>, `jaxnerf/nerf/utils.py`,
Copyright The Google Research Authors, licensed under the Apache License,
Version 2.0, whose text is at the end of this file).

Plenoxels' license:

```
BSD 2-Clause License

Copyright (c) 2021, the Plenoxels authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Datasets

No video is distributed with DSAD; each dataset is downloaded from its
official source under its own terms (`data/README.md`):

| dataset | used for | license / terms |
|---|---|---|
| [UVG](https://ultravideo.fi/dataset.html) (Mercat et al., MMSys 2020) | UVG-Small and UVG-Large (Tab. 2) | CC BY-NC 3.0 Unported, under which Digiturk, the owner of the sequences, licenses them (notice below) |
| *Big Buck Bunny* (© Blender Foundation), the 13 GoPs in the D2GV repository | quick start, smoke test, Bunny benchmarks | CC BY 3.0 |
| [DAVIS 2017](https://davischallenge.org/) (Perazzi et al., CVPR 2016; Pont-Tuset et al., 2017) | DAVIS-Small and DAVIS-Large (Tab. 2) | CC BY-NC 4.0, per the Terms of Use in the dataset archive's README (`data/README.md`) |

`data/` holds SHA-256 checksums of the pixels of the paper's frames (no frame
data), to check a local preparation. The figures in `assets/` show frames of
the UVG sequences Jockey and ReadySetGo (UVG's terms) and of Big Buck Bunny
(`bunny.gif`, `bunny.mp4`: CC BY 3.0, (c) Blender Foundation).

**UVG.** The sequences and all intellectual property rights in them are
Digiturk's, which licenses them under CC BY-NC 3.0 Unported. The notice that
ships with each sequence reads, verbatim:

```
This sequence and all intellectual property rights therein remain the property of Digiturk.
You may use this sequence according to Creative Commons Attribution-NonCommercial 3.0 Unported (http://creativecommons.org/licenses/by-nc/3.0/deed.en_US).
```

Please cite A. Mercat, M. Viitanen and J. Vanne, "UVG dataset: 50/120fps 4K
sequences for video codec analysis and development", ACM MMSys 2020.

## Models we release

The released models are the stored parameters of DSAD fits. They are derived
from the videos they were fitted to, and each keeps that video's terms:

| archive | holds | terms |
|---|---|---|
| `dsad-records-v1` | scores, training logs and command lines of the fits; no model | MIT, as the code |
| `dsad-models-uvg-v1` | the stored parameters of the fits of UVG GoPs | CC BY-NC 3.0 Unported (owner and licensor of the sequences: Digiturk): non-commercial use, with attribution |
| `dsad-models-bunny-v1` | the stored parameters of the fits of Big Buck Bunny GoPs | CC BY 3.0, with attribution to the Blender Foundation |

The fits of DAVIS videos are not distributed; `dsad-records-v1` holds their
scores, training logs and command lines.

## Apache License 2.0

The license of JaxNeRF (above), Adan and gsplat:

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## The license of PyTorch (for `k_adam` and `_adam_step`)

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
