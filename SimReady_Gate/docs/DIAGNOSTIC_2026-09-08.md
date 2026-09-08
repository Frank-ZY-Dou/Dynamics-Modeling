# Diagnostic, corrected (2026-09-08)

This file supersedes the diagnostic pass of 2026-09-07. A methodological audit of that pass found
several errors that had made the two third-party baselines look worse than they are and that
invalidated some of its claims. The experiments below were re-run after the fixes listed at the end.
Every table here is backed by a tracked file under `results/` (result JSONs and the persisted
layouts are committed).

## Corrections to the earlier pass

- **Wrong fixture for the RoboLab pre-settle replica.** The replica seated objects on `franka_table`
  at the origin; RoboLab's scenegen skill stages them on `table` (table_oak) of `base_empty.usda`
  (top x in [0.20, 0.90], y in [-0.5, 0.5], z = 0). The "RoboLab bounds bug" (x <= 0.80 beyond a
  0.724 m plate), the `within(table.top)` failures, the "orange rolls off the table" G5 verdicts and
  the agent-route conclusions built on them were artifacts of the wrong table and are withdrawn.
- **Narrowed solver bounds.** The replica gave RoboLab's SpatialSolver (0.30, 0.80, -0.40, 0.40);
  the skill passes (0.25, 0.85, -0.45, 0.45). Fixed; success rates below use the skill's bounds.
- **Unseeded layouts.** RoboLab's solver draws from Python's `random`, which nothing seeded, so
  the diagnostic, the G5 sweep and the videos ran on different layouts. Layouts are now generated
  once per (N, seed) with `random.seed` and stored under `results/layouts/`.
- **Inverted front/back.** RoboLab's `front-of` is +x; the DSL had `in_front_of` as -x, so
  relations were "preserved" against the inverted intent and the repair fought it. The DSL now uses
  RoboLab's convention, and the relation gap is the predicate's distance (no hidden 5 cm slack).
- **Wrong RoboCasa sampler.** RoboCasa's `UniformRandomSampler` tests MJCF objects with a
  separating-axis test on the rotated `reg_bbox` corners plus a corner-in-region test, not a
  bounding cylinder. The "half the capacity" claim and the earlier RoboCasa video premise are
  withdrawn; the faithful replica is `experiments/robocasa_sampler.py`.
- **Raw versus repaired was not like for like.** RoboLab's z = dims/2 + 2 mm hovers base-origin
  assets by up to 18 cm; the reported raw peak speeds were largely free falls. The tables now carry a
  `raw, seated` row (same x, y, yaw, dropped onto the table): the state the repair starts from.
- **MJCF geometry.** The hand-written MJCF loader ignored `refquat` and `<default>` classes, so 52 %
  of RoboCasa's objects were rotated and the `reg_bbox` box counted as collision geometry. Objects
  are now compiled by MuJoCo itself (`simready/io/mjcf_io.py`).
- **Claims dropped**: "the 8 single-hull objects are exactly the ones RoboCasa excludes by hand"
  (the excluded lists are commented-out objaverse entries; two names coincide); "layouts carry 1-6
  penetrating pairs" (only the solver-rejected layouts do).

## RoboLab, shipped library (68 scenes, already physics-settled by their pipeline)
`experiments/robolab_shipped_sweep.py` -> `results/robolab_shipped_g2.json`.

- Object-object penetration in 26 / 68 scenes (>= 1 mm in 12; the largest score 9.86 mm in
  `workdesk_snacks`: keyboard/smartphone). Sub-millimeter resting penetration into fixtures in
  64 / 68 (median 0.27 mm). A body wholly inside another body (invisible to a surface test) in 4
  scenes: `cooking_table` (spoon in the plates), `front_of_shelf` (cutlery in the rack),
  `ladle_pot` (fork in a plate), `tools_picking` (clamp in a bin). 30 / 68 scenes have a free body
  that nothing holds from below.
- `max_pen` values are FCL scores, not depths: only the sign is exact.

## RoboLab, pre-settle layouts from its own SpatialSolver (N in {4, 6, 8, 10} x 3 seeds)
Layouts: skill bounds, objects from the catalog, 2 relations per layout, staged in `base_empty.usda`.
`experiments/robolab_diagnostic.py` -> `results/robolab_diag.json`.

| N | solver ok | seated pairs (ok cells) | seated pairs (failed cells) | S4R pen after | RMSD_xy (m) | time (s) |
|---|---|---|---|---|---|---|
| 4 | 3/3 | 0, 0, 0 | - | 0, 0, 0 | 0.031, 0.000, 0.016 | 3.7-4.2 |
| 6 | 1/3 | 0 | 1, 2 | 0, 0, 0 | 0.165, 0.032, 0.018 | 4.3-10.0 |
| 8 | 1/3 | 0 | 3, 2 | 0, 0, 0 | 0.091, 0.079, 0.031 | 5.2-10.6 |
| 10 | 0/3 | - | 7, 2, 7 | 0, 0, 0 | 0.049, 0.040, 0.062 | 5.9-16.5 |

- Layouts the disc solver ACCEPTS carry no mesh penetration once seated (5 / 5 cells). The
  penetration is in the layouts it REJECTS (its fallback is "reduce the object count"): 1-7 pairs.
  S4R turns all 7 rejected layouts into penetration-free ones with the requested relations kept
  (one `within(table.top)` miss: `mayonnaise_bottle` in N=10 s1, placed beyond the table edge).
- RMSD is the planar RMS of the reference-center displacement. Times are per cell in a warm
  process (decimation cached), full-mesh verification included.

## G5 in MuJoCo, RoboLab pre-settle layouts (12 cells, 2 s, hull and CoACD proxies)
Same harness for every state (base scene, slab at the measured table top, floor, proxies, density,
timestep). Pass = no body off the table and peak speed <= 1.0 m/s and peak displacement <= 0.15 m.
`experiments/robolab_g5_sweep.py` -> `results/robolab_g5.json`.

| state | peak speed median (max), hull / CoACD | peak displacement median (max) | cells with a body off the table | pass |
|---|---|---|---|---|
| raw, as the skill writes it (hovering) | 1.52 (4.45) / 1.44 (3.96) m/s | 0.24 (2.01) / 0.20 (0.83) m | 2 / 2 | 1 / 12 |
| raw, seated | 1.01 (3.97) / 0.70 (3.96) m/s | 0.16 (1.08) / 0.11 (0.83) m | 3 / 1 | 4 / 6 of 12 |
| after S4R | 0.41 (1.03) / 0.33 (1.03) m/s | 0.07 (0.12) / 0.05 (0.12) m | 0 / 0 | 8 / 8 of 12 |

- The seated row isolates the repair's effect: the remaining raw motion is penetration-driven
  (`mayonnaise_bottle` at 3.96 m/s leaves the table in N=10 s1; `milkjug_a02` 0.87 m/s in N=6 s0;
  `pomegranate01` 0.53 m/s in N=8 s0) and disappears after S4R.
- The four repaired cells that miss the 1.0 m/s bar (1.03 m/s) are `remote_control` toppling: a
  16 cm tall body on a 3.6 x 2.5 cm footprint, present identically in the seated raw state. That is
  an asset/placement instability, not a repair outcome; G5 reports it per body.
- Round objects drift on MuJoCo mesh contact (0.7-3 cm in 2 s); with the correct table nothing sits
  at an edge, and no repaired cell loses a body.

## RoboCasa, counter-region packing with RoboCasa's own sampler (5 seeds per cell)
`experiments/robocasa_capacity.py` -> `results/robocasa_capacity.json`; geometry = MuJoCo's own
(`mjcf_io.compile_object`).

| region | N=4 | N=6 | N=8 | N=10 | N=12 |
|---|---|---|---|---|---|
| 0.3x0.3 m, RoboCasa sampler | 5/5 | 5/5 | 4/5 | 1/5 | 0/5 |
| 0.3x0.3 m, gate (overlap -> S4R) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 0.4x0.4 m, sampler | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |
| 0.4x0.4 m, gate | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 0.5x0.5 m, sampler | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 0.5x0.5 m, gate | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

- RoboCasa's box test is conservative and reliable: none of its 70 accepted layouts carries mesh
  penetration on the collision geometry, and the collision meshes do not extend below the
  `reg_bbox` bottom (0 of 1594 objects), so its z rule does not sink objects into the counter.
  Its cost appears only in the tight 0.3 m region at N >= 8, where it gives up and the gate still
  places every object (2-41 s per cell).
- G0 over the 1594 AI-generated objects (`results/robocasa_aigen_g0.json`): no visual mesh is
  watertight (so volume-based checks are invalid there), V-HACD pieces median 32 (the cap), and 8
  objects have a single convex piece (boxed_food_2, cereal_5, cutting_board_3, cutting_board_9,
  spaghetti_box_0, spaghetti_box_1, tofu_4, tofu_6): the failure class RoboCasa's authors flagged
  for objaverse assets ("self turning due to single collision geom"); RoboCasa does not exclude
  these eight.

## Fixes the reruns depend on
- Evaluator: contact normals oriented by probing (the center-line heuristic pushed bodies into
  concave fixtures); bodies wholly inside another body counted as penetrating; FCL's coplanar-touch
  artifact resolved by a 0.05 mm probe; `max_pen` documented as a score.
- Seating: the support height is the surface under the body's lowest vertices (downward rays),
  not the highest support vertex (which seated bodies on rims and table legs).
- Repair: rows for penetrating pairs with a vertical FCL normal use the footprint-separation
  direction; support pairs keep rows for non-vertical contacts (walls, rims); a pair the local
  linearization oscillates on for 3 tail iterations is separated along the footprint overlap;
  region rows use exact projection intervals; regions on free reference bodies follow them.
- IO: the USD writer now reproduces the repaired poses (it had silently written the original
  translations); MJCF objects compiled by MuJoCo; unresolved payloads reported.
- DSL/CLI: gate thresholds consumed by `settle`; `check` compiles the program; parser rejects
  single-space statement merges; exit codes 0/1/2; certificates carry provenance (hashes, versions,
  tolerances, the program text).
