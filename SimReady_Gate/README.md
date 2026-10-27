<div align="center">
<h1>SimReady Gate: Language-Driven, Verifier-in-the-Loop Repair of Generated Scenes</h1>

<a href="../Penetration_Solving/"><img src="https://img.shields.io/badge/Built_on-S4R_(SIGGRAPH_Asia_2026)-green" alt="Built on S4R"></a>
<a href="docs/DSL.md"><img src="https://img.shields.io/badge/Language-DSL.md-0b6fae" alt="The language"></a>
<a href="docs/EVAL_DSL_vs_direct.md"><img src="https://img.shields.io/badge/Comparison-program_vs_poses-orange" alt="The comparison"></a>
<a href="results/"><img src="https://img.shields.io/badge/Results-tracked_JSON-orange" alt="Results"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT"></a>

**Built on [S4R](../Penetration_Solving/): Scaling for Rigid-Body Interpenetration Resolution, ACM Transactions on Graphics (SIGGRAPH Asia 2026)**

Part of [Dynamics-Modeling](https://github.com/Frank-ZY-Dou/Dynamics-Modeling), research code and projects by [Frank Zhiyang Dou](https://frank-zy-dou.github.io/) from [MIT CDFG](https://cdfg.mit.edu/people/#current)
</div>

<table>
<tr>
<td width="50%"><img src="docs/media/pile_n20_gate.gif" width="100%" alt="twenty RoboLab objects heaped on a table, shrunk, laid out by the model in scale-space and restored to full size"/></td>
<td width="50%"><img src="docs/media/robocasa_ring_gate.gif" width="100%" alt="ten AI-generated RoboCasa objects in a heap, laid out on a circle"/></td>
</tr>
<tr>
<td width="50%"><img src="docs/media/shelf_books_gate.gif" width="100%" alt="books and notebooks heaped on a bookcase's shelves, then shelved by the program"/></td>
<td width="50%"><img src="docs/media/shelf_pantry_gate.gif" width="100%" alt="kitchen things heaped on a rack, then stocked by the program"/></td>
</tr>
</table>
<p align="center"><i>Generated scenes laid out and repaired from language: a heap of twenty RoboLab objects set out for cooking, ten RoboCasa objects on a ring, books shelved by height, and a rack stocked with heavy things low and cans in a grid.</i></p>

---

## 🔥 News

- **[2026-10]** Robot scenes and a written semantics: SimReady and [Scenario](https://huggingface.co/datasets/scenario-labs/3d-collection) assets (CC BY 4.0, credit: Scenario), shelves and rings; a failed requirement comes with its nearest relaxation ([the language](docs/DSL.md#what-a-program-means)). Programs pass 141 / 144, typed poses 128 / 144 ([results](#a-program-or-the-poses)).
- **[2026-09]** Into the engines: `export` writes a repaired scene for MuJoCo, Genesis and Isaac Sim, and the twenty-object layout runs in all three with no body leaving the table ([details](#-into-the-engines)).
- **[2026-09]** Twenty-object demo: a heap of twenty RoboLab objects laid out by language in scale-space, with every program, tool output and certificate in [`docs/examples/pile_n20/`](docs/examples/pile_n20/).
- **[2026-09]** Scale-space placement: `place(a, x, y, yaw)` positions bodies at the shrink scale, and the continuation restores full scale under the program.
- **[2026-09]** Initial release: RoboLab and RoboCasa scenes, the constraint language and its compiler, the S4R repair, the mesh-level evaluator, the MuJoCo settle test, certificates and the agent skill, with the diagnostics in [`docs/DIAGNOSTIC_2026-09-08.md`](docs/DIAGNOSTIC_2026-09-08.md).

## 📖 Overview

Generated scenes reach a simulator with mesh-level interpenetration and with bodies that nothing supports. Settled as it is, the twenty-object heap above reaches 3.8 m/s in MuJoCo and throws a body off the table.

**SimReady Gate** is the layer between a scene generator and the simulator: a request becomes a typed constraint program, S4R repairs the scene under it in scale-space, and a mesh-level evaluator and a MuJoCo settle certify the result.

- **Reads the scenes robotics pipelines produce.** RoboLab USD scenes through `usd-core` (no Isaac Sim needed), RoboCasa MJCF objects compiled by MuJoCo itself, and NVIDIA SimReady packages with their mass, friction and semantic class.
- **Measures on the meshes, not on proxies.** An FCL evaluator with probed contact normals, a containment test and a resting-contact test.
- **Repairs with S4R under a program.** Bodies shrink about their reference centers and grow back through minimum-norm QPs, upright on their support, under the program's relations, regions and the statements robot scenes ask for ([the language](#-the-language)).
- **Lays scenes out in scale-space.** At the shrink scale a language model places bodies anywhere (`place`); restoring the scale resolves what still overlaps.
- **Certifies by settling.** MuJoCo with the proxies an engine would use (convex hulls, CoACD pieces), against the program's own thresholds, and a certificate with provenance.
- **Runs as an agent loop.** An agent following [the skill](skills/simready-scenegen/SKILL.md) writes and edits the program and acts on exit codes; it never overrides a number.

On sixteen SimReady tasks, programs pass 141 / 144 scenes and typed poses 128 / 144, with 1.8 to 3.9 times fewer output tokens per trial ([results](#a-program-or-the-poses)).

<p align="center"><img src="docs/media/scale_space_editing.png" width="420" alt="scale-space editing: two bodies overlap; S4R shrinks them; the model moves the shrunken bodies; the scale is restored"/></p>
<p align="center"><i>Scale-space editing. (1) Two bodies overlap. (2) S4R shrinks every body about its reference center. (3) The model moves the shrunken bodies into the requested arrangement. (4) The scale is restored under the program, penetration-free.</i></p>

## 🛠️ Environment Setup

Python 3.10 or newer:

```bash
pip install numpy scipy trimesh rtree fast-simplification python-fcl osqp mujoco coacd usd-core
python -m unittest discover tests        # regression tests
```

- `rtree` runs trimesh's ray and point-in-mesh queries. A query that cannot be evaluated is an error (exit code 2), never "no hit".
- `fast-simplification` decimates the bodies: the continuation runs on 2000-face copies and verifies on the full meshes. Without it the repair warns and runs on the full meshes, several times slower.
- Optional: `anthropic` for the SDK backend, which reads `ANTHROPIC_API_KEY` from the environment only (nothing in this repository holds a credential); Blender 4.5 for the renders in `viz/` (`BLENDER=/path/to/blender`); Genesis 1.2.3 and Isaac Sim 5.1.0 for the [engines](#-into-the-engines), each in its own environment (`bash examples/install_engines.sh <python3.11> <envs dir>`).

## 📦 Assets

**NVIDIA SimReady.** Search through [SimReady Central](https://huggingface.co/spaces/nvidia/simready-central) (text and SigLIP2 image search) and fetch from [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets), every file checked against the package's bill of materials and the dataset commit recorded:

```bash
python -m simready.cli assets search "coffee mug" --max-size 0.2    # packages with their size, mass and license
python -m simready.cli assets fetch dishware_mug_coffee_a01         # checked against the bill of materials (textures with --textures)
python -m simready.cli assets info dishware_mug_coffee_a01          # what the gate reads
```

The gate reads each package's mesh, mass, centre of mass and inertia, friction, restitution and density, semantic class and grasp hint (the side of a mug's handle, for `facing`). The settle test and the export use each body's own friction and mass, and `balanced_on` its centre of mass. Fetched files live under `SIMREADY_ASSETS_DIR`; a repair never downloads. A layout names packages directly:

```json
{"table": {"package": "table_metal_stainless_a03", "position": [0.548, 0.0, -0.055], "yaw": 90},
 "objects": [{"name": "mug", "package": "dishware_mug_coffee_a01", "x": 0.517, "y": 0.110, "z": 0.0, "yaw": 30}]}
```

A layout can also:

- add fixtures: `"fixtures": [{"name": "case", "package": "cabinet_file_d02", "position": [0.65, 0, 0], "yaw": 180}]`;
- stand an object on another side: `"up": "+y"` (a book on its edge);
- name the side that faces out: `"front": "-x"` (its spine), for `axis=front`.

`--repo` fetches from the other datasets SimReady Central indexes; the shelf examples also use [PalatialSim/palatial-nv-asset-library](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library) (CC BY 4.0, credit: PalatialSim).

**Scenario 3D Collection.** [`simready/data/scenario_assets.json`](simready/data/scenario_assets.json) lists 37 tabletop objects chosen from [scenario-labs/3d-collection](https://huggingface.co/datasets/scenario-labs/3d-collection) (1,342 AI-generated assets, CC BY 4.0, credit: Scenario), pinned to commit `55b4c002`. For each it gives the GLB and its sha256, the class, one real dimension, the mass, friction and restitution, the side of the handle and whether things fit inside. The meshes carry no physics and no real size, so these values are estimates per kind of object, not measurements.

```bash
python -m simready.cli assets list scenario
python -m simready.cli assets fetch scenario:mug_white    # sha256 checked, +y up turned to +z, scaled to its dimension
python -m simready.cli assets info scenario:mug_white
```

A fetched asset has its origin at the centre of its footprint on its lowest point; a mesh above 40,000 faces is decimated, and the record keeps how far the result lies from the original (at most 4.9 mm). A layout names it with `"scenario": "mug_white"`. The list keeps single rigid objects a robot meets on a table, and leaves out characters, scenes, objects in many loose pieces or with a separate part beside them, and Gaussian splats.

**RoboLab and RoboCasa.** Not redistributed here. Set `ROBOLAB_DIR` to a checkout of [NVLabs/RoboLab](https://github.com/NVLabs/RoboLab) (with its LFS assets pulled) and `ROBOCASA_AIGEN_DIR` to RoboCasa's `aigen_objs` folder.

<details>
<summary><b>Assets authored in a pose they cannot rest in</b></summary>

RoboLab's remote control is authored standing on its 3.6 × 2.5 cm end, 16 cm tall, a pose that tips at 8.7°; every shipped scene that contains it has it lying on its back. [`simready/data/asset_rest_orientations.json`](simready/data/asset_rest_orientations.json) lists the catalog assets whose authored pose cannot stand (tipping angle below 12°, mass centre at the box centre), with the authored axis that points up in the shipped scenes. The loaders re-express those assets in that resting frame: the vertices turn once, the pose rotation absorbs the inverse, and the world geometry is unchanged. A layout that sets only a yaw then lays the remote flat, `upright` keeps it flat, the placement solver sees its resting footprint, and the export writes it lying. A SimReady package authored in such a pose (a cutting board on its edge) is re-expressed in its most stable resting frame by the same rule.

`experiments/asset_rest_orientations.py` regenerates the table from the catalog and the shipped scenes. Assets that can stand keep their authored frame, because the shipped scenes are physics-settled heaps in which cans and cartons also lie knocked over.

</details>

<details>
<summary><b>Repository layout</b></summary>

```
SimReady_Gate/
├── simready/
│   ├── scene/          # Body, Scene, mesh proxies, surface-based seating
│   ├── gates/          # verify.py (G2), clearance.py (G2 min_gap), settle_mujoco.py (G5)
│   ├── repair/         # upright_s4r.py: S4R upright-on-plane continuation with program rows
│   ├── dsl/            # model.py (parser), schema.py (JSON schema, prompt), compile.py, statements.py (the v1 statements), text2dsl.py, anthropic_backend.py
│   ├── io/             # usd_io.py (RoboLab USD), mjcf_io.py (RoboCasa MJCF via MuJoCo), export.py (meshes, manifest, MJCF, USD),
│   │                   # nvidia_assets.py (SimReady packages), scenario_assets.py (Scenario 3D Collection)
│   ├── data/           # asset_rest_orientations.json, scenario_assets.json
│   ├── errors.py       # GeometryQueryError: a query that could not be evaluated is an error, not a result
│   └── cli.py          # summarize | prompt | check | verify | repair | settle | export | assets
├── examples/           # run_mujoco.py, run_genesis.py, run_isaac.py: load an export and simulate it; install_engines.sh
├── tests/              # regression tests (python -m unittest discover tests)
├── skills/             # the agent skill that runs the loop
├── experiments/        # the diagnostics (RoboLab, RoboCasa, settle); shelf_layouts.py;
│                       # dsl_vs_direct/: the program-or-poses comparison (tasks, rubrics, isolated CLI calls, leak test, tables)
├── viz/                # Blender renders on the solver's meshes, video drivers, the engine clip
├── results/            # tracked result tables and the persisted layouts
└── docs/               # the language (DSL.md), the comparison (EVAL_DSL_vs_direct.md), diagnostics, media, recorded examples
```

</details>

## 🚀 Quick Start

A scene goes through five commands; an agent runs the same ones ([the agent loop](#-the-agent-loop)):

```bash
S="$ROBOLAB_DIR/assets/scenes/workdesk_snacks.usda"
python -m simready.cli summarize "$S"                                  # bodies, sizes, tags, support rectangles
python -m simready.cli prompt "$S" --request "..."                     # the text-to-program prompt with the schema
python -m simready.cli check "$S" program.json                         # compile the program as the repair will
python -m simready.cli repair "$S" program.json --out repaired.usda    # S4R under the program, verification, write-back, certificate
python -m simready.cli settle repaired.usda --program program.json     # the MuJoCo settle test, thresholds from the program
```

Exit code 0 means the outcome holds, 1 that it does not, and 2 that the program or an input is invalid or a geometric query could not be evaluated. A gate value that cannot be parsed, a coordinate that is not a finite number, a statement with the wrong number of arguments: all are exit 2, none is silently dropped.

Scenes are RoboLab `.usda` files (metres, Z up; a stage that says otherwise is refused) or layout JSONs (`base_scene` + objects); RoboCasa objects come in through `simready.io.mjcf_io` (see `experiments/robocasa_capacity.py`).

<details>
<summary><b>Commands</b></summary>

| command | does | exit 0 means |
|---|---|---|
| `summarize <scene>` | body list, sizes, tags, support rectangles | — |
| `prompt <scene> --request "..."` | the text-to-program prompt with the schema | — |
| `check <scene> program.json` | validate, parse and compile the program | the program is usable |
| `verify <scene>` | the mesh-level evaluator on the scene as is | no penetrating pair |
| `repair <scene> program.json --out <file>` | S4R under the program, full-mesh verification, write-back, certificate | pen 0, every predicate holds and, with a `G2: min_gap` gate, every pair keeps that clearance |
| `settle <scene> --program program.json` | the MuJoCo settle test, thresholds from the program's gate | the scene stays at rest (with a `G2: min_gap` gate, the scene as given also keeps that clearance) |
| `export <scene> --out <dir>` | meshes, manifest, MJCF and USD for MuJoCo, Genesis and Isaac Sim | the files were written |
| `assets search\|fetch\|info\|list ...` | find SimReady packages, fetch them checked, print their physics | the packages were found or fetched |

Every command is `python -m simready.cli <command>`.

</details>

<details>
<summary><b>Gates</b></summary>

| gate | question | tool |
|---|---|---|
| G0 | is the asset admissible: collision and visual geom counts, a watertight merged collision mesh, collision extents against `reg_bbox` | `mjcf_object_stats` |
| G2 | does any pair interpenetrate, is any body inside another, is any body unsupported, and (with `min_gap`) does every pair other than a body and its declared support keep the clearance? | FCL score with probed normals, containment and resting-contact tests (`simready.gates.verify`); pair-by-pair clearance (`simready.gates.clearance`) |
| G3 | the repair: S4R scale continuation under the program | `simready.repair.upright_s4r` |
| G5 | does the scene stay at rest in a physics engine? | MuJoCo settle with hull and CoACD proxies (`python -m simready.cli settle`) |

</details>

## 🧪 Examples

### Example 1: a scene RoboLab ships

`workdesk_snacks.usda` is one of RoboLab's library scenes. It has been through RoboLab's physics settle and still carries three object-object interpenetrations, the largest 9.9 mm (keyboard against smartphone). The request: *"Keep everything on the table, the keyboard in front of the smartphone with at least 5 cm between them, and nothing interpenetrating."*

<p align="center"><img src="docs/media/workdesk_strip.png" width="100%" alt="raw layout (penetrating bodies tinted), scale continuation at s = 0.05 and s = 0.69, repaired"/></p>
<p align="center"><i>As shipped, with the interpenetrating bodies tinted; S4R shrinking every free body about its reference center and growing it back while the QP moves bodies in the plane; repaired.</i></p>

**1. Summarize.** The agent reads this; it never looks at geometry itself.

```
$ python -m simready.cli summarize "$ROBOLAB_DIR/assets/scenes/workdesk_snacks.usda"
- table: FIXED fixture,support; size 0.70x1.00x0.70 m; at (0.55, -0.00, -0.35); top rect x[0.20, 0.90] y[-0.50, 0.50] at z=0.003
- franka_table: FIXED fixture,support; size 0.90x0.76x0.79 m; at (-0.36, 0.00, -0.40); ...
- ceramic_mug: object; size 0.10x0.13x0.08 m; at (0.56, 0.40, 0.04)
- glasses: object; size 0.20x0.20x0.05 m; at (0.39, -0.10, 0.02)
- keyboard: object; size 0.14x0.43x0.02 m; at (0.57, 0.00, 0.01)
...
```

**2. Text to program.** The model answers the prompt of `python -m simready.cli prompt` with one JSON object (schema-constrained; a statement can carry the words that justify it):

```json
{"statements": [
  {"op": "no_penetration", "reason": "no interpenetration"},
  {"op": "fixed", "a": "table"}, {"op": "fixed", "a": "franka_table"}, {"op": "fixed", "a": "GroundPlane"},
  {"op": "on_support", "a": "*", "b": "table", "reason": "everything stays on the table"},
  {"op": "upright", "a": "*"},
  {"op": "within", "a": "*", "b": "table.top", "inset": 0.0},
  {"op": "farther", "a": "keyboard", "b": "smartphone", "gap": 0.05, "reason": "keyboard in front of the phone, 5 cm"},
  {"op": "minimize"}],
 "gate": {"settle_v_max": 1.0, "settle_dx": 0.15},
 "assumptions": ["'in front of' read in RoboLab's frame: +x, away from the robot"]}
```

**3. Check.** `check` compiles the program exactly as the repair will; unknown bodies, self-relations and a region with no room stop here with exit code 2.

```
$ python -m simready.cli check "$ROBOLAB_DIR/assets/scenes/workdesk_snacks.usda" program.json
{"ok": true, "statements": 9, "rows": 27, "predicates": 40, ...}
```

**4. Repair.** S4R runs on decimated proxies, verifies on the full meshes, checks every predicate, writes the poses into a copy of the USD and reads it back. `pen_before` counts every negative pair, including the sub-millimetre resting contacts with the table that PhysX leaves; three of the twelve are object-object.

```
$ python -m simready.cli repair "$ROBOLAB_DIR/assets/scenes/workdesk_snacks.usda" program.json --out workdesk_repaired.usda
{"ok": true, "pen_before": 12, "pen_after": 0, "rmsd_xy": 0.037, "time_s": 3.5,
 "failed_predicates": [], "written_pose_error_m": 2.1e-05,
 "certificate": "workdesk_repaired.certificate.json"}
```

**5. Settle.** MuJoCo, once with convex hulls and once with CoACD pieces; the thresholds come from the program's `gate`:

```
$ python -m simready.cli settle workdesk_repaired.usda --program program.json
{"pass": true, "thresholds": {"v_max": 1.0, "d_max": 0.15, "source": "program gate"},
 "hull":  {"peak_speed": 0.351, "peak_disp": 0.0144, "left_support": [], "pass": true},
 "coacd": {"peak_speed": 0.256, "peak_disp": 0.0134, "left_support": [], "pass": true}}
```

<p align="center"><img src="docs/media/workdesk_settle.png" width="100%" alt="MuJoCo settle, as shipped versus repaired"/></p>

**6. Certificate.** `workdesk_repaired.certificate.json` holds the program text and JSON, the hashes of the scene read, the scene written and every asset file, the git commit, library versions, the schedule, the tolerances, every predicate value and the settle report.

<details>
<summary><b>When a certificate says <code>ready</code></b></summary>

- `settle` adds its report to the certificate only when the scene, the program and every asset hash to what the repair used, and then sets `ready`: the repair's `ok` and the settle's `pass`. Otherwise the report goes to a separate `.settle.json`.
- The settle report also evaluates the program's predicates on the settled poses (`after_settle`, at settle tolerances: 1 cm, 15° of tilt, 10 cm for a placement) and gives the final tilt of every body.
- With `--hold-predicates` those predicates must hold for the settle to pass; the certificate records the policy.

</details>

### Example 2: twenty objects from a heap

Twenty catalog objects are dropped into a 13 cm radius on RoboLab's `table_oak` (seed 2 of `viz/make_robolab_scene_video.py --pile`): 82 object-object interpenetrating pairs. The request:

> *Twenty things were dumped in a heap on the kitchen table. Lay them out for cooking: the bowl in front of the pitcher, the ladle and the big spoon within reach of the bowl, the fruit (both oranges, the lime, the lemon) grouped together on the left, the cans and bottles in a row at the back, the hammer, the bin, the remote and the spatula out of the way on the right. Everything upright on the table, at least 2 cm from the edge, nothing interpenetrating, and do not move anything further than necessary.*

The first program has 46 statements and the certified second one 44: the usual guards, one `place` target per object (the placement in scale-space), and the relations that make the request checkable:

```json
{"op": "place", "a": "bowl", "x": 0.62, "y": 0.00, "reason": "bowl in front (+x) of the pitcher"},
{"op": "place", "a": "orange_02", "x": 0.48, "y": 0.34, "reason": "fruit group on the left (+y)"},
{"op": "place", "a": "pineapple_slices_can", "x": 0.80, "y": 0.42, "reason": "cans and bottles in a row at the back (+x)"},
{"op": "farther", "a": "bowl", "b": "pitcher", "gap": 0.12},
{"op": "close_to", "a": "ladle", "b": "bowl", "r": 0.22},
{"op": "left_of", "a": "orange_02", "b": "bowl", "gap": 0.18},
{"op": "right_of", "a": "hammer_8", "b": "bowl", "gap": 0.22},
...
"gate": {"settle_v_max": 1.0, "settle_dx": 0.15}
```

Two rounds of the agent loop, from the tools' own outputs (`viz/run_agent_rounds.sh`); round 2's program is Claude Opus 5.5's answer to round 1's report (`viz/agent_revise.py`). The request, the programs, the call, every tool output and both certificates are in [`docs/examples/pile_n20/`](docs/examples/pile_n20/).

| round | repair | outcome | settle (hull / CoACD) |
|---|---|---|---|
| 1 | 82 → 0 pairs, RMSD 0.335 m, 9 s | the hammer's target missed by 5.3 cm, the ladle 2.6 cm beyond its 22 cm | 0.38 / 0.33 m/s, pass |
| 2 | 82 → 0 pairs, RMSD 0.325 m, 4 s | every predicate holds, `ready: true` | 0.38 / 0.33 m/s, pass |

- **Round 1.** The 33 cm hammer does not fit at the requested corner next to the bin, and the hammer and the bin end at the table's edge. The report lists the rewrites: drop the hammer's target, relax the ladle's radius to 24.7 cm, or make the 2 cm edge inset a preference, which the request rules out. A smaller margin alone still misses the hammer's target.
- **Round 2.** The hammer and the bin are turned to fit inside the inset, with the smaller margin and the ladle's and the spoon's reach as a 25 cm preference.
- **The heap itself**, settled as is, reaches 5.2 m/s with eleven bodies off the table (hull proxies) and 3.8 m/s with one (CoACD).
- **After the settle.** The program evaluated on the settled poses (`after_settle`) shows what the thresholds do not: the round fruit rolls out of `upright` (the lime and one orange, under either proxy) while every other statement holds. The default policy leaves `ready` to the G5 thresholds and reports this; with `--hold-predicates` the certificate is not ready until the request is changed for those bodies.

The program's schedule starts the continuation at s = 0.3, large enough to see the bodies while they are placed.

### Example 3: shelves

A fixture's levels come from its mesh. The scene summary lists each one, with the room above it:

```
case.level2 z=0.315 x[0.48, 0.83] y[-0.25, 0.25] headroom 0.261 m
```

What a program can say about them:

- `on_support(book_*, case.level2)`: rest on that level.
- `within`, `keep_out`: stay in, or out of, part of a level.
- `flush(book_1, case.level2, side=near, gap=0.01)`: an edge 1 cm behind the level's front.
- `row(..., edge=min)`: a row whose near edges line up (spines in line).
- `yaw(book_*, deg=180, axis=front)`: the side a body's data names faces out.

`check` refuses a body taller than the room above its level, and names the levels that have room. Three runs on SimReady and PalatialSim assets ([clips](#shelves)); every request, program, tool output and certificate is in [`docs/examples/`](docs/examples/), and the layouts come from `experiments/shelf_layouts.py`.

| example | request | repair | settle (hull / CoACD) |
|---|---|---|---|
| `shelf_books` | books upright by height, spines out; the two tallest on top; notebooks stacked at the back | 18 → 0 pairs, 13 s | 0.06 / 0.16 m/s |
| `shelf_pantry` | 45 things, heavy ones low; mugs and soup in a front row, handles right; cans in a 2 × 9 grid; jars by mass | 23 → 0 pairs, 28 s | 0.87 / 0.21 m/s |
| `shelf_fix` | 45 things in rows 3 mm too tight: fix them, moving nothing more than 2 cm | 36 → 0 pairs, 20 s | 0.11 / 0.04 m/s |

- `shelf_books` took two rounds: the first lined up the books' centres, so their spines could not line up too; the second uses `edge=min`. Round 1's report lists the nearest relaxations (the row's tolerance from 1.0 to 1.3 cm, a spine from 5 to 8 mm off its line); they accept its layout, but the request puts every spine 1 cm behind the front edge, which `edge=min` keeps.
- The hull settle fills the bowls in, and the upper bowls slide off; CoACD keeps their shape.

## 🤖 The Agent Loop

The agent is a coding agent following [`skills/simready-scenegen/SKILL.md`](skills/simready-scenegen/SKILL.md), or the SDK backend in `simready/dsl/text2dsl.py` (Claude with schema-constrained output). It

1. reads `summarize`, writes the program from the request and validates it with `check`;
2. runs `repair --out` and reads `pen_after`, `failed_predicates` and `rmsd_xy`;
3. acts on the outcome for at most three rounds, with the report's rewrites: each failed requirement comes with the nearest values of its bounds at which it holds and whether it may be weakened, and the report says what the repair ends with at a smaller margin; a body in `left_support` after `settle` gets a `within(..., inset=)` statement;
4. reports the certificate path: a scene is ready when that certificate says `ready: true`, and nothing else counts.

It never overrides a number, and final poses are not something it types by hand: a body is placed through relations, regions and `place` targets, and S4R decides what is feasible. In scale-space every body is shrunk about its reference center, where overlaps can remain and a body can be moved anywhere; `place(a, x, y, yaw)` is that move, and restoring the scale resolves whatever still overlaps, with the target kept as a soft pull. The pipeline is shrink, arrange, restore.

## 🧩 The Language

```
program
  no_penetration(*)
  fixed(table)   on_support(*, table)   upright(*)
  within(*, table.top, inset=0.02)
  place(bowl, x=0.62, y=0.00)            # a target pose in the shrunken scale-space (soft)
  farther(bowl, pitcher, gap=0.12)       # farther from the robot: a.x >= b.x + gap (+x away from it, +y left)
  close_to(ladle, bowl, r=0.22)   left_of(orange_02, bowl, gap=0.18)   right_of(hammer_8, bowl, gap=0.22)
  minimize displacement(*)
schedule
  margin = 0.01                          # how the repair gets there, never what is accepted
gate
  G5: v_max <= 1.0, dx <= 0.15
```

| statement | compiles to |
|---|---|
| `minimize displacement(*)` / `prefer(a, w)` | the QP objective: minimum-norm step, plus a pull toward a pose |
| `no_penetration(*)` | every contact row S4R generates from FCL |
| `on_support(a, B)`, `upright(a)` | a's lowest point on B's surface, roll and pitch removed from the QP variables |
| `within(a, R)`, `inside(a, C)` | per-step rectangle rows on a's exact projected footprint |
| `left_of / right_of / farther / nearer`, `min_distance / close_to` | linear rows on the reference centers |
| `place(a, x, y, yaw)` | a's pose at the shrunken scale, then a soft pull while the scale is restored |
| `schedule` | the contacts' clearance (`margin`) and the continuation's steps |
| `gate` | thresholds for the verifier and the settle test, not the QP |

Every statement that constrains a pose is also a predicate the final scene is checked against, and a line of the certificate. A program is the conjunction of its statements ([what a program means](docs/DSL.md#what-a-program-means)); when a requirement fails, the report gives each failed statement the nearest values of its bounds under which it holds on the returned layout ([rewrites](docs/DSL.md#when-a-requirement-fails)).

**What robot scenes ask for.** Where the arm can reach and grasp, how a tidy scene is arranged, which zones stay free, how far a shipped scene may change, what stands on what. Each statement compiles to rows of the same step QP, and to a predicate:

```
program
  reach(mug, r=0.60, min=0.45)   facing(mug, axis=-x, tol=20)   approach(mug, length=0.6, width=0.15)
  clearance(mug, gap=0.03)       keep_out(*, x0=0.25, x1=0.40, y0=0.20, y1=0.40)
  at(plate, x=0.55, y=0.0)       beside(fork, plate, side=left, gap=0.02, max=0.05)   symmetric(fork, knife, plate, axis=y)
  yaw(fork, deg=0, axis=long)    parallel(spoon, knife)
  row(can_*, start=left, gap=0.02, order=height, spacing=even, mid=0)   evenly_spaced(jar_1, jar_2, jar_3, axis=x)
  grid(box_1, box_2, box_3, box_4, box_5, box_6, cols=3)   cluster(apple, orange, lime, r=0.12)
  ring(cup_*, r=0.18, x=0.55, y=0, face=out, axis=handle)
  within(apple, table.top, side=left, frac=0.33)   stack(box_large, box_medium, box_small)   balanced_on(box_medium, box_large)
  max_move(*, d=0.03)   max_turn(*, deg=10)   left_of(cup, plate, gap=0.10, soft=0.5)
```

A pattern (`can_*`, `fruit_?`) stands for every free body whose name matches, in natural order, so a program stays the same length as the scene grows: sorting a heap takes 11, 13 and 16 statements at 10, 16 and 24 objects, against 18, 25 and 35 with every body named, and no member of a group can be left out by a slip.

<details>
<summary><b>How each family compiles</b></summary>

| family | statements | compiles to |
|---|---|---|
| the robot | `reach`, `clearance`, `approach`, `facing` | the tangent half-plane of the distance to the base; the contact rows' margin; the free side of a corridor toward the robot; first-order rows on the heading toward a point |
| positions and headings | `at`, `beside`, `yaw`, `parallel`, `max_turn` | exact rows on centres, footprint projections and headings (`axis=long`: a body's long side) |
| arrangements | `row` (`start=`, `order=height` / `width` / `length` / `mass` sorted by the compiler, `spacing=even`, `mid`, `edge=min`), `align`, `evenly_spaced`, `grid`, `ring`, `cluster`, `symmetric`, `between`, `centered`, `offset`, `attach` | exact rows on centres and footprints; for `ring`, at the angle that fits the current poses; the tangent row of each member for `cluster`; first-order rows for an offset in a body's own frame |
| regions | `within(a, b, side, frac, depth, x0..y1)`, `keep_out`, `flush(a, b, side, gap)`; `b.levelK` is a fixture's level | rectangle rows on the footprint; for `keep_out`, the side that needs the least motion and leaves the body room in its own regions |
| edits and stacks | `max_move`, `stack`, `balanced_on` | a 16-gon inscribed in the allowed disc; supports that follow the body below, centring rows; the centre of mass inside the base's rectangle in its own frame |
| preferences | `soft=w` on any of them | the same rows with a slack weight a hundred times smaller: reported, not required |

A program with these statements is first solved alone at the shrunken scale, where the bodies can pass each other, before the scale is restored (the relational counterpart of `place`); a program the contacts contradict ends penetration-free with the failing statement named. Every linearised row is checked against finite differences of its own residual in `tests/test_dsl_v1.py`.

</details>

[`docs/DSL.md`](docs/DSL.md) lists every statement with its rows, its predicate and its defaults, and recipes for place settings, shelves, sorting a heap, pick-ready scenes, small fixes, palletizing and towers.

## 🔧 Into the Engines

`export` writes a scene the engines load directly: one OBJ per body in its own frame, a manifest with every pose (metres, Z up), an MJCF file, and a USD stage with UsdPhysics rigid bodies, colliders and a physics scene. The runners in `examples/` load it, step it and report the peak and final displacement of every free body.

```bash
python -m simready.cli export pile_repaired.json --out pile_export --program program.json
python examples/run_mujoco.py pile_export --seconds 2      # MuJoCo, from scene.xml
python examples/run_genesis.py pile_export --seconds 2     # Genesis, from manifest.json (add --cpu without a GPU)
python examples/run_isaac.py pile_export --seconds 2       # Isaac Sim (its own Python), from scene.usda
```

Each runner takes `--record poses.json` (the world pose of every free body at 30 frames per second) and `--report report.json`; `viz/make_engine_video.py` turns the recordings into a side-by-side clip, and `viz/run_engines.sh` runs the whole check: the export, each engine for 2 s, and the clip.

<p align="center"><img src="docs/media/pile_n20_engines.gif" width="100%" alt="the certified twenty-object layout simulated for 2 s in MuJoCo, Genesis and Isaac Sim, left to right"/></p>
<p align="center"><i>The twenty-object layout certified in September 2026, 2 s in MuJoCo 3.10, Genesis 1.2.3 and Isaac Sim 4.5 (left to right), rendered from the recorded trajectories. The round fruit rolls a little (an orange by 9 cm in MuJoCo, 5 cm in Genesis, 3 cm in Isaac Sim); nothing tips over and no body leaves the table.</i></p>

| engine | peak displacement of a free body | bodies below the ground |
|---|---|---|
| MuJoCo 3.10 | 0.089 m (an orange rolling, as in the settle test) | none |
| Genesis 1.2.3, GPU, conjugate-gradient solver | 0.053 m; 0.050 m with one hull per body | none |
| Isaac Sim 4.5 (PhysX) | 0.029 m | none |

<details>
<summary><b>Collision proxies and the ground</b></summary>

- Fixtures keep their full meshes; a fixture with shelves is written as CoACD pieces checked against the bodies resting on it (`--fixture-grid` for an open mesh).
- The MJCF file gives every free body one convex hull unless `--decompose` writes CoACD pieces; Genesis decomposes the meshes itself (`--convex` for one hull per body); Isaac Sim's stage asks PhysX for a convex decomposition.
- A fixed body whose mesh is a single plane is kept as a 1 cm solid unless it is the ground (a horizontal sheet at the ground height, or one tagged `ground`), which the export's own ground replaces. Such a sheet is marked `flat` in the manifest, and every runner uses the export's ground height.
- Genesis's Newton constraint solver needs more shared memory than a 2080-class GPU offers at 120 free degrees of freedom, so the runner uses its conjugate-gradient solver by default (`--solver newton` to switch).
- Each OBJ is named by the body's index, so no two bodies can share a file.

</details>

## 🎬 Gallery

> Every clip is rendered from the solver's own meshes with the assets' textures, without overlays.

### Shelves

<table>
<tr>
<td align="center" width="33%"><img src="docs/media/shelf_books_gate.gif" alt="books and notebooks heaped on a bookcase's shelves, then shelved by the program"/></td>
<td align="center" width="33%"><img src="docs/media/shelf_pantry_gate.gif" alt="kitchen things heaped on a rack, then stocked by the program"/></td>
<td align="center" width="33%"><img src="docs/media/shelf_fix_gate.gif" alt="rows on a rack 3 mm too tight, then spaced out"/></td>
</tr>
<tr>
<td align="center"><sub><code>shelf_books</code></sub></td>
<td align="center"><sub><code>shelf_pantry</code></sub></td>
<td align="center"><sub><code>shelf_fix</code></sub></td>
</tr>
</table>
<p align="center"><i>SimReady and PalatialSim assets on a bookcase and a rack, interpenetrating bodies tinted, then S4R's scale continuation under the agent's program: books by height with their spines in line; heavy things low and cans in a grid; rows 3 mm too tight, spaced out (<a href="#example-3-shelves">Example 3</a>).</i></p>

### Twenty objects from a heap

<table>
<tr><td align="center"><img src="docs/media/pile_n20_gate.gif" alt="twenty RoboLab objects heaped on the table, shrunk, laid out by the model in scale-space, restored to full size"/></td></tr>
<tr><td align="center"><img src="docs/media/pile_n20_settle.gif" alt="MuJoCo settle of the heap (left) and of the repaired layout (right)"/></td></tr>
</table>
<p align="center"><i>Top: twenty RoboLab catalog objects dropped into a 13 cm radius on RoboLab's own table (82 interpenetrating pairs, tinted), laid out for cooking and penetration-free with every predicate satisfied (<a href="#example-2-twenty-objects-from-a-heap">Example 2</a>). Bottom, in MuJoCo with CoACD proxies: the heap (left) reaches 3.8 m/s and throws a body off the table; the repaired scene (right) peaks at 0.33 m/s with nothing leaving the table.</i></p>

### A scene RoboLab ships

<table>
<tr>
<td align="center"><img src="docs/media/workdesk_gate.gif" alt="RoboLab's shipped workdesk_snacks scene repaired"/></td>
<td align="center"><img src="docs/media/workdesk_settle.gif" alt="MuJoCo settle of the shipped scene, before and after"/></td>
</tr>
</table>
<p align="center"><i><code>workdesk_snacks</code>, already physics-settled by RoboLab's pipeline, still has three object-object interpenetrations, the largest 9.9 mm. One sentence of intent and one certificate later it is clean, with 4.0 cm of planar motion (<a href="#example-1-a-scene-robolab-ships">Example 1</a>). Left: the repair, interpenetrating bodies tinted. Middle and right: settled in MuJoCo, as shipped and after the repair.</i></p>

### A layout RoboLab's solver rejects

<table>
<tr>
<td align="center"><img src="docs/media/layout_n10_gate.gif" alt="a solver-rejected RoboLab layout: seated with four interpenetrating pairs, shrunk, grown back into a clean arrangement"/></td>
<td align="center"><img src="docs/media/layout_n10_settle.gif" alt="MuJoCo settle: as generated versus repaired"/></td>
</tr>
</table>
<p align="center"><i>Ten catalog objects placed by RoboLab's own disc solver, which reports failure; seated, the layout carries four interpenetrating pairs. The gate repairs it to zero with the requested left/right/front/back relations kept. Left: the repair, interpenetrating bodies tinted. Middle and right: in MuJoCo, the generated layout tips the tomato can onto its side and moves a body 23 cm; the repaired one moves less than 1 cm.</i></p>

### A RoboCasa counter

<table>
<tr><td align="center"><img src="docs/media/robocasa_ring_gate.gif" alt="ten AI-generated RoboCasa objects in a heap, laid out on a circle"/></td></tr>
</table>
<p align="center"><i>Ten AI-generated RoboCasa objects in a heap at the centre of a counter, with nine interpenetrating pairs. One statement, <code>ring(*, r=0.2, x=0, y=0)</code>, lays them out evenly on a 20 cm circle; S4R reaches zero penetration in 0.3 s on the objects' V-HACD collision pieces, under <code>on_support</code> and <code>upright</code>.</i></p>

## 📈 Results

### A program or the poses

The same requests on the same SimReady scenes, answered by Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 in isolated `claude -p` calls (no tools, memory, CLAUDE.md or MCP servers; a leak test with planted canaries before the run):

- **program**: the model writes a SimReady program, which the gate compiles and repairs;
- **poses**: the model types the final pose of every object itself.

Sixteen tasks, from place settings and rows of drinks to sorting a heap, towers, palletizing and dense packing, two of them written only after the language was frozen; three seeds; up to three rounds, each told what its condition's check found. The method and every table are in [`docs/EVAL_DSL_vs_direct.md`](docs/EVAL_DSL_vs_direct.md), every trial in [`results/dsl_vs_direct/`](results/dsl_vs_direct/).

<p align="center"><img src="docs/media/dsl_vs_direct.png" width="100%" alt="pass rate, output tokens per passing scene and seconds per passing scene of programs, of poses and of programs restricted to the earlier statements, for Haiku, Sonnet and Opus 5.5"/></p>

**Programs and poses on sixteen tasks** (three seeds; the better answer per model in bold).

<table align="center">
<tr><th><sub>Model</sub></th><th><sub>Answer</sub></th><th><sub>Scenes that pass</sub></th><th><sub>Output tokens / trial</sub></th><th><sub>Cost / trial</sub></th><th><sub>Seconds / passing scene</sub></th></tr>
<tr><td rowspan="2"><sub>Haiku 5.5</sub></td><td><sub>program</sub></td><td><sub><b>45 / 48</b></sub></td><td><sub><b>6,761</b></sub></td><td><sub><b>$0.0041</b></sub></td><td><sub><b>36</b></sub></td></tr>
<tr><td><sub>poses</sub></td><td><sub>42 / 48</sub></td><td><sub>12,474</sub></td><td><sub>$0.0068</sub></td><td><sub>63</sub></td></tr>
<tr><td rowspan="2"><sub>Sonnet 5.5</sub></td><td><sub>program</sub></td><td><sub><b>48 / 48</b></sub></td><td><sub><b>812</b></sub></td><td><sub><b>$0.019</b></sub></td><td><sub><b>10</b></sub></td></tr>
<tr><td><sub>poses</sub></td><td><sub>41 / 48</sub></td><td><sub>2,134</sub></td><td><sub>$0.029</sub></td><td><sub>22</sub></td></tr>
<tr><td rowspan="2"><sub>Opus 5.5</sub></td><td><sub>program</sub></td><td><sub><b>48 / 48</b></sub></td><td><sub><b>795</b></sub></td><td><sub><b>$0.036</b></sub></td><td><sub><b>10</b></sub></td></tr>
<tr><td><sub>poses</sub></td><td><sub>45 / 48</sub></td><td><sub>3,122</sub></td><td><sub>$0.078</sub></td><td><sub>33</sub></td></tr>
</table>

- **Programs pass where poses need geometry the model cannot see.** Typed poses put ten drinks in the wrong order of height in 7 / 9 rows (the scene summary gives heights to the centimetre; the compiler sorts on the meshes), and packed fruit interpenetrated in 3 / 18 typed packings. Where the summary says enough, typed poses do as well: they stacked the three cartons centred in all nine towers. Every scene of either answer stays at rest in MuJoCo.
- **A program does not grow with the scene.** Sorting a heap of 10, 16 and 24 objects, Haiku's poses grow from 6,100 to 24,300 output tokens per trial while its programs stay between 4,600 and 6,700; Sonnet's go from 1,300 to 5,000 against 400 to 1,000, Opus's from 2,100 to 6,500 against 880 to 1,100.
- **Cheaper and faster per scene.** Programs take 1.8 to 3.9 times fewer output tokens per trial, cost 1.5 to 2.2 times less, and need 1.7 to 3.3 times less time per passing scene, the solver's 2-3 s per trial included. The system prompt that carries the language is longer (14-18k prompt tokens per trial against 3-4k), and it is read from the prompt cache.
- **Where poses do as well.** On the two held-out tasks, of six and seven objects, programs pass 6 / 6 with every model and poses 6 / 6, 5 / 6 and 6 / 6. Poses take less time with Haiku and Sonnet (13-15 s per passing scene against 22-25 s) and about as much with Opus (14 s against 12 s): when the scene is small and nothing in it needs exact geometry, the solver's seconds are the program's price.
- **A program is reusable.** Applied unchanged to five new heaps of its task, it passes 697 of the 705 scenes; with every object replaced by another asset of its category it passes 140 of 141 scenes, typed poses 79 of 128.
- **The new statements and the request both matter.** Programs restricted to the statements the language had before pass 106 / 144. Poses repaired by S4R with only the geometric statements every task shares (and the supports the answer named) pass 115 / 144, fewer than the poses themselves: the repair removes interpenetration but does not know the request (it pushed packed fruit out of their square).
- **After a failure, the rewrites help.** Told the rewrites of its failed statements, each of Haiku's three failing programs passes. On 92 faulted first answers per model both reports recover nearly every fault; with the rewrites no request is weakened (the plain report weakened three), and after a margin fault the next round takes 2.3 to 4.4 times fewer output tokens.

<p align="center"><img src="docs/media/dsl_vs_direct_scaling.png" width="90%" alt="output tokens and model seconds per trial of programs and poses when sorting a heap of 10, 16 and 24 objects, for Haiku, Sonnet and Opus 5.5"/></p>

<details>
<summary><b>The runs</b></summary>

The numbers are those of the run of 2026-10-26 with the current language. The two earlier runs (2026-10-24 and its replay of 2026-10-25) answered an earlier prompt and are published with it ([the runs](docs/EVAL_DSL_vs_direct.md#the-runs)). One rubric check was corrected after the first model's run of the first run: "within 1 cm of a common line" had been measured from the centres' mean, which asks for more than the request; every round of the first run it could change was rebuilt and graded again.

</details>

### Diagnostics

Measured on RoboLab's 68 shipped scenes, on pre-settle layouts from RoboLab's own placement solver staged in its base scene, and on RoboCasa counter regions with RoboCasa's own placement test. Every number is backed by a tracked file under [`results/`](results/).

- **RoboLab library.** Object-object mesh penetration in 26 / 68 scenes (12 at or above 1 mm; all of them surface intersections, none a body wholly inside another), and a body nothing supports in 30.
- **RoboLab pre-settle layouts** (N = 4–10, 3 seeds). The disc solver accepts 5 / 12, and those are clean; the 7 it rejects carry 1–7 penetrating pairs, and S4R repairs all 7 to zero with every requested relation and region kept (planar RMSD 0.03–0.16 m, 3–15 s per cell).
- **Settle** (MuJoCo, identical harness). Median peak speed 0.5 → 0.3 m/s (hull proxies) and 0.4 → 0.2 m/s (CoACD) from the seated raw layouts to the repaired ones; every repaired cell passes and nothing leaves the table.
- **RoboCasa** (its own placement test, 5 seeds). The box test is conservative and reliable; it gives up in a 0.3 m square at 8–12 objects (4/5, 1/5, 0/5), where the gate places every object (5/5).

Full tables: [`docs/DIAGNOSTIC_2026-09-08.md`](docs/DIAGNOSTIC_2026-09-08.md). Design note: [`docs/design.html`](docs/design.html).

## 📝 Citation

If you find SimReady Gate useful in your research, please consider citing it together with S4R, which it builds on:

```bibtex
@article{dou2026s4r,
  title   = {S4R: Scaling for Rigid-Body Interpenetration Resolution},
  author  = {Dou, Zhiyang and Zhao, Ang and Peng, Chen and Guo, Minghao and
             Wu, Haixu and Lin, Cheng and Liu, Yuan and Yao, Junfeng and
             Guo, Xiaohu and Wang, Wenping and Matusik, Wojciech},
  journal = {ACM Transactions on Graphics (SIGGRAPH Asia)},
  year    = {2026},
}

@misc{dou2026simreadygate,
  title        = {SimReady Gate: Language-Driven, Verifier-in-the-Loop Repair of Generated Scenes},
  author       = {Dou, Zhiyang and Matusik, Wojciech},
  year         = {2026},
  howpublished = {\url{https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/main/SimReady_Gate}},
  note         = {Part of Dynamics-Modeling},
}
```

## ⚖️ License

The code is released under the [MIT License](LICENSE). RoboLab and RoboCasa assets are not redistributed here; the experiments read them from their own checkouts (`ROBOLAB_DIR`, `ROBOCASA_AIGEN_DIR`). Assets fetched from NVIDIA SimReady, the Scenario 3D Collection and PalatialSim keep their own licenses.

## 🤝 Acknowledgement

The scenes and assets come from [RoboLab](https://github.com/NVLabs/RoboLab), [RoboCasa](https://github.com/robocasa/robocasa), [NVIDIA SimReady](https://huggingface.co/datasets/nvidia/simready-assets), the [Scenario 3D Collection](https://huggingface.co/datasets/scenario-labs/3d-collection) (credit: Scenario) and [PalatialSim](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library) (credit: PalatialSim). Contacts, QPs, simulation and decomposition run on [FCL](https://github.com/flexible-collision-library/fcl), [OSQP](https://github.com/osqp/osqp), [MuJoCo](https://github.com/google-deepmind/mujoco), [CoACD](https://github.com/SarahWeiii/CoACD) and [OpenUSD](https://openusd.org). We thank the authors for making them available.
