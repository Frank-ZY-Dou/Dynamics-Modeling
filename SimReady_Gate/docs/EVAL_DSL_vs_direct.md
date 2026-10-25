# A program or the poses: an isolated comparison

The same layout requests, on the same NVIDIA SimReady scenes, answered by the same model, in three
conditions: the model writes a SimReady program that the gate compiles and repairs (`dsl_v1`), the
same with only the statements the language had before this release (`dsl_v0`), or the model writes
the final pose of every object itself, with no program and no solver (`direct`). A fourth row,
`direct+repair`, takes direct's answers through S4R with only the geometric statements every task
shares (no penetration, upright on the support, over its top), without another call. Every model
call is an isolated `claude -p` process. The code is `experiments/dsl_vs_direct/`; the tables come
from the tracked trial files under `results/dsl_vs_direct/`.

## Tasks

Fourteen tasks in nine families, each a heap or a scatter of SimReady packages (`assets.json`, 59
packages of `nvidia/simready-assets` pinned to commit `524747d9`) on the stainless prep table turned
to span y in front of the robot (top x in [0.20, 0.90], y in [-0.90, 0.90], z = 0), or on a pallet:

| task | objects | what the request asks |
|---|---|---|
| `setting` | 6 | a place setting: plate position and reach, fork and knife mirror images about the plate with 2-5 cm gaps, cutlery parallel to x, spoon beside the knife, glass beyond the plate, mug handle at the robot |
| `drinks_row` | 6, 10 | one row parallel to y at x 0.70-0.80, shortest to tallest from left to right, 2 cm gaps, evenly spaced, centred on y = 0 |
| `fruit_cans` | 10, 16, 24 | fruit grouped on the left, cans in one line parallel to x on the right with 2 cm gaps, bowl (and mug) in the middle, a square in front of the robot kept empty |
| `pick_ready` | 8 | mug 45-60 cm from the base with its handle at the robot, a free 15 cm corridor to the base, every other object 18 cm away, a drop zone kept empty |
| `minimal_fix` | 12 | a scene with four interpenetrating pairs, fixed while moving nothing more than 3 cm and turning nothing more than 10 degrees |
| `stack` | 6 | three cartons stacked largest to smallest, centred within 1 cm, the tower on the right half, the rest on the left |
| `palletize` | 4, 6 | cartons on a pallet in rows parallel to y of at most 3, rows aligned within 1 cm, sides square to the pallet, 1 cm between boxes, 2 cm in from its edges |
| `kitchen_prep` | 12 | a cutting board placed and oriented, knife and spoon beside it with gaps, bowl beside it, fruit grouped beyond the bowl, drinks in a row at the far edge, nothing else near the board |
| `pack` | 10, 14 | all the fruit inside a square whose area is only 1.25 (1.12) times the sum of the fruit's bounding rectangles: dense clutter, where boxes do not fit but the shapes do |

A request states every number its rubric checks, in the frame both conditions are told (+x away
from the robot, +y to its left). The rubric (`tasks.py`, `rubric.py`) is a list of geometric
checks written from the request's wording, the same for every condition, with 5 mm and 2 degrees of
slack beyond the request's numbers; it also requires what every task asks: no interpenetrating pair
(the mesh evaluator), every body resting on something, upright within 5 degrees, over the support.
No model sees it. Every task has a reference program that passes it (`reference` in `tasks.py`),
shown to no model: the tasks are feasible.

## Conditions and rounds

The model gets a fixed system prompt per condition (`results/dsl_vs_direct/<model>/system_*.txt`)
and a message with the scene summary (`python -m simready.cli summarize`: every body's size, local
size, heading, tags, the support's top rectangle) and the request. The answer format is given as a
JSON schema (structured output), with body names restricted to the scene's.

- `dsl_v1`, `dsl_v0`: the answer is a program; the gate validates it against the scene, compiles
  it and repairs the starting scene with S4R (`simready.cli.repair_scene`, what `repair` runs:
  decimated proxies and support slabs, full-mesh verification and polish), and reports whether the
  program's own predicates hold, which fail and by how much, and whether pairs still penetrate.
- `direct`: the answer is the x, y and yaw of every free body (and, to stack, the body it rests on);
  each body is set down upright on the surface under its footprint, and the verifier reports the
  pairs that interpenetrate, the bodies that rest on nothing, leave the support or were not given.

A trial has up to three rounds. Each round is a new isolated call whose message repeats the request
with the previous answer and that condition's report; a trial ends when its own check passes. The
rubric grades the scene of every round.

## Isolation

Every call is a new `claude -p` process started by the harness (`llm.py`) with `--safe-mode`
(CLAUDE.md, skills, plugins, hooks, MCP servers, custom agents and output styles off),
`--no-session-persistence`, `--tools ""`, `--strict-mcp-config`, the condition's
`--system-prompt-file`, a fixed `--model` and `--effort`, `--json-schema` and `--output-format
json`, in a new empty working directory outside any repository, with an environment built from an
allowlist (HOME, PATH, locale) plus `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`: none of the variables of
the session that runs the experiment (its session ids, its messaging socket) reaches the call. The
message is built from the task files alone; no history crosses trials, and a later round of a
trial carries its earlier answer in the message itself.

`isolation_check.py` tests this before a run. It plants random canaries in a project CLAUDE.md, in
that project's auto-memory file and in a file of the working directory, then sends the same probe
("quote everything you were given besides this message") from that directory twice: with the
comparison's flags and environment, and with the ordinary configuration as a control. In the
recorded check (`results/dsl_vs_direct/isolation_check.json`) the isolated call shows none of the
canaries and its init event lists no tool but the structured-output one and no MCP server; the
control quotes the CLAUDE.md and auto-memory canaries, so the probe does see a leak when there is
one. What the CLI still adds to an isolated call is its own account and environment block (the
account's address, platform and date, about 680 tokens, the same for every call); it carries
nothing of the tasks or of earlier trials.

## Results

(filled from the recorded runs)
