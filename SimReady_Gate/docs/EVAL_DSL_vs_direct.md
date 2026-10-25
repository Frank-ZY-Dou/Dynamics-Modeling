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

Sixteen tasks in eleven families, each a heap or a scatter of SimReady packages (`assets.json`, 60
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
| `tool_layout` (held out) | 7 | six tools in one row parallel to y, sorted by length from the left, 3 cm gaps, each pointing at the robot, all on the near part of the table; the tape dispenser on the far part |
| `desk` (held out) | 6 | the keyboard placed and turned along y, the mouse and the notebook beside it with gaps, the marker beyond the notebook, the mug beyond the keyboard with its handle at the robot, the tape dispenser 20 cm from everything |

The language's last revisions came from pilots with Haiku on the first fourteen tasks: name patterns
(`can_*`), after a model left a can out of a listed row; `row`'s `start` (the end the first member is
at) and `mid` (where the two end members are centred), after programs reversed a row or could not
centre one whose ends only the compiler's sort decides. The same pilots showed that the sixth can of
`drinks_row` held soup, which a model rightly kept out of "all the drinks"; a beverage can replaced
it. The two held-out tasks were written after that, with the language frozen, and were run only in
the final comparison; the tables report them apart. The pilots are not part of the results.

A first attempt at the final run was stopped two minutes in: the environment lacked the mesh
decimator, the repair ran on full meshes, and the workers ran out of memory. Its trials were
discarded; the run reported here started again from scratch with the decimator installed.

A request states every number its rubric checks, in the frame both conditions are told (+x away
from the robot, +y to its left). The rubric (`tasks.py`, `rubric.py`) is a list of geometric
checks written from the request's wording, the same for every condition, with 5 mm and 2 degrees of
slack beyond the request's numbers; it also requires what every task asks: no interpenetrating pair
(the mesh evaluator), every body resting on something, upright within 5 degrees, over the support.
No model sees it. Every task has a reference program that passes it (`reference` in `tasks.py`),
shown to no model: the tasks are feasible.

One check was corrected after the first model's run. "Centres within 1 cm of a common line" was
measured during the runs as each centre's distance from the centres' mean, which asks for more than
the request says; it is now half the range of the centres, their distance from the best common
line (`rubric.line_offset`). The correction can only turn a failure into a pass, and only for a
round whose failed checks were all line checks: `regrade.py` rebuilds exactly those rounds' scenes
from the recorded answers (no model is called; the pipelines are deterministic) and grades them
again, and each rebuilt scene must reproduce the outcome of every other check. The tables give the
verdicts as run and as regraded (`<model>/regrade.json` lists every changed trial).

## Conditions and rounds

The model gets a fixed system prompt per condition (`results/dsl_vs_direct/<model>/system_*.txt`)
and a message with the scene summary (`python -m simready.cli summarize`: every body's size, local
size, heading, tags, the support's top rectangle) and the request. The answer format is given as a
JSON schema (structured output), one per condition: body names are checked against the scene when
the answer is validated, not listed in the schema, so that every call of a condition shares one
cached prompt prefix.

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

## Costs and timing

Tokens and cost are the CLI's own accounting of each call (`usage`, `total_cost_usd`): output tokens
include the model's thinking, prompt tokens are split into those read from the prompt cache and the
rest. Model time is the call's duration as the CLI reports it. Solver time is the wall-clock time of
the pipeline after the answer (validation, compilation, S4R and verification for a program; seating
and verification for poses), with the worker's CPU seconds recorded beside it. The run used six
worker processes, one thread each, on eight physical cores reserved for it (no SMT siblings in use);
models ran one after another, three seeds per task and condition, the trials submitted in one fixed
order.

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

Every number below comes from the trial files of the run (`results/dsl_vs_direct/<model>/trials/`);
`analyze.py` writes every table, per task and per scene size, to
[`results/dsl_vs_direct/summary.md`](../results/dsl_vs_direct/summary.md). Pass rates carry 95 %
Wilson intervals; a condition's time per passing scene is its total over its passing trials. The
verdicts are those of the corrected line check; the verdicts as first graded are in the last table.

![pass rate, output tokens and seconds per passing scene](media/dsl_vs_direct.png)

### Pass rates and costs (sixteen tasks, 48 trials per answer and model)

| model | answer | pass (95% CI) | round 1 | rounds | output tokens / trial | prompt tokens / trial (not cached) | cost / trial | model s / trial | solver s / trial | s / passing scene |
|---|---|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 | program | 47/48 (89-100%) | 98% | 1.10 | 4,051 | 13,464 (1,640) | $0.0025 | 19 | 6.7 | 26 |
|  | program, earlier statements | 36/48 (61-85%) | 73% | 1.21 | 7,565 | 8,868 (2,946) | $0.0044 | 34 | 7.1 | 54 |
|  | poses | 35/48 (59-83%) | 65% | 1.48 | 15,425 | 3,905 (2,059) | $0.0081 | 66 | 1.8 | 93 |
|  | poses, repaired | 36/48 (61-85%) | 75% | 1.48 | 15,425 | 3,905 (2,059) | $0.0081 | 66 | 7.0 | 98 |
| Sonnet 5.5 | program | 48/48 (93-100%) | 94% | 1.06 | 633 | 12,989 (1,655) | $0.0141 | 5 | 7.6 | 13 |
|  | program, earlier statements | 30/48 (48-75%) | 56% | 1.17 | 1,563 | 7,300 (2,197) | $0.0249 | 11 | 6.9 | 29 |
|  | poses | 35/48 (59-83%) | 56% | 1.60 | 3,341 | 4,211 (2,625) | $0.0441 | 26 | 1.9 | 39 |
|  | poses, repaired | 36/48 (61-85%) | 69% | 1.60 | 3,341 | 4,211 (2,625) | $0.0441 | 26 | 7.2 | 45 |
| Opus 5.5 | program | 48/48 (93-100%) | 100% | 1.02 | 764 | 12,414 (1,582) | $0.0301 | 8 | 6.9 | 15 |
|  | program, earlier statements | 38/48 (66-88%) | 75% | 1.06 | 2,473 | 6,018 (1,745) | $0.0643 | 24 | 6.8 | 39 |
|  | poses | 40/48 (70-91%) | 77% | 1.29 | 2,783 | 3,281 (2,012) | $0.0720 | 29 | 1.7 | 36 |
|  | poses, repaired | 39/48 (68-90%) | 81% | 1.29 | 2,783 | 3,281 (2,012) | $0.0720 | 29 | 7.0 | 44 |

A program passed in 143 of 144 trials and typed poses in 110. The one failing program (Haiku, ten
drinks) left the centres up to 1.7 cm from a common line and the line at x = 0.69, short of the band
from 0.70 to 0.80 the request named. Programs
passed at the first round in 94-100 % of trials and needed 1.02-1.10 rounds on average; poses needed
1.29-1.60. Programs took 3.6 to 5.3 times fewer output tokens per trial and cost 2.4 to 3.3 times
less, and a passing scene took 2.5 to 3.6 times less time, the solver's 6.7-7.6 s per trial included
(its wall-clock and CPU seconds agree to 0.1 s: the cores were the run's alone). The language makes
the system prompt longer, 12-13k prompt tokens per trial against 3-4k, but that part is read from
the prompt cache: a program trial sent fewer tokens fresh (1.6k) than a pose trial (2.0-2.6k), whose
longer answers come back in its later rounds.

### Where each answer fails

- Poses: interpenetration in 23 of the 34 failing trials (every tower of three cartons, 9 / 9; six
  cartons on a pallet, 8 / 9, and four, 3 / 9; the square of ten fruit, 3 / 9); the order of height
  in the row of ten drinks (8 / 9: the scene summary gives sizes to the centimetre, the compiler sorts
  on the meshes); in every tower, cartons resting on nothing.
- Programs restricted to the earlier statements: no region bounded by coordinates (15 failed square
  containments in the packing tasks), no sorting (all nine rows of ten drinks), no handle turned to
  the robot (setting, desk), no stack.
- Programs: the one row above.

### Held-out tasks

| model | answer | development tasks (42 trials) | held-out tasks (6 trials) | held out: output tokens / trial | held out: s / passing scene |
|---|---|---|---|---|---|
| Haiku 5.5 | program | 41/42 | 6/6 | 4,689 | 34 |
|  | program, earlier statements | 31/42 | 5/6 | 7,249 | 53 |
|  | poses | 29/42 | 6/6 | 3,052 | 17 |
| Sonnet 5.5 | program | 42/42 | 6/6 | 1,364 | 31 |
|  | program, earlier statements | 26/42 | 4/6 | 2,140 | 43 |
|  | poses | 29/42 | 6/6 | 957 | 11 |
| Opus 5.5 | program | 42/42 | 6/6 | 897 | 22 |
|  | program, earlier statements | 32/42 | 6/6 | 1,818 | 31 |
|  | poses | 34/42 | 6/6 | 1,113 | 16 |

The two tasks written after the language was frozen passed in every trial, with programs and with
poses, for every model. They are small (six and seven objects) and need no geometry the model cannot
read off the summary; poses were the cheaper answer there for Haiku and Sonnet and the faster one for
all three (11-17 s per passing scene against 22-34 s), as the solver took 13-22 s per trial on the
tools' and the desk's meshes (up to 92k faces each).

### Scene size

| model | objects | program: pass, output tokens / trial, model s / trial | poses: pass, output tokens / trial, model s / trial |
|---|---|---|---|
| Haiku 5.5 | 10 | 3/3, 3,516, 17 | 3/3, 6,616, 28 |
|  | 16 | 3/3, 5,455, 25 | 3/3, 38,715, 165 |
|  | 24 | 3/3, 4,898, 22 | 3/3, 38,720, 164 |
| Sonnet 5.5 | 10 | 3/3, 351, 3 | 3/3, 2,464, 19 |
|  | 16 | 3/3, 413, 3 | 3/3, 4,737, 39 |
|  | 24 | 3/3, 975, 7 | 3/3, 6,398, 52 |
| Opus 5.5 | 10 | 3/3, 821, 8 | 3/3, 1,208, 12 |
|  | 16 | 3/3, 955, 10 | 3/3, 4,754, 46 |
|  | 24 | 3/3, 1,015, 10 | 3/3, 5,283, 53 |

Sorting a heap of 10, 16 and 24 objects: a program's output barely grows (patterns name the groups,
the solver places every body), typed poses grow with every object.

![output tokens and model seconds per trial against the number of objects](media/dsl_vs_direct_scaling.png)

### Physics, reuse and the corrected check

| model | answer | pass as first graded | pass | settles | passes and settles | new starts pass | spread of the layouts | other assets pass |
|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 | program | 44/48 | 47/48 | 48/48 | 47/48 | 97% of 235 | 8.9 cm | 100% of 47 |
|  | program, earlier statements | 36/48 | 36/48 | 47/48 | 35/48 | — | — | — |
|  | poses | 35/48 | 35/48 | 45/48 | 35/48 | 91% of 175 | 0.0 cm | 66% of 35 |
| Sonnet 5.5 | program | 45/48 | 48/48 | 48/48 | 48/48 | 94% of 240 | 13.1 cm | 98% of 48 |
|  | program, earlier statements | 30/48 | 30/48 | 48/48 | 30/48 | — | — | — |
|  | poses | 35/48 | 35/48 | 47/48 | 35/48 | 91% of 175 | 0.0 cm | 46% of 35 |
| Opus 5.5 | program | 45/48 | 48/48 | 48/48 | 48/48 | 93% of 240 | 14.3 cm | 100% of 48 |
|  | program, earlier statements | 38/48 | 38/48 | 48/48 | 38/48 | — | — | — |
|  | poses | 40/48 | 40/48 | 47/48 | 40/48 | 92% of 200 | 0.0 cm | 45% of 40 |

Every scene a program produced stays at rest in MuJoCo (CoACD pieces, the assets' friction and
mass); five towers from poses do not, three of them dropping a carton. Reuse applies each passing
answer, with no further call, to five new starting heaps of its task and to the task's scene with
every object replaced by another package of its category. A program is solved again from each start
and still passes 93-97 % of them, its bodies 9 to 14 cm from where they stood for another start (it
moves each body as little as the statements allow); the starts it fails are in the denser fruit
square, the kitchen layout and the pick-ready scene, where the solver stops short of the statements
from some heaps. Typed poses are one layout and pass 91-92 %: they fail only where the request depends
on the start (nothing moved more than 3 cm). With other assets,
programs pass 98-100 % and poses 45-66 %: poses typed for one geometry interpenetrate or fall out of
order with another.

Poses repaired by S4R with only the statements every task shares (`direct+repair` in the first
table) pass 36, 36 and 39 of 48: the repair removes interpenetration, so the pallets pass, but it does
not know the request, and it pushed packed fruit out of their square.

### Limits

- The language was revised on pilots over the fourteen development tasks; the held-out pair, the
  only tasks the revisions never saw, is small.
- One check of the rubric was corrected after the first model's run (above); every verdict it could
  change was regraded from the recorded answers, and both verdicts are published.
- Costs are the CLI's accounting at list price. The CLI adds its account and environment block
  (about 680 tokens) to every call, the same in every condition.
- Three seeds per task and answer: the intervals of a single task are wide; the pooled rates are
  tighter.
- Solver seconds depend on the machine: eight physical cores, one worker thread each.

## Reproducing

The calls need the `claude` CLI signed in (2.1.310 in the recorded run); a trial makes one to three
calls. The packages come from `assets.json`, fetched at its pinned commit:

```
export SIMREADY_ASSETS_DIR=/path/to/simready-assets SIMREADY_CACHE_DIR=/path/to/cache OMP_NUM_THREADS=1
python -m simready.cli assets fetch --revision 524747d9b5e1f21b5b18b56041788c01d745f1a1 $(python -c "import json; \
  d = json.load(open('experiments/dsl_vs_direct/assets.json')); \
  print(' '.join(sorted({p for v in d['pool'].values() for p in v} | {d['table'], d['pallet']})))")
python experiments/dsl_vs_direct/isolation_check.py --model claude-sonnet-5-5 --out runs
for M in claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5; do
  python experiments/dsl_vs_direct/run.py --model $M --seeds 0 1 2 --workers 6 --out runs/$M
  python experiments/dsl_vs_direct/regrade.py runs/$M
  python experiments/dsl_vs_direct/finals.py runs/$M --draw setting_n6 drinks_row_n10 fruit_cans_n24 tool_layout_n7
  python experiments/dsl_vs_direct/reuse.py runs/$M --starts 5     # or in slices: --first K --count 16, then --merge
done
python experiments/dsl_vs_direct/publish.py runs results/dsl_vs_direct
python experiments/dsl_vs_direct/analyze.py results/dsl_vs_direct/claude-* --md results/dsl_vs_direct/summary.md --json results/dsl_vs_direct/summary.json
python experiments/dsl_vs_direct/figures.py results/dsl_vs_direct/summary.json docs/media/dsl_vs_direct.png docs/media/dsl_vs_direct_scaling.png
```
