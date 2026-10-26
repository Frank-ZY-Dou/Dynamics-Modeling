# A program or the poses: an isolated comparison

The same layout requests, on the same NVIDIA SimReady scenes, answered by the same model, in three
conditions: the model writes a SimReady program that the gate compiles and repairs (`dsl_v1`), the
same with only the statements the language had before this release (`dsl_v0`), or the model writes
the final pose of every object itself, with no program and no solver (`direct`). A fourth row,
`direct+repair`, takes direct's answers through S4R with only the geometric statements every task
shares (no penetration, upright on the support or on the body the answer named, over the support's
top), without another call. Every model
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
(`can_*`), after a model left a can out of a listed row; `row`'s `start` (the end the first member
is at) and `mid` (where the two end members are centred), after programs reversed a row or could not
centre one whose ends only the compiler's sort decides. The same pilots showed that the sixth can of
`drinks_row` held soup, which a model rightly kept out of "all the drinks"; a beverage can replaced
it. The two held-out tasks were written after that, with the language frozen, and were run only in
the final comparison and its replay; the tables report them apart. The pilots are not part of the
results.

A first attempt at the final run was stopped two minutes in: the environment lacked the mesh
decimator, the repair ran on full meshes, and the workers ran out of memory. Its trials were
discarded; the first run below started again from scratch with the decimator installed.

A request states every number its rubric checks, in the frame both conditions are told (+x away
from the robot, +y to its left). The rubric (`tasks.py`, `rubric.py`) is a list of geometric
checks written from the request's wording, the same for every condition, with 5 mm and 2 degrees of
slack beyond the request's numbers; it also requires what every task asks: no interpenetrating pair
(the mesh evaluator), every body resting on something, upright within 5 degrees, over the support.
No model sees it. Every task has a reference program that passes it (`reference` in `tasks.py`),
shown to no model: the tasks are feasible.

One check was corrected after the first model of the first run. "Centres within 1 cm of a common
line" was measured during the runs as each centre's distance from the centres' mean, which asks for
more than the request says; it is now half the range of the centres, their distance from the best
common line (`rubric.line_offset`). The correction can only turn a failure into a pass, and only for
a round whose failed checks were all line checks: `regrade.py` rebuilds exactly those rounds' scenes
from the recorded answers (no model is called; the pipelines are deterministic) and grades them
again, and each rebuilt scene must reproduce the outcome of every other check. The first run's
tables give the verdicts as run and as regraded (`first_run/<model>/regrade.json` lists every
changed trial); the replay below was graded with the corrected check.

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
  each body is set down from above, upright, on the surface under its footprint (as the repair
  seats a body last, raised clear of a residual of the drop), and the verifier reports the
  pairs that interpenetrate, the bodies that rest on nothing, leave the support or were not given.

A trial has up to three rounds. Each round is a new isolated call whose message repeats the request
with the previous answer and that condition's report; a trial ends when its own check passes. The
rubric grades the scene of every round.

## Costs and timing

Tokens and cost are the CLI's own accounting of each call (`usage`, `total_cost_usd`): output tokens
include the model's thinking, prompt tokens are split into those read from the prompt cache and the
rest. Model time is the call's duration as the CLI reports it. Solver time is the wall-clock time of
the pipeline after the answer (validation, compilation, S4R and verification for a program; seating
and verification for poses), with the worker's CPU seconds recorded beside it. Each run used six
worker processes, one thread each, on eight physical cores reserved for it (no SMT siblings in use);
models ran one after another, three seeds per task and condition, the trials submitted in one fixed
order.

## The first run and the replay

The comparison was first run on 2026-10-24 (`results/dsl_vs_direct/first_run/`) and replayed on
2026-10-25 with the current solver: seating through a grid of the meshes' triangles, contacts without
a new BVH for each scale of the continuation, the contact witness over all of FCL's contacts, a body
centred over another separated along their axis of least overlap, a program that stops short at the
shrunken scale solved again from other starts, the contacts' margin yielding to the requirements,
and a body recognised as resting on a surface the sampled heights miss.

Typed poses are set down as the repair sets a body down: from above what the body rests on, and
seated clear of it at the end. The first run set them down from their starting height, so a carton
put on a taller carton, or a fruit on a fruit, met the inside of it (every typed tower, 9 / 9, though
the answers stacked them as asked), and a drop residual of a fraction of a millimetre counted as
interpenetration; its results for typed towers, pallets and fruit set on fruit carry this. The
ablation that repairs typed poses with the shared statements keeps the supports the answer named (the
first run set every body on the table). What a model receives is unchanged: the system prompts, the
formats and the first message of every trial (the request and the scene) are the same, byte for byte.

The replay (`run.py --replay`) runs every trial again, keeping the recorded answer for each round
whose message is the recorded one (the first round always; a later round when the answer before it
was the recorded one and checking it found word for word what it found then), and calls the model, as
a run does, for any other round: where the current report differs from the recorded one, or where the
recorded trial had stopped and the current check does not pass. It ran in two passes, the second
replaying the first after the seating of typed poses was updated (the first pass's records are not
kept; each `run_meta.json` names the commits of the replay, of the pass it replays and of the first
run). The passes called the model for 59 and 3 rounds; of the replay's 469 rounds, 446 keep the first
run's answer and 23 hold an answer of such a call (`answer_from` in each round), and each answers
exactly the message that the final code sends. The first answer of every trial is the first run's, so
the two runs share their samples. The results below are the replay's, and
`results/dsl_vs_direct/replay_diff.md` lists every trial whose verdict changed.

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

Every number below comes from the trial files of the replay (`results/dsl_vs_direct/<model>/trials/`);
`analyze.py` writes every table, per task and per scene size, to
[`results/dsl_vs_direct/summary.md`](../results/dsl_vs_direct/summary.md), and the same tables of
the first run are in `results/dsl_vs_direct/first_run/summary.md`. Pass rates carry 95 % Wilson
intervals; a condition's time per passing scene is its total over its passing trials.

![pass rate, output tokens and seconds per passing scene](media/dsl_vs_direct.png)

### Pass rates and costs (sixteen tasks, 48 trials per answer and model)

| model | answer | pass (95% CI) | round 1 | rounds | output tokens / trial | prompt tokens / trial (not cached) | cost / trial | model s / trial | solver s / trial | s / passing scene |
|---|---|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 | program | 47/48 (89-100%) | 98% | 1.08 | 4,020 | 13,241 (1,600) | $0.0024 | 19 | 2.8 | 22 |
|  | program, earlier statements | 38/48 (66-88%) | 73% | 1.12 | 6,960 | 8,914 (3,238) | $0.0042 | 31 | 3.1 | 43 |
|  | poses | 43/48 (78-95%) | 81% | 1.12 | 14,374 | 2,919 (1,352) | $0.0075 | 62 | 0.9 | 70 |
|  | poses, repaired | 37/48 (63-87%) | 77% | 1.12 | 14,374 | 2,919 (1,352) | $0.0075 | 62 | 3.1 | 84 |
| Sonnet 5.5 | program | 48/48 (93-100%) | 94% | 1.06 | 620 | 12,990 (1,568) | $0.0136 | 5 | 3.1 | 8 |
|  | program, earlier statements | 30/48 (48-75%) | 60% | 1.06 | 1,451 | 6,689 (1,959) | $0.0228 | 10 | 2.7 | 21 |
|  | poses | 39/48 (68-90%) | 75% | 1.21 | 3,223 | 3,200 (1,764) | $0.0394 | 25 | 1.0 | 33 |
|  | poses, repaired | 37/48 (63-87%) | 75% | 1.21 | 3,223 | 3,200 (1,764) | $0.0394 | 25 | 3.2 | 37 |
| Opus 5.5 | program | 48/48 (93-100%) | 100% | 1.02 | 764 | 12,414 (1,582) | $0.0301 | 8 | 2.6 | 10 |
|  | program, earlier statements | 37/48 (63-87%) | 75% | 1.04 | 2,317 | 5,891 (1,700) | $0.0608 | 22 | 2.4 | 32 |
|  | poses | 45/48 (83-98%) | 90% | 1.04 | 2,481 | 2,661 (1,665) | $0.0631 | 25 | 0.9 | 28 |
|  | poses, repaired | 39/48 (68-90%) | 81% | 1.04 | 2,481 | 2,661 (1,665) | $0.0631 | 25 | 3.1 | 35 |

A program passed in 143 of 144 trials and typed poses in 127. The one failing program (Haiku, ten
drinks) placed one can at x = 0.75 within 4 cm instead of the row; the solver, which moves each body
as little as the statements allow, left that can at 0.71 and the others at 0.69, and the row's line
at 0.693, short of the band from 0.70 to 0.80 the request named. Programs passed at the first round
in 94-100 % of trials and needed 1.02-1.08 rounds on average; poses passed at the first round in
75-90 % and needed 1.04-1.21. Programs took 3.2 to 5.2 times fewer output tokens per trial and cost
2.1 to 3.1 times less, and a passing scene took 2.7 to 4.0 times less time, the solver's 2.6-3.1 s
per trial included (its wall-clock and CPU seconds agree to 0.1 s: the cores were the run's alone).
The language makes the system prompt longer, 12-13k prompt tokens per trial against 3k, but that
part is read from the prompt cache: a program trial sent about as many tokens fresh (1.6k) as a pose
trial (1.4-1.8k).

### Where each answer fails

- Poses: the order of height in the row of ten drinks (8 / 9, and 1 / 9 of six: the scene summary
  gives sizes to the centimetre, the compiler sorts on the meshes); 1 cm between the six cartons on
  the pallet (2 / 9, at 5-6 mm); interpenetration in the square of ten fruit (2 / 9, by 0.5-0.6 mm);
  a group not clustered (2), the mug's handle (1), and a tower too far from the robot (1). Every
  typed tower stacked the three cartons centred, as every program did.
- Programs restricted to the earlier statements: no region bounded by coordinates (15 failed square
  containments in the packing tasks), no sorting (all nine rows of ten drinks), no handle turned to
  the robot (setting, desk).
- Programs: the one row above.

### Held-out tasks

| model | answer | development tasks (42 trials) | held-out tasks (6 trials) | held out: output tokens / trial | held out: s / passing scene |
|---|---|---|---|---|---|
| Haiku 5.5 | program | 41/42 | 6/6 | 4,689 | 26 |
|  | program, earlier statements | 33/42 | 5/6 | 7,249 | 43 |
|  | poses | 37/42 | 6/6 | 3,052 | 15 |
| Sonnet 5.5 | program | 42/42 | 6/6 | 1,260 | 16 |
|  | program, earlier statements | 26/42 | 4/6 | 2,079 | 29 |
|  | poses | 33/42 | 6/6 | 957 | 10 |
| Opus 5.5 | program | 42/42 | 6/6 | 897 | 12 |
|  | program, earlier statements | 31/42 | 6/6 | 1,818 | 21 |
|  | poses | 39/42 | 6/6 | 1,113 | 14 |

The two tasks written after the language was frozen passed in every trial, with programs and with
poses, for every model. They are small (six and seven objects) and need no geometry the model cannot
read off the summary; poses were the cheaper and the faster answer there for Haiku and Sonnet (10-16 s
per passing scene against 16-26 s) and about as fast as programs for Opus (14 s against 12 s), as the
solver took 4-8 s per trial on the tools' and the desk's meshes (up to 92k faces each).

### Scene size

| model | objects | program: pass, output tokens / trial, model s / trial | poses: pass, output tokens / trial, model s / trial |
|---|---|---|---|
| Haiku 5.5 | 10 | 3/3, 3,516, 17 | 3/3, 6,616, 28 |
|  | 16 | 3/3, 5,455, 25 | 3/3, 23,038, 99 |
|  | 24 | 3/3, 4,898, 22 | 3/3, 29,007, 122 |
| Sonnet 5.5 | 10 | 3/3, 351, 3 | 3/3, 1,753, 14 |
|  | 16 | 3/3, 413, 3 | 3/3, 4,737, 39 |
|  | 24 | 3/3, 975, 7 | 2/3, 5,875, 47 |
| Opus 5.5 | 10 | 3/3, 821, 8 | 3/3, 1,208, 12 |
|  | 16 | 3/3, 955, 10 | 3/3, 4,754, 46 |
|  | 24 | 3/3, 1,015, 10 | 3/3, 5,283, 53 |

Sorting a heap of 10, 16 and 24 objects: a program's output barely grows (patterns name the groups,
the solver places every body), typed poses grow with every object.

![output tokens and model seconds per trial against the number of objects](media/dsl_vs_direct_scaling.png)

### Physics and reuse

| model | answer | pass | settles | passes and settles | new starts pass | spread of the layouts | other assets pass |
|---|---|---|---|---|---|---|---|
| Haiku 5.5 | program | 47/48 | 48/48 | 47/48 | 235 / 235 | 8.8 cm | 47 / 47 |
|  | program, earlier statements | 38/48 | 47/48 | 37/48 | — | — | — |
|  | poses | 43/48 | 47/48 | 42/48 | 200 / 215 | 0.0 cm | 28 / 43 |
| Sonnet 5.5 | program | 48/48 | 48/48 | 48/48 | 240 / 240 | 12.9 cm | 48 / 48 |
|  | program, earlier statements | 30/48 | 48/48 | 30/48 | — | — | — |
|  | poses | 39/48 | 47/48 | 38/48 | 180 / 195 | 0.0 cm | 22 / 39 |
| Opus 5.5 | program | 48/48 | 48/48 | 48/48 | 239 / 240 | 13.9 cm | 48 / 48 |
|  | program, earlier statements | 37/48 | 48/48 | 37/48 | — | — | — |
|  | poses | 45/48 | 48/48 | 45/48 | 210 / 225 | 0.0 cm | 25 / 45 |

Every scene a program produced stays at rest in MuJoCo (CoACD pieces, the assets' friction and
mass). Two passing scenes of typed poses do not: each sets a fruit on another fruit (an answer may
name the body a pose rests on), which the rubric's resting check accepts and MuJoCo rolls off. Reuse
applies each passing answer, with no further call, to five new starting heaps of its task and to the
task's scene with every object replaced by another package of its category. A program is solved
again from each start and passes 714 of 715 starts (the one failure, a pallet of four, laid its
cartons in one row), its bodies 9 to 14 cm from where they stood for another start (it moves each
body as little as the statements allow). Typed poses are one layout and pass 92-93 % of the new
starts: they fail only where the request depends on the start (nothing moved more than 3 cm). With
other assets, programs pass every scene and typed poses 56-65 %: poses typed for one geometry
interpenetrate or fall out of order with another.

Poses repaired by S4R with only the statements every task shares, and the supports the answer
named (`direct+repair` in the first table), pass 37, 37 and 39 of 48, fewer than the poses
themselves: the repair removes interpenetration, so one more pallet passes, but it does not know the
request: it pushed packed fruit out of their square (9 trials), moved stacked cartons off each
other's centre (4), and left unsupported the fruit that two answers set on other fruit (2).

### Limits

- The language was revised on pilots over the fourteen development tasks; the held-out pair, the
  only tasks the revisions never saw, is small.
- One check of the rubric was corrected after the first model's run of the first run (above); every
  verdict of that run it could change was regraded from the recorded answers, and both verdicts are
  published. The replay was graded with the corrected check.
- The rubric's resting check is geometric: it accepts a fruit balanced on another, which the settle
  test reports.
- Costs are the CLI's accounting at list price. The CLI adds its account and environment block
  (about 680 tokens) to every call, the same in every condition.
- Three seeds per task and answer: the intervals of a single task are wide; the pooled rates are
  tighter.
- Solver seconds depend on the machine: eight physical cores, one worker thread each.

## Reproducing

The calls need the `claude` CLI signed in (2.1.310 in the recorded runs); a trial makes one to three
calls. The recorded runs were made at the commits their `run_meta.json` names; a replay stops when a
condition's system prompt or answer format is no longer the recorded one. The packages come from `assets.json`, fetched at its pinned commit (`regrade.py` grades a run
made before the line check's correction again, as it graded the first run):

```
export SIMREADY_ASSETS_DIR=/path/to/simready-assets SIMREADY_CACHE_DIR=/path/to/cache OMP_NUM_THREADS=1
python -m simready.cli assets fetch --revision 524747d9b5e1f21b5b18b56041788c01d745f1a1 $(python -c "import json; \
  d = json.load(open('experiments/dsl_vs_direct/assets.json')); \
  print(' '.join(sorted({p for v in d['pool'].values() for p in v} | {d['table'], d['pallet']})))")
python experiments/dsl_vs_direct/isolation_check.py --model claude-sonnet-5-5 --out runs
for M in claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5; do
  python experiments/dsl_vs_direct/run.py --model $M --seeds 0 1 2 --workers 6 --out runs/$M
  # or replay a recorded run, calling the model only for a round whose message changed:
  #   python experiments/dsl_vs_direct/run.py --replay results/dsl_vs_direct/first_run/$M --workers 6 --out runs/$M
  python experiments/dsl_vs_direct/finals.py runs/$M --draw setting_n6 drinks_row_n10 fruit_cans_n24 stack_n6 tool_layout_n7
  python experiments/dsl_vs_direct/reuse.py runs/$M --starts 5     # or in slices: --first K --count 16, then --merge
done
python experiments/dsl_vs_direct/publish.py runs results/dsl_vs_direct
python experiments/dsl_vs_direct/analyze.py results/dsl_vs_direct/claude-* --md results/dsl_vs_direct/summary.md --json results/dsl_vs_direct/summary.json
python experiments/dsl_vs_direct/replay_diff.py $(for M in claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5; do \
  echo results/dsl_vs_direct/first_run/$M results/dsl_vs_direct/$M; done) --md results/dsl_vs_direct/replay_diff.md --json results/dsl_vs_direct/replay_diff.json
python experiments/dsl_vs_direct/figures.py results/dsl_vs_direct/summary.json docs/media/dsl_vs_direct.png docs/media/dsl_vs_direct_scaling.png
```
