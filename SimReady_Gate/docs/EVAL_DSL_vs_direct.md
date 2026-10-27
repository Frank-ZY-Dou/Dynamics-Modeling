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

## After a failure: two reports compared

`dsl_v1_rewrites` is `dsl_v1` with more in the report of a failed program: the rewrites of each
failed statement (the nearest values of its bounds at which it holds on the layout returned, and
whether it may be weakened) and what the repair ends with from the same start at a smaller margin
([`docs/DSL.md`](DSL.md#when-a-requirement-fails)). Its first answer is dsl_v1's for the same task
and seed, so the two differ only after that answer fails, and each such trial is a pair; pairs are
compared with the exact McNemar test.

A model's first answer seldom fails, so the reports are also compared on faulted first answers. A
fault is one edit of a task's reference program (`tasks.py`, `faults`), of one of four kinds, each
with the recovery the request asks for:

| kind | the edit | the recovery the request asks for |
|---|---|---|
| tight | a bound the program chose, tightened past what the scene allows | relax it within what the request allows |
| target | a `place` target the request contradicts | drop it |
| margin | a schedule margin that leaves the contacts no room | lower it |
| conflict | a statement the program chose that contradicts a number the request states | change that statement: relaxing the stated number breaks the request |

`run.py --check-faults` runs every reference and every faulted program without a call, on three
seeds (`faults_check.json`). A fault takes part where the reference passes the gate and the rubric
and the faulted program fails the gate. A trial then goes on as any other from its second round:
the request, the faulted program as the previous answer and the condition's report, at most two
calls. The rubric decides; a final program that passes the gate but not the rubric has weakened the
request.

## Costs and timing

Tokens and cost are the CLI's own accounting of each call (`usage`, `total_cost_usd`): output tokens
include the model's thinking, prompt tokens are split into those read from the prompt cache and the
rest. Model time is the call's duration as the CLI reports it. Solver time is the wall-clock time of
the pipeline after the answer (validation, compilation, S4R and verification for a program; seating
and verification for poses), with the worker's CPU seconds recorded beside it. Each run used six
worker processes, one thread each, on eight physical cores reserved for it (no SMT siblings in use; in the run of
2026-10-26 two siblings carried another job for 13 minutes, and the 201 fault trials that overlap them were timed
again from their recorded answers on the idle cores: 1.004 times the recorded solver time, which is kept);
models ran one after another, three seeds per task and condition, the trials submitted in one fixed
order.

## The runs

The comparison was first run on 2026-10-24 (`results/dsl_vs_direct/first_run/`) and replayed on
2026-10-25 with the updated solver, keeping every recorded answer whose message was unchanged
(`results/dsl_vs_direct/replay_2026-10-25/`, with `replay_diff.md`). The language then changed: a
written semantics, the solver's margin moved into the program's `schedule`, three statements
renamed, and the rewrites of failed requirements in the report ([`docs/DSL.md`](DSL.md)). The system
prompts changed with it, so the earlier answers answer another prompt: the comparison was run afresh
on 2026-10-26 with every condition, together with the faulted first answers. The results below are
that run's.

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

Every number below comes from the trial files of the run of 2026-10-26
(`results/dsl_vs_direct/<model>/trials/`, the faulted ones in `results/dsl_vs_direct/faults/`);
`analyze.py` writes every table, per task and per scene size, to
[`results/dsl_vs_direct/summary.md`](../results/dsl_vs_direct/summary.md), and the tables of the
earlier runs are in `replay_2026-10-25/summary.md` and `first_run/summary.md`. Pass rates carry 95 %
Wilson intervals; a condition's time per passing scene is its total over its passing trials.

![pass rate, output tokens and seconds per passing scene](media/dsl_vs_direct.png)

### Pass rates and costs (sixteen tasks, 48 trials per answer and model)

| model | answer | pass (95% CI) | round 1 | rounds | output tokens / trial | prompt tokens / trial (not cached) | cost / trial | model s / trial | solver s / trial | s / passing scene |
|---|---|---|---|---|---|---|---|---|---|---|
| Haiku 5.5 | program | 45/48 (83-98%) | 90% | 1.19 | 6,761 | 17,519 (3,130) | $0.0041 | 31 | 2.8 | 36 |
|  | program, rewrites in the report | 48/48 (93-100%) | 90% | 1.17 | 6,446 | 17,575 (3,062) | $0.0040 | 29 | 4.0 | 33 |
|  | program, earlier statements | 37/48 (63-87%) | 69% | 1.10 | 8,834 | 10,734 (4,691) | $0.0054 | 39 | 2.2 | 54 |
|  | poses | 42/48 (75-94%) | 81% | 1.15 | 12,474 | 4,082 (3,011) | $0.0068 | 54 | 0.9 | 63 |
|  | poses, repaired | 37/48 (63-87%) | 77% | 1.15 | 12,474 | 4,082 (3,011) | $0.0068 | 54 | 2.7 | 74 |
| Sonnet 5.5 | program | 48/48 (93-100%) | 90% | 1.12 | 812 | 15,533 (2,493) | $0.0194 | 7 | 2.6 | 10 |
|  | program, rewrites in the report | 48/48 (93-100%) | 90% | 1.15 | 859 | 15,922 (2,595) | $0.0203 | 8 | 5.4 | 13 |
|  | program, earlier statements | 31/48 (50-77%) | 50% | 1.19 | 1,619 | 7,231 (2,307) | $0.0259 | 13 | 2.2 | 24 |
|  | poses | 41/48 (73-93%) | 79% | 1.10 | 2,134 | 2,923 (1,934) | $0.0292 | 18 | 0.8 | 22 |
|  | poses, repaired | 36/48 (61-85%) | 75% | 1.10 | 2,134 | 2,923 (1,934) | $0.0292 | 18 | 2.8 | 28 |
| Opus 5.5 | program | 48/48 (93-100%) | 100% | 1.00 | 795 | 13,749 (2,213) | $0.0359 | 8 | 2.2 | 10 |
|  | program, rewrites in the report | 48/48 (93-100%) | 100% | 1.00 | 795 | 13,749 (2,213) | $0.0359 | 8 | 2.6 | 10 |
|  | program, earlier statements | 38/48 (66-88%) | 69% | 1.12 | 2,463 | 7,016 (2,361) | $0.0691 | 23 | 2.1 | 32 |
|  | poses | 45/48 (83-98%) | 88% | 1.06 | 3,122 | 2,813 (1,881) | $0.0777 | 30 | 0.8 | 33 |
|  | poses, repaired | 42/48 (75-94%) | 85% | 1.06 | 3,122 | 2,813 (1,881) | $0.0777 | 30 | 2.6 | 38 |

A program passed in 141 of 144 trials and typed poses in 128. The three failing programs (Haiku,
packing ten and fourteen fruit) left fruit outside the square the request names; told the rewrites of
their failed statements, the same first answers passed (`program, rewrites in the report`, 144 of
144). Programs passed at the first round in 90-100 % of trials and needed 1.00-1.19 rounds on
average; poses passed at the first round in 79-88 % and needed 1.06-1.15. Programs took 1.8 to 3.9
times fewer output tokens per trial and cost 1.5 to 2.2 times less, and a passing scene took 1.7 to
3.3 times less time, the solver's 2.2-2.8 s per trial included. The language makes the system prompt
longer, 14-18k prompt tokens per trial against 3-4k, but that part is read from the prompt cache: a
program trial sent 2.2-3.1k tokens fresh, a pose trial 1.9-3.0k.

### Where each answer fails

- Poses: the order of height in the row of ten drinks (7 / 9, and 1 / 9 of six: the scene summary
  gives sizes to the centimetre, the compiler sorts on the meshes); interpenetration in the packing
  tasks (3); the mug's handle (2); a group not clustered (1), two objects beyond the robot's reach
  (1) and the gaps of a row of tools (1).
- Programs restricted to the earlier statements: no region bounded by coordinates (the squares of
  the packing tasks), no sorting (the rows of ten drinks), no handle turned to the robot.
- Programs: the three packing trials above.

### Held-out tasks

| model | answer | development tasks (42 trials) | held-out tasks (6 trials) | held out: output tokens / trial | held out: s / passing scene |
|---|---|---|---|---|---|
| Haiku 5.5 | program | 39/42 | 6/6 | 4,622 | 25 |
|  | program, earlier statements | 31/42 | 6/6 | 6,923 | 34 |
|  | poses | 36/42 | 6/6 | 3,075 | 15 |
| Sonnet 5.5 | program | 42/42 | 6/6 | 1,783 | 22 |
|  | program, earlier statements | 26/42 | 5/6 | 1,519 | 18 |
|  | poses | 36/42 | 5/6 | 974 | 13 |
| Opus 5.5 | program | 42/42 | 6/6 | 926 | 12 |
|  | program, earlier statements | 32/42 | 6/6 | 2,186 | 23 |
|  | poses | 39/42 | 6/6 | 1,134 | 14 |

The two tasks written after the language was frozen passed in every trial with programs, for every
model; poses failed one of Sonnet's (the gaps of the row of tools). They are small (six and seven
objects); poses were the cheaper and the faster answer there for Haiku and Sonnet (13-15 s per
passing scene against 22-25 s) and about as fast as programs for Opus (14 s against 12 s).

### Scene size

| model | objects | program: pass, output tokens / trial, model s / trial | poses: pass, output tokens / trial, model s / trial |
|---|---|---|---|
| Haiku 5.5 | 10 | 3/3, 5,651, 26 | 3/3, 6,063, 27 |
|  | 16 | 3/3, 4,644, 21 | 3/3, 24,315, 107 |
|  | 24 | 3/3, 6,659, 30 | 3/3, 23,577, 102 |
| Sonnet 5.5 | 10 | 3/3, 400, 4 | 3/3, 1,267, 11 |
|  | 16 | 3/3, 880, 7 | 2/3, 5,017, 41 |
|  | 24 | 3/3, 1,011, 8 | 3/3, 4,243, 34 |
| Opus 5.5 | 10 | 3/3, 875, 11 | 3/3, 2,088, 20 |
|  | 16 | 3/3, 1,073, 10 | 3/3, 6,472, 62 |
|  | 24 | 3/3, 1,111, 10 | 3/3, 6,231, 62 |

Sorting a heap of 10, 16 and 24 objects: a program stays within 400 to 1,100 output tokens for
Sonnet and Opus and 4,600 to 6,700 for Haiku at every size (patterns name the groups, the solver
places every body); typed poses grow three- to fourfold from 10 to 16 objects.

![output tokens and model seconds per trial against the number of objects](media/dsl_vs_direct_scaling.png)

### Physics and reuse

| model | answer | pass | settles | passes and settles | new starts pass | spread of the layouts | other assets pass |
|---|---|---|---|---|---|---|---|
| Haiku 5.5 | program | 45/48 | 48/48 | 45/48 | 222 / 225 | 10.0 cm | 45 / 45 |
|  | program, earlier statements | 37/48 | 48/48 | 37/48 | — | — | — |
|  | poses | 42/48 | 48/48 | 42/48 | 195 / 210 | 0.0 cm | 26 / 42 |
| Sonnet 5.5 | program | 48/48 | 48/48 | 48/48 | 239 / 240 | 13.0 cm | 47 / 48 |
|  | program, earlier statements | 31/48 | 48/48 | 31/48 | — | — | — |
|  | poses | 41/48 | 48/48 | 41/48 | 190 / 205 | 0.0 cm | 23 / 41 |
| Opus 5.5 | program | 48/48 | 48/48 | 48/48 | 236 / 240 | 13.8 cm | 48 / 48 |
|  | program, earlier statements | 38/48 | 48/48 | 38/48 | — | — | — |
|  | poses | 45/48 | 48/48 | 45/48 | 210 / 225 | 0.0 cm | 30 / 45 |

Every scene of every answer stays at rest in MuJoCo (CoACD pieces, the assets' friction and mass).
Reuse applies each passing answer, with no further call, to five new starting heaps of its task and
to the task's scene with every object replaced by another package of its category. A program is
solved again from each start and passes 697 of 705 starts, its bodies 10 to 14 cm from where they
stood for another start (it moves each body as little as the statements allow); typed poses are one
layout and pass 595 of 640. With other assets, programs pass 140 of 141 scenes and typed poses 79 of
128: poses typed for one geometry interpenetrate or fall out of order with another.

Poses repaired by S4R with only the statements every task shares, and the supports the answer named
(`poses, repaired`), pass 37, 36 and 42 of 48, fewer than the poses themselves: the repair removes
interpenetration but does not know the request, and it pushed packed fruit out of their square (14
trials) and a knife and a spoon out of their gap (3).

### After a failure

| model | faults (pairs) | pass: plain / rewrites | gate passes at round 2 | request weakened | output tokens after round 1 | plain only / rewrites only | p |
|---|---|---|---|---|---|---|---|
| Haiku 5.5 | all (100) | 97 / 100 | 91 / 95 | 2 / 0 | 6,646 / 5,191 | 0 / 3 | 0.25 |
|  | tight (40) | 40 / 40 | 36 / 37 | 0 / 0 | 5,548 / 4,917 | 0 / 0 | 1 |
|  | target (27) | 27 / 27 | 26 / 26 | 0 / 0 | 4,507 / 4,170 | 0 / 0 | 1 |
|  | margin (7) | 7 / 7 | 6 / 6 | 0 / 0 | 11,792 / 3,423 | 0 / 0 | 1 |
|  | conflict (18) | 18 / 18 | 16 / 18 | 0 / 0 | 6,817 / 4,735 | 0 / 0 | 1 |
|  | model (8) | 5 / 8 | 7 / 8 | 2 / 0 | 14,464 / 12,574 | 0 / 3 | 0.25 |
| Sonnet 5.5 | all (98) | 97 / 98 | 95 / 95 | 1 / 0 | 979 / 920 | 0 / 1 | 1 |
|  | tight (40) | 40 / 40 | 40 / 39 | 0 / 0 | 925 / 957 | 0 / 0 | 1 |
|  | target (27) | 27 / 27 | 27 / 27 | 0 / 0 | 801 / 793 | 0 / 0 | 1 |
|  | margin (7) | 7 / 7 | 6 / 6 | 0 / 0 | 1,298 / 569 | 0 / 0 | 1 |
|  | conflict (18) | 17 / 18 | 16 / 18 | 1 / 0 | 1,248 / 1,024 | 0 / 1 | 1 |
|  | model (6) | 6 / 6 | 6 / 5 | 0 / 0 | 968 / 1,341 | 0 / 0 | 1 |
| Opus 5.5 | all (92) | 92 / 92 | 91 / 92 | 0 / 0 | 1,033 / 898 | 0 / 0 | 1 |
|  | tight (40) | 40 / 40 | 40 / 40 | 0 / 0 | 886 / 918 | 0 / 0 | 1 |
|  | target (27) | 27 / 27 | 27 / 27 | 0 / 0 | 856 / 844 | 0 / 0 | 1 |
|  | margin (7) | 7 / 7 | 6 / 7 | 0 / 0 | 2,560 / 580 | 0 / 0 | 1 |
|  | conflict (18) | 18 / 18 | 18 / 18 | 0 / 0 | 1,033 / 1,061 | 0 / 0 | 1 |

The first rows of each model pool the 92 faulted first answers (34 faults on three seeds, where the
reference passes and the fault fails) and the pairs whose own first answer failed (`model`). Both
reports recover nearly every fault: with the plain report Sonnet missed one conflict and Haiku three
of its own packing answers, three of these final programs passing the gate but not the request (the
request weakened); with the rewrites every pair passed and no request was weakened. The difference is not significant at this size (3 and 1 pairs pass with the rewrites
alone; exact McNemar p = 0.25 and 1). Where the fault is the margin, the rewrites name the smaller
margin that ends the repair: the round after the failure took 2.3 to 4.4 times fewer output tokens.

### Limits

- The language was revised on pilots over the fourteen development tasks; the held-out pair, the
  only tasks the revisions never saw, is small.
- One check of the rubric was corrected after the first model's run of the first run (above); every
  verdict of that run it could change was regraded from the recorded answers, and both verdicts are
  published. The later runs were graded with the corrected check.
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
