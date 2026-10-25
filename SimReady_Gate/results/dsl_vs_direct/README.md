# Program or poses: recorded runs

Written by `experiments/dsl_vs_direct/` (see `docs/EVAL_DSL_vs_direct.md`): one run per model,
sixteen tasks (two held out), three conditions, three seeds. The run at the top level is the replay
of 2026-10-25, with the current solver and seating of typed poses; `first_run/` is the
run of 2026-10-24 that it replays, with the same layout of files (and `regrade.json`, the verdicts
of the corrected line check).

```
isolation_check.json           the leak test run before the comparison (canaries planted, isolated call vs control);
                               what each launch showed of the canaries, not its answer (the control's quotes the
                               configuration of the account that ran it)
summary.md, summary.json       the tables (analyze.py over every model's run)
replay_diff.md, replay_diff.json
                               the replay against the first run, per model and condition: passes in each, the trials
                               that changed, the rounds that kept the first run's answer and those that called the
                               model, the solver's seconds (replay_diff.py)
<model>/
  run_meta.json                the commit the harness ran from, the CLI version, model, effort, rounds, seeds,
                               flags, the environment allowlist, the asset revision, tasks, CPUs and workers;
                               in a replay, the run it replays
  system_<condition>.txt       the fixed system prompt of each condition
  schema_<condition>.json      the structured-output format of each condition
  trials/<task>_<condition>_s<seed>.json
                               every round of a trial: the model's answer, its token usage, cost and timing,
                               the condition's own report, the rubric's verdict and failed checks, the
                               solver's wall-clock and CPU seconds; in a replay, where each answer came from
                               (answer_from: "recorded", the first run's; "called", a call of the replay)
  finals.json                  the final scene of every trial settled in MuJoCo with CoACD pieces (finals.py)
  reuse.json                   passing answers applied to new starting heaps and to other assets (reuse.py)
  figures/<task>_<condition>.png
                               top views of the seed-0 final scenes of some tasks
```

Layouts are not stored: `scenes.build(task, seed)` rebuilds the starting scene of any trial from
`assets.json` (pinned dataset commit) and the seed. Addresses and local directories are removed
from the recorded files. The pilots that led to the language's last revisions are not here; the
held-out tasks were run only in these runs. The replay ran in two passes (`docs/EVAL_DSL_vs_direct.md`,
"The first run and the replay"); the second kept the first's answers where the message was unchanged,
and only the second is here.
