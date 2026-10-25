# Program or poses: recorded runs

Written by `experiments/dsl_vs_direct/` (see `docs/EVAL_DSL_vs_direct.md`): one run per model,
sixteen tasks (two held out), three conditions, three seeds.

```
isolation_check.json           the leak test run before the comparison (canaries planted, isolated call vs control)
summary.md, summary.json       the tables (analyze.py over every model's run)
<model>/
  run_meta.json                the commit the harness ran from, the CLI version, model, effort, rounds, seeds,
                               flags, the environment allowlist, the asset revision, tasks, CPUs and workers
  system_<condition>.txt       the fixed system prompt of each condition
  schema_<condition>.json      the structured-output format of each condition
  trials/<task>_<condition>_s<seed>.json
                               every round of a trial: the model's answer, its token usage, cost and timing,
                               the condition's own report, the rubric's verdict and failed checks, the
                               solver's wall-clock and CPU seconds
  finals.json                  the final scene of every trial settled in MuJoCo with CoACD pieces (finals.py)
  reuse.json                   passing answers applied to new starting heaps and to other assets (reuse.py)
  figures/<task>_<condition>.png
                               top views of the seed-0 final scenes of some tasks
```

Layouts are not stored: `scenes.build(task, seed)` rebuilds the starting scene of any trial from
`assets.json` (pinned dataset commit) and the seed. Addresses and local directories are removed
from the recorded files. The pilots that led to the language's last revisions are not here; the
held-out tasks were run only in these runs.
