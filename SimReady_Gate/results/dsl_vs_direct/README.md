# Program or poses: recorded runs

Written by `experiments/dsl_vs_direct/` (see `docs/EVAL_DSL_vs_direct.md`).

```
isolation_check.json           the leak test run before the comparison (canaries planted, isolated call vs control)
summary.md, summary.json       the tables (analyze.py over every model's run)
<model>/
  run_meta.json                CLI version, model, effort, rounds, seeds, flags, the environment allowlist
  system_<condition>.txt       the fixed system prompt of each condition
  schema_<condition>.json      the structured-output format of each condition
  trials/<task>_<condition>_s<seed>.json
                               every round of a trial: the model's answer, its token usage, cost and timing,
                               the condition's own report, the rubric's verdict and failed checks
  finals.json                  the final scene of every trial settled in MuJoCo (finals.py)
  reuse.json                   passing answers applied to new starting heaps and to other assets (reuse.py)
```

Layouts are not stored: `scenes.build(task, seed)` rebuilds the starting scene of any trial from
`assets.json` (pinned dataset commit) and the seed. Addresses and local directories are removed
from the recorded files.
