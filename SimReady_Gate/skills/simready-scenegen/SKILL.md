---
name: simready-scenegen
description: Turn a layout request into a verified, penetration-free scene. The agent writes the DSL program (text2function); the gates and the S4R repair are deterministic tools.
---

# simready-scenegen

You are the text2function step of the SimReady Gate. You never judge geometry yourself:
you write a constraint program, then call the tools and read their numbers.

Exit codes of every tool: 0 = the requested outcome holds, 1 = it does not (read the JSON),
2 = the program or an input is invalid (fix the program: every number must be finite, every field
must belong to its statement, every gate line must parse) or a geometric query could not be
evaluated (report it; do not retry the same command). Reports are JSON on stdout.

## Procedure
1. Load or build the scene:
   `python -m simready.cli summarize <scene.usda|layout.json>` prints the body list with tags and sizes.
   If the summary lists `dropped children`, the scene references assets that did not resolve;
   stop and report that — a certificate cannot cover a missing container.
2. Produce the program: run `python -m simready.cli prompt <scene> --request "<text>"`, answer
   the printed prompt with ONE JSON object (schema included in the prompt), save it to `program.json`.
3. Validate and compile: `python -m simready.cli check <scene> program.json` (exit 2 = fix the
   named statement in the JSON: unknown body, self-relation, missing `r`, bad keyword, inset with
   no room). `check` compiles the program exactly as `repair` will.
4. Verify and repair: `python -m simready.cli repair <scene> program.json --out <scene_repaired.usda>`
   (the `--out` extension must match the input kind: `.usda` for USD scenes, `.json` for layouts;
   a USD `--out` must sit in the scene's directory). Read `pen_after`, `failed_predicates`,
   `rmsd_xy`; exit 0 means pen 0, every predicate satisfied and, when the gate sets `min_gap`,
   every pair at that clearance (`clearance.violations` lists the pairs that are not). An
   `on_support` predicate whose value is `null` means the body's bottom is over no part of that
   support at all. `solver_notes` says when a statement could not be met within a step of the continuation.
5. Route failures, never override the numbers (at most 3 rounds in total, then report failure):
   - `pen_after > 0` (tight packing, nested containers): shrink the intent — fewer objects,
     a larger region, or different objects — and go back to step 2.
   - a failed relational predicate (`left_of`, `near`, ...): relax `gap`/`r` or change the pair
     in the JSON and rerun step 4.
   - a failed `within`/`inside`: the region is too small for the objects; enlarge the region or
     drop an object. Positions are not statements — do not try to "move" a body by editing.
6. Certify by settling (G5): `python -m simready.cli settle <scene_repaired.usda> --program program.json`.
   It simulates the repaired scene in MuJoCo with the same proxies a physics engine would use and
   prints `peak_speed`, `peak_disp`, `left_support` (bodies that rolled or tipped off their support)
   and `free_fall_bound_m_s` (the speed a plain drop from the largest hover could reach); the
   thresholds come from the program's `gate` section when present.
   - `left_support` names a body: it was placed too close to the support edge for its shape. Add
     `{"op": "within", "a": "<body>", "b": "<support>.top", "inset": 0.06}` (0.10 for round objects)
     and rerun step 4 (this counts as one of the 3 rounds).
   - `faster_than_free_fall` true with `left_support` empty: an unstable stack or a body resting on
     an edge; simplify that placement and rerun step 4.
   `settle` writes its report into the repaired scene's certificate only when that scene, the
   program and every asset file are the ones the repair used (their hashes match), and then sets
   `ready` in it. Its `after_settle` block lists the predicates that no longer hold on the settled
   poses (a body that tipped over fails `upright`); pass `--hold-predicates` when the task needs
   the layout's intent to survive the settle, and read `final_tilt_deg` per body either way.
7. Report the certificate path the tool prints; the scene is ready only when that certificate
   says `ready: true`.
8. When the scene is to be simulated elsewhere: `python -m simready.cli export <scene_repaired> --out <dir> --program program.json`
   writes meshes, a manifest, `scene.xml` (MuJoCo, Genesis) and `scene.usda` (Isaac Sim); the
   runners in `examples/` load that directory. Report the export directory with the certificate.
