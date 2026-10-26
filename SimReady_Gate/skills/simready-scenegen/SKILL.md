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
evaluated (report it; do not retry the same command). `check`, `repair` and `settle` print a JSON
report on stdout; `summarize`, `prompt` and `verify` print text.

## Procedure
0. To build a scene from NVIDIA SimReady assets: `python -m simready.cli assets search "coffee mug" --max-size 0.2`
   lists packages with their size, mass and license; `python -m simready.cli assets fetch <package> ...`
   downloads them (geometry and physics, checked against each package's bill of materials) into
   `SIMREADY_ASSETS_DIR`; a layout JSON names them: `{"table": {"package": ..., "position": [x, y, z], "yaw": 90},
   "objects": [{"name": "mug", "package": "dishware_mug_coffee_a01", "x": ..., "y": ..., "z": ..., "yaw": ...}]}`.
   The repair, the settle test and the export then use each asset's own friction and mass.
1. Load or build the scene:
   `python -m simready.cli summarize <scene.usda|layout.json>` prints the body list with tags and sizes
   (for free bodies also their local size, heading, mass and the sides their data names, and for a
   fixture with shelves each level's height, rectangle and headroom).
   If loading warns about unresolved assets, or `repair` reports a nonempty `dropped_children`, assets did not resolve;
   stop and report that — a certificate cannot cover a missing container.
2. Produce the program: run `python -m simready.cli prompt <scene> --request "<text>"`, answer
   the printed prompt with ONE JSON object (schema included in the prompt), save it to `program.json`.
   State what must hold and let the solver find the poses: footprint gaps with `beside`, lines with
   `row`, `align`, `evenly_spaced`, `grid`, circles with `ring`, groups with `cluster`, headings with `yaw`, `parallel`,
   `facing`, the robot's needs with `reach`, `clearance`, `approach`, zones with `within ... side=` and
   `keep_out`, small edits with `max_move`, `max_turn`, towers with `stack`, shelves with
   `on_support(a, <fixture>.level<K>)` and `flush` (edges against a level's front or back), spines or
   handles with `axis=front` / `axis=handle`. Name a group of similar
   bodies by a pattern (`members: ["can_*", "bottle_1"]`, `a: "fruit_*"`) instead of listing them, so
   none is left out. `docs/DSL.md` lists every statement with what it compiles to; `soft=` marks a
   preference.
3. Validate and compile: `python -m simready.cli check <scene> program.json` (exit 2 = fix the
   named statement in the JSON: unknown body, self-relation, missing `r`, bad keyword, inset with
   no room). `check` compiles the program exactly as `repair` will.
4. Verify and repair: `python -m simready.cli repair <scene> program.json --out <scene_repaired.usda>`
   (the `--out` extension must match the input kind: `.usda` for USD scenes, `.json` for layouts;
   a USD `--out` may be written to another directory; asset references are re-anchored). Read `pen_after`, `failed_predicates`,
   `rmsd_xy`; exit 0 means pen 0, every predicate satisfied and, when the gate sets `min_gap`,
   every pair at that clearance (`clearance.violations` lists the pairs that are not). An
   `on_support` predicate whose value is `null` means the body's bottom is over no part of that
   support at all. `solver_notes` says when a statement could not be met within a step of the continuation.
5. Route failures, never override the numbers (at most 3 rounds in total, then report failure):
   - a failed requirement: the report's `rewrites` name each failed statement (its index in the JSON
     `statements`) with `relax`, the nearest values of its own bounds at which it holds on the layout
     returned (`{"gap": [written, nearest]}`), and `weaken` (`soft`: it may become a preference;
     `drop`: it may be removed). `accepted_if_relaxed` / `accepted_if_weakened` say whether those
     edits make that layout pass. Apply a rewrite only when the request allows it (a number the
     request states is not yours to relax); otherwise change what the statement asks for, or
     another statement, and rerun step 4 (the repair starts again from the same scene).
   - a tight packing whose `solver_notes` say the contacts set the program rows aside: lower the
     clearance the repair keeps, `"schedule": {"margin": 0.002}` (default 0.005 m); it changes how
     the repair gets there, never what is accepted.
   - `pen_after > 0` (tight packing, nested containers): shrink the intent — fewer objects,
     a larger region, or different objects — and go back to step 2.
   - a failed `within`/`inside`: the region is too small for the objects; enlarge the region or
     drop an object. Positions are not statements — do not try to "move" a body by editing.
   - `check` refusing a level ("leaves ... m under the surface above it"): the body does not fit
     there; put it on one of the levels the message names.
   - a failed `keep_out` or `approach` next to a `grid` or a `row`: the side was chosen locally; add
     `within(..., side=)` for the structure, or a `place` target, so that it starts on the free side.
   - a failed preference (`failed_preferences`) does not fail the repair; report it.
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
