# The SimReady language

A program says what must hold in a scene; the repair finds the poses. Every statement compiles to
rows of the quadratic program S4R solves at each step of its scale continuation, and to a predicate
that the final scene is checked against and that the certificate records. This note lists the
statements, what each compiles to, and recipes for the layouts robotics pipelines ask for.
`simready/dsl/model.py` (text), `simready/dsl/schema.py` (JSON, the prompt) and
`simready/dsl/compile.py`, `simready/dsl/statements.py` (rows and predicates) are the
implementation; `tests/test_dsl_v1.py` checks every linearised row against finite differences of
its own residual.

## Frame

Metres and degrees, z up. The robot base is at the origin and looks along +x (RoboLab's frame):
+x is away from the robot, -x toward it, +y left, -y right. `near` and `far` always mean toward and
away from the robot: regions name their sides `near` (-x), `far` (+x), `left` (+y) and `right` (-y),
and `farther(a, b)` puts a farther from the robot than b (RoboLab's front-of). A body's local frame
is the frame its asset rests in; `yaw`, `facing` and `parallel` are about its local axes, and the
scene summary lists each free body's local size and present heading.

## What a program means

A layout gives every free body a pose (x, y and a heading) on its support. A program accepts a
layout when

1. no two bodies interpenetrate (the signed distance of their meshes is at least -1 µm);
2. every free body rests upright on its support (roll and pitch within 1°);
3. the predicate of every requirement holds (the table below), within the statement's tolerance
   and a check slack of 2 mm and 0.5°;
4. the gates hold: G2 (`min_gap`) and G5 (the settle's speed and displacement).

A preference (`soft=w`) is reported apart and never required.

A program is the conjunction of its statements, in any order. Two kinds of statement give a body one
value instead of a condition: its support (`on_support`, `stack`) and its pull (`place`, `prefer`).
The most specific statement that reaches a body gives that value: a body named outranks a pattern,
which outranks `*`. Two statements of the same rank are an error, whatever their order.

The schedule (a `schedule` section, `"schedule"` in JSON) says how the repair reaches a layout the
program accepts, never which layouts it accepts. The certificate records it.

| entry | default | |
|---|---|---|
| `margin` | 0.005 m | the clearance the contacts keep between bodies |
| `s_min`, `ds_max` | 0.05, 0.05 | the continuation's first scale and largest step |
| `tail_iters` | 30 | the closing steps at full scale |
| `proxy_faces` | 2000 | the faces of the decimated copies the repair runs on |

```
program
  ...
schedule
  margin = 0.002, s_min = 0.3
```

The repair starts from the given layout and returns where its continuation ends. When the report
is `ok`, that layout is accepted, and the certificate records the predicate values that say so. S4R
is a local method: the layout it returns lies near the start, not always the nearest accepted one.

## Predicates

Every statement is a predicate on three measures of a body a: its centre c_a (the centre of its
bounding box, in the plane), the extent of its footprint along a unit direction u,
e_a⁻(u) ≤ (p − c_a)·u ≤ e_a⁺(u), and the heading h_a(k) of one of its axes k. Its footprint box is
the axis-aligned box of its footprint; c⁰ and h⁰ are where a body started; τ is the statement's
tolerance, with its default.

| statement | holds when | τ |
|---|---|---|
| `no_penetration(*)` | every pair of meshes at a signed distance of at least -1 µm | |
| `on_support(a, b)` | a's lowest point on the surface of b under it; on `b.levelK`, on that level with its footprint box over the level's rectangle | |
| `upright(a)` | roll and pitch within 1° | |
| `stack(m1, ..., mk)` | each member on the one before it, their centres within τ | 0.01 m |
| `within(a, R)`, `inside(a, b)` | a's footprint box inside R (a rectangle, part of one, a level), or b's, shrunk by `inset` | |
| `keep_out(a, R, gap)` | the hull of a's footprint outside R grown by `gap` | |
| `flush(a, R, side, gap)` | a's footprint edge on `side` at `gap` inside the same edge of R | 0.005 m |
| `left_of`, `right_of`, `farther`, `nearer` `(a, b, gap)` | (c_a − c_b)·u ≥ gap, with u = +y, −y, +x, −x | |
| `close_to`, `min_distance` `(a, b, r)` | ‖c_a − c_b‖ ≤ r, ≥ r | |
| `beside(a, b, side, gap, max)` | gap ≤ (c_a − c_b)·u + e_a⁻(u) − e_b⁺(u) ≤ max, u the side's direction | |
| `at(a, x, y)` | ‖c_a − (x, y)‖∞ ≤ τ | 0.01 m |
| `offset(a, b, dx, dy, frame)` | ‖c_a − c_b − d‖∞ ≤ τ, d = (dx, dy) in the scene's frame or in b's | 0.01 m |
| `centered(a, b)` | `offset(a, b, dx=0, dy=0)` | 0.01 m |
| `attach(a, b)` | a's offset and turn relative to b as they were at the start | 0.01 m, 2° |
| `symmetric(a, b, c, axis)` | c_a + c_b = 2 c_c along `axis`, and equal across it | 0.01 m |
| `between(a, b, c, axis, gap)` | c_a between c_b and c_c along `axis`, at least `gap` from each | |
| `row(m1, ..., mk, axis, ...)` | cross coordinates (centres, or the edges `edge` names) within τ; footprint gaps between neighbours along the line ≥ `gap`; with `spacing=even`, centre spacings within τ of their mean; with `mid`, the end centres' midpoint at `mid` | 0.01 m |
| `align(m1, ..., axis, edge)` | the centres, or the edges, equal along `axis` | 0.005 m |
| `evenly_spaced(m1, ..., axis)` | centre spacings along `axis` within τ of their mean | 0.005 m |
| `grid(m1, ..., cols, dx, dy)` | ‖c_k − c_1 − (row dx, column dy)‖∞ ≤ τ, a common pitch fitted when `dx`, `dy` are omitted | 0.01 m |
| `ring(m1, ..., mn, r, x, y, angle, face)` | ‖c_k − c − r (cos, sin)(θ + 2πk/n)‖∞ ≤ τ, c = (x, y) or the members' mean, θ = `angle` or fitted; with `face`, each axis within 15° of the direction to or from c | 0.01 m |
| `cluster(m1, ..., r)` | every centre within r of the members' mean | |
| `reach(a, r, min, x, y)` | min ≤ ‖c_a − (x, y)‖ ≤ r | |
| `clearance(a, gap)` | a's mesh at least `gap` from every body but its support and what rests on it | |
| `approach(a, length, width, toward)` | a corridor `width` wide and `length` long, from a's footprint toward the robot or a side, overlaps no other footprint | |
| `facing(a, b, axis)` | h_a(axis) within τ of the direction from c_a to c_b (or to x, y) | 15° |
| `yaw(a, deg, axis, mod)` | h_a(axis) within τ of `deg`, modulo `mod` | 5° |
| `parallel(a, b, axis, mod)` | h_a(axis) and h_b(axis) within τ, modulo `mod` | 5° |
| `max_turn(a, deg)`, `max_move(a, d)` | \|h_a − h_a⁰\| ≤ deg, ‖c_a − c_a⁰‖ ≤ d | |
| `balanced_on(a, b, margin)` | a's centre of mass over b's footprint in b's frame, `margin` from its edges | |
| `place(a, x, y, yaw)`, `prefer(a, w)` | pulls, reported within 5 cm (`place`) or not at all | |
| `minimize displacement(*)` | an objective | |

`centered` is a derived form: it is `offset` with no offset, and the compiler runs it as one.

## How a statement becomes optimization

At every step the QP's variables are the in-plane translation and the yaw of every free body,
`(dx, dy, dyaw)`. Contacts from the mesh evaluator are hard rows; the trust region bounds each step.
A program row is `lo <= sum(c * step) <= hi` with an elastic slack:

| kind | statements | rows |
|---|---|---|
| linear in the poses | `at`, `beside`, `within`, `inside`, `keep_out` (one side), `flush`, `left_of`, `farther` ..., `row`, `align`, `evenly_spaced`, `grid`, `symmetric`, `between`, `centered`, `offset` (scene frame), `yaw`, `parallel`, `max_turn`, `stack` | exact; re-evaluated every step only because footprints change with the scale and the heading |
| convex | `reach` (outer radius), `cluster`, `close_to`, `balanced_on`, `max_move` | the tangent half-plane at the current poses (a superset that tightens as the steps re-linearise); `max_move` uses a 16-gon inscribed in its disc, exact rows at most 2 % conservative |
| non-convex | `reach` (inner radius), `min_distance`, `facing`, `offset` in a body's frame, `attach` | the first-order expansion at the current poses |
| disjunctive | `keep_out`, `approach` | the side that needs the least motion at the current poses, chosen again at every step |
| fitted | `ring` | rows on the centres at the angle that fits the current poses best, fitted again at every step (exact with `angle`); `face` as `facing` |
| contact margins | `clearance` | the contact rows of that body keep the gap instead of the schedule's margin |

A requirement's slack costs 100 per m² (per rad² for headings), so it yields only to the contacts;
a preference (`soft=w`) costs `w` and yields to requirements too. Programs that use the v1
statements are first solved alone at the shrunken scale (`s_min`, where bodies are small enough to
pass each other): the relational counterpart of `place`, so that a row or a region is reached before
the scale is restored. A non-convex statement can stop that solve short from where the bodies start
(a linearized `min_distance` pushes a body straight away from what it must keep clear of, never
around it): the program alone is then solved again from other starts of the bodies over their
supports, and the poses closest to every requirement are kept, the bodies' own start winning a tie.
When pairs still penetrate in the closing fifth of the tail (a program the contacts contradict), the
rest of the tail serves the contacts alone: the scene comes out penetration-free and the predicate
of the statement that could not hold says so. The clearance the contacts keep is the schedule's
`margin`: a packing too tight for it ends with the requirement that could not hold reported, and a
smaller margin leaves more room.

Every statement is also a predicate on the final scene. Requirements decide whether a repair is
`ok`; preferences are reported apart (`preferences`, `failed_preferences`). The repair checks lengths
to 2 mm and headings to 0.5 degree beyond a statement's own tolerance; a settle check uses 1 cm
and 10 degrees.

## When a requirement fails

The repair report lists, for each statement whose requirement failed, the edits under which it holds
on the layout returned (`rewrites`; `rewrites` in `simready/dsl/compile.py`):

- `relax`: the nearest values of the statement's own bounds at which it holds, as
  `{keyword: [written, nearest]}`. Each is found on the statement's predicate without the check's
  slack, on a grid of 1 mm or 0.5°, one bound at a time with the ones after it at their loosest.
- `weaken`: `soft` when the statement may become a preference, `drop` when it may only be removed
  (`place`, `clearance`), nothing for what rests where and what is fixed (`no_penetration`, `fixed`,
  `on_support`, `upright`, `stack`).

`schedule_rewrites` are the repair run again from the same start with half and then a quarter of
the schedule's margin (not below 1 mm), until one ends ok, with what each ends with. A schedule never
changes what is accepted, so these keep the request, and a program given that margin ends exactly as
reported.

| statements | bounds that relax |
|---|---|
| `left_of`, `right_of`, `farther`, `nearer`, `between`, `keep_out`, `clearance` | `gap` down |
| `close_to`, `cluster` / `min_distance` | `r` up / down |
| `reach` | `r` up, `min` down |
| `beside` | `gap` down, `max` up |
| `within` | `inset` down |
| `approach` | `length`, `width` down |
| `row` | `gap` down, `tol` up |
| `max_move`, `max_turn` | `d`, `deg` up |
| `balanced_on` | `margin` down |
| `at`, `offset`, `centered`, `attach`, `symmetric`, `align`, `evenly_spaced`, `grid`, `ring`, `flush`, `stack`, `facing`, `yaw`, `parallel` | `tol` up |

Where a statement puts a body (a region's edges, a side, a target, a level) is what it asks, not how
strictly, and is never relaxed.

Every rewrite asks less than its statement did. Each predicate is monotone in its bounds
(`tests/test_dsl_v1.py` checks each on scattered layouts), so a layout the program accepted, the
edited program accepts too, and the edit keeps the value it replaces. `accepted_if_relaxed` says whether relaxing
every failed statement as listed makes the layout returned accepted; `accepted_if_weakened`, whether
weakening the ones that cannot be relaxed does. Whether an edit keeps the request is the author's
call: a number the request states is not one to relax. An edited program is repaired again from the
same start, and the layout it returns can differ.

## Statements

`a`, `b`, `c` are body names (`b.top` is b's top rectangle; `b.levelK` is level K of a fixture, from 1 at
the bottom, whose height, rectangle and headroom the scene summary lists); group statements take a list of bodies
in order. In the first argument and among a group's members, `*` is every free body and a pattern
(`can_*`, `fruit_?`) every free body whose name matches, in natural order (`can_2` before `can_10`).
A pattern leaves out the bodies its statement names itself: `row(can_3, can_*)` puts can_3 first and
the other cans after it. `beside`, `attach` and `balanced_on` with a pattern are one statement per
body; `place`, `centered`, `offset`, `symmetric` and `between` take one named body. A pattern that
matches no free body is an error, so a misspelt group is not silently empty.

### The robot and the task

| statement | holds when | use |
|---|---|---|
| `reach(a, r, min=0, x=0, y=0)` | a's centre within `r` of the robot base (and at least `min` from it) | the arm's workspace annulus |
| `clearance(a, gap)` | every other body is at least `gap` from a's surface (its support and what rests on it excepted) | room for gripper fingers |
| `approach(a, length=0.20, width=0.10, toward=robot)` | no other body's footprint in the corridor of that size from a toward the robot (or `near`, `far`, `left`, `right`) | a clear line of approach |
| `facing(a, b, axis=+x, tol=15)` | a's local axis (`front` / `handle`: the side the body's data names) points at b (or at `x=, y=`, by default the robot) | handles, labels, fronts |
| `at(a, x, y, tol=0.01)` | a's centre at `x` and/or `y` | a hard position (`place` is a soft pull) |
| `yaw(a, deg, tol=5, mod=360, axis=x)` | the heading of a's local `+x` (`axis=long`: of its long side, a line; `axis=front`: of its front side) is `deg`; `mod` 90 or 180 for shapes that repeat | square to the table, a board along x, spines out |
| `parallel(a, b, tol=5, axis=long)` | a's and b's long sides parallel (`axis=x`: their local x axes) | cutlery, tools, books |

### Arrangements

| statement | holds when | use |
|---|---|---|
| `beside(a, b, side, gap=0, max)` | a's footprint on b's `side` with between `gap` and `max` of table between the two footprints | "the fork 2 to 5 cm left of the plate" |
| `row(m1, m2, ..., axis=y, start, gap=0.01, tol=0.01, order, spacing, mid)` | the members on one line along `axis` (`x` or `y`), in the order listed from the end `start` names (`left` or `right` for a line along y, `near` or `far` along x; without it the list runs toward +y or +x), footprints at least `gap` apart; `order=height` / `width` / `length` / `mass` (`-` descending) sorts them first; `spacing=even` keeps every centre spacing within `tol` of the mean; `mid` puts the midpoint of the two end members' centres at that coordinate along `axis`, whichever members the order puts at the ends; `edge=min` / `max` lines up the footprints' near or far edges across the row instead of the centres | drinks on a shelf, cans in a line, spines in line |
| `align(m1, ..., axis=x, edge=center, tol=0.005)` | equal x (or y) of the centres or of the footprints' `min` / `max` edge | flush backs, spines in line |
| `evenly_spaced(m1, m2, m3, ..., axis=y, tol=0.005)` | every centre spacing within `tol` of the mean spacing | spice jars, test tubes |
| `grid(m1, ..., cols, dx, dy, tol=0.01)` | rows of `cols` members, member k at `(row dx, col dy)` from the first; a free shared pitch without `dx`, `dy` | trays, racks, egg cartons |
| `ring(m1, m2, m3, ..., r, x, y, angle, face, axis=+x, tol=0.01)` | the members evenly spaced counter-clockwise on a circle of radius `r` about `(x, y)` (without them, their own centre), the first at `angle` degrees from +x (without it, where they move least); `face=in` / `out` turns each one's `axis` toward or away from the centre | cups round a teapot, chairs round a table |
| `cluster(m1, ..., r)` | every member within `r` of the group's centre | "the fruit together" |
| `symmetric(a, b, c, axis=y, tol=0.01)` | a and b mirror images about c | fork and knife about the plate |
| `between(a, b, c, axis=y, gap=0)` | a's centre between b's and c's | "the cup between the plate and the bottle" |
| `centered(a, b, tol=0.01)` | a's centre over b's | a cup on its saucer, a box on a pallet |
| `offset(a, b, dx, dy, frame=world, tol=0.01)` | a's centre at `(dx, dy)` from b's, in the scene frame or b's own (`frame=local`) | templates relative to a body |
| `attach(a, b)` | a keeps its starting pose relative to b | a lid on its pot, a set kept together |

### Regions, edits, stacking, preferences

| statement | holds when | use |
|---|---|---|
| `within(a, b, inset=0.02, side, frac=0.5, depth, x0, x1, y0, y1)` | a's footprint inside the region: b's rectangle, the part of it `side` and `frac` select, or the strip beside it of that `depth`, bounded by `x0 .. y1` (bounds alone define a region) | "the fruit on the left third" |
| `keep_out(a, b, gap=0, region fields as within)` | a's footprint outside the region | drop zones, the swing of a door (`side=near, depth=0.3`) |
| `flush(a, b, side, gap=0, tol=0.005)` | a's footprint edge on `side` `gap` inside the same edge of b's rectangle (`b`, `b.top` or `b.levelK`) | books 1 cm behind a shelf's front, boxes against the back |
| `max_move(a, d)`, `max_turn(a, deg)` | a moves at most `d` / turns at most `deg` from where it started | repairing a shipped or human-made scene |
| `stack(m1, m2, ...)` | each member rests on the one before it, centred within `tol` | towers, pallet layers |
| `balanced_on(a, b, margin=0.01)` | a's centre of mass (from the asset's physics when it has it) over b's top, `margin` in from its edge, in b's own frame | stability of a stack |
| any statement but `no_penetration`, `fixed`, `on_support`, `upright`, `place`, `clearance`, `stack`, with `soft=w` | a preference of weight `w` in (0, 1] | "ideally", "if possible" |

### The statements of v0

`no_penetration(*)`, `fixed(a)`, `on_support(a, b)`, `upright(a)`, `within(a, b, inset)`,
`inside(a, b)`, `left_of`, `right_of`, `farther`, `nearer` `(a, b, gap)`, `min_distance` and
`close_to` `(a, b, r)`, `place(a, x, y, yaw)`, `prefer(a, w)`, `minimize displacement(*)`. A body's
support is the most specific one given to it: `on_support(can_*, tray)` over `on_support(*, table)`,
and `on_support` or a `stack` naming the body over both. On a fixed support a body rests
on the level under it (a shelf below the fixture's top); on a free one (a carton on a carton, fruit in
a bowl) it rests on the highest surface under it, wherever it starts.

Levels:

- `on_support(a, shelf.level2)` puts a on that level, wherever it starts.
- The compiler refuses a body, or a stack, taller than the room above its level (with the schedule's margin),
  and names the levels that have room.
- A body on a level it is given, or on a shelf under another part of its fixture, keeps its
  footprint on that level, as a contact does.

## Recipes

Each of these is a program that passes the rubric of the matching task of the comparison
(`experiments/dsl_vs_direct/tasks.py`); the shared lines are

```
no_penetration(*)   on_support(*, table)   upright(*)   within(*, table.top, inset=0.02)
```

**A place setting.**
```
at(plate, x=0.55, y=0.0, tol=0.005)
yaw(fork, deg=0, axis=long)   yaw(knife, deg=0, axis=long)   parallel(spoon, knife)
symmetric(fork, knife, plate, axis=y)
beside(fork, plate, side=left, gap=0.02, max=0.05)   beside(knife, plate, side=right, gap=0.02, max=0.05)
beside(spoon, knife, side=right, gap=0.02)   beside(mug, fork, side=left, gap=0.03)   facing(mug, axis=-x, tol=10)
farther(glass, plate, gap=0.12)   close_to(glass, knife, r=0.14)
```

**Drinks in a row, shortest at the left, evenly spaced, centred on y = 0.** The patterns list every
drink, the compiler sorts them by height, and `mid` centres the two ends it chose on the table's centre
line. With only drinks on the table, `row(*, ...)` says the same.
```
row(can_*, bottle_*, glass_*, start=left, gap=0.025, order=height, spacing=even, mid=0, tol=0.004)
at(*, x=0.75, tol=0.004)
```

**Dense clutter.** The square's area is 1.12 times the sum of the fourteen fruits' bounding
rectangles: boxes do not fit, the shapes do, and the mesh contacts pack them (schedule: `margin = 0.002`).
```
within(*, x0=0.40, x1=0.72, y0=-0.16, y1=0.16)
```

**Sorting a heap into zones.** The patterns keep the program the same size however many fruits and
cans there are: 11, 13 and 16 statements for the 10, 16 and 24 objects of the comparison's three
sizes, against 18, 25 and 35 with every body named.
```
within(fruit_*, table.top, y0=0.27)   cluster(fruit_*, r=0.10)
within(can_*, table.top, y1=-0.27)   row(can_*, start=near, gap=0.025)
within(bowl, y0=-0.18, y1=0.18)   reach(bowl, r=0.75, min=0.57)   keep_out(*, x0=0.29, x1=0.51, y0=-0.11, y1=0.11)
```

**A scene ready for a pick.**
```
reach(mug, r=0.58, min=0.47)   facing(mug, axis=-x, tol=10)   approach(mug, length=0.6, width=0.17)
min_distance(*, mug, r=0.19)   keep_out(*, x0=0.24, x1=0.41, y0=0.19, y1=0.41)   reach(*, r=0.78)
```

**Fixing a scene without rearranging it.**
```
max_move(*, d=0.028)   max_turn(*, deg=8)
```

**Palletizing.**
```
within(*, pallet.top, inset=0.03)   yaw(*, deg=0, mod=90, tol=2)   clearance(*, gap=0.015)
row(box_1, box_2, box_3, start=right, gap=0.02)   row(box_4, box_5, box_6, start=right, gap=0.02)   farther(box_4, box_1, gap=0.30)
```

**A tower.**
```
stack(box_large, box_medium, box_small, tol=0.005)   within(box_large, table.top, side=right, frac=0.5)   reach(box_large, r=0.68)
within(apple, table.top, side=left, frac=0.5)  ...
```

### Shelves

The programs of the examples in [`examples/`](examples/) (`shelf_books`, `shelf_pantry`, `shelf_fix`).

**Books upright, spines out, by height; the ones too tall for the shelves on top; notebooks stacked at the back.**
```
on_support(book_*, case.level2)   on_support(book_2, case.level4)   on_support(book_3, case.level4)   yaw(book_*, deg=180, axis=front)
row(book_1, book_4, book_5, book_6, book_7, start=left, order=-height, gap=0.005, mid=0, edge=min)   flush(book_1, case.level2, side=near, gap=0.01)
on_support(notebook_1, case.level1)   stack(notebook_1, notebook_2, notebook_3, notebook_4)   flush(notebook_1, case.level1, side=far, gap=0.01)
```

**A pantry rack of 45 things, heavy ones low** (schedule: `margin = 0.003`).
```
within(*, rack, x0=0.57)
on_support(mug_*, rack.level1)   on_support(soup_*, rack.level1)   yaw(mug_*, deg=-90, axis=handle)
row(mug_*, soup_*, start=left, gap=0.02, edge=min)   flush(mug_1, rack.level1, side=near, gap=0.03)
on_support(bowl_1, rack.level1)   on_support(bowl_3, rack.level1)   stack(bowl_1, bowl_2)   stack(bowl_3, bowl_4)
within(bowl_*, rack.level1, x0=0.7)
on_support(can_*, rack.level2)   grid(can_*, cols=9, dx=0.09, dy=0.09)
on_support(jar_*, rack.level3)   row(jar_*, start=left, order=-mass, spacing=even, gap=0.01)   within(jar_*, rack.level3, x1=0.69)
on_support(bottle_*, rack.level3)   row(bottle_*, gap=0.02)   within(bottle_*, rack.level3, x0=0.72)
```

**Fixing a stocked rack.** `on_support(*, rack)` keeps every body on the level under it (schedule: `margin = 0.002`).
```
no_penetration(*)   on_support(*, rack)   max_move(*, d=0.02)   max_turn(*, deg=10)
```

### A ring

The program of the RoboCasa demo in the [README](../README.md#a-robocasa-counter): ten objects heaped at the
counter's centre, laid out on a circle (schedule: `margin = 0.003`).
```
no_penetration(*)   fixed(counter)   on_support(*, counter)   upright(*)   ring(*, r=0.2, x=0, y=0)
```

## Defaults

Omitted keywords take recorded defaults (`simready/dsl/schema.py`: `DEFAULTS`, `OP_DEFAULTS`), which
the certificate lists under `provenance.tolerances.statement_defaults`; the schedule it ran with is
`provenance.schedule`.

## Limits

- A disjunction (`keep_out`, `approach`) picks its side from the current poses. Combined with a rigid
  structure (a fixed-pitch `grid` that would have to leave a keep-out region as a whole), the choice
  can stall; the other starts of the program's solve at the shrunken scale find another side when
  the stall is there, and otherwise the scene stays penetration-free and the predicate reports the
  shortfall.
- Footprints are the projections of a body's mesh on the plane: corridors, regions and gaps do not
  know heights (a flat sheet in a corridor blocks it).
- `balanced_on` takes b's top as its footprint rectangle in b's own frame.
- A level is a horizontal surface of a fixture: a wire deck (the tops of wires) is not found, and a
  level's rectangle is a box in the scene frame, exact for a fixture turned by a quarter turn. It is
  one rectangle: a divider standing on a board inside it is not kept apart from the bodies on that
  board (their contact with the fixture is the board they rest on).
- `max_move`, `max_turn` and `attach` are measured from the poses the program was compiled on; after
  a settle they are measured from the poses the settle started from.
