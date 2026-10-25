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
+x is away from the robot, -x toward it, +y left, -y right. Regions name their sides `near` (-x),
`far` (+x), `left` (+y) and `right` (-y). `left_of`, `right_of`, `in_front_of` and `behind` keep
RoboLab's convention (front = +x). A body's local frame is the frame its asset rests in; `yaw`,
`facing` and `parallel` are about its local axes, and the scene summary lists each free body's
local size and present heading.

## How a statement becomes optimization

At every step the QP's variables are the in-plane translation and the yaw of every free body,
`(dx, dy, dyaw)`. Contacts from the mesh evaluator are hard rows; the trust region bounds each step.
A program row is `lo <= sum(c * step) <= hi` with an elastic slack:

| kind | statements | rows |
|---|---|---|
| linear in the poses | `at`, `beside`, `within`, `inside`, `keep_out` (one side), `left_of` ..., `row`, `align`, `evenly_spaced`, `grid`, `symmetric`, `between`, `centered`, `offset` (scene frame), `yaw`, `parallel`, `max_turn`, `stack` | exact; re-evaluated every step only because footprints change with the scale and the heading |
| convex | `reach` (outer radius), `cluster`, `near`, `balanced_on`, `max_move` | the tangent half-plane at the current poses (a superset that tightens as the steps re-linearise); `max_move` uses a 16-gon inscribed in its disc, exact rows at most 2 % conservative |
| non-convex | `reach` (inner radius), `min_distance`, `facing`, `offset` in a body's frame, `attach` | the first-order expansion at the current poses |
| disjunctive | `keep_out`, `approach` | the side that needs the least motion at the current poses, chosen again at every step |
| contact margins | `clearance` | the contact rows of that body keep the gap instead of the program's margin |

A requirement's slack costs 100 per m² (per rad² for headings), so it yields only to the contacts;
a preference (`soft=w`) costs `w` and yields to requirements too. Programs that use the v1
statements are first solved alone at the shrunken scale (`s_min`, where bodies are small enough to
pass each other): the relational counterpart of `place`, so that a row or a region is reached before
the scale is restored. When pairs still penetrate in the closing fifth of the tail (a program the
contacts contradict), the rest of the tail serves the contacts alone: the scene comes out
penetration-free and the predicate of the statement that could not hold says so.

Every statement is also a predicate on the final scene. Requirements decide whether a repair is
`ok`; preferences are reported apart (`preferences`, `failed_preferences`). The repair checks lengths
to 2 mm and headings to 0.5 degree beyond a statement's own tolerance; a settle check uses 1 cm
and 10 degrees.

## Statements

`a`, `b`, `c` are body names (`b.top` is b's top rectangle); `*` (first argument only) is every free
body; group statements take a list of named bodies in order.

### The robot and the task

| statement | holds when | use |
|---|---|---|
| `reach(a, r, min=0, x=0, y=0)` | a's centre within `r` of the robot base (and at least `min` from it) | the arm's workspace annulus |
| `clearance(a, gap)` | every other body is at least `gap` from a's surface (its support and what rests on it excepted) | room for gripper fingers |
| `approach(a, length=0.20, width=0.10, toward=robot)` | no other body's footprint in the corridor of that size from a toward the robot (or `near`, `far`, `left`, `right`) | a clear line of approach |
| `facing(a, b, axis=+x, tol=15)` | a's local axis points at b (or at `x=, y=`, by default the robot) | handles, labels, fronts |
| `at(a, x, y, tol=0.01)` | a's centre at `x` and/or `y` | a hard position (`place` is a soft pull) |
| `yaw(a, deg, tol=5, mod=360, axis=x)` | the heading of a's local `+x` (`axis=long`: of its long side, a line) is `deg`; `mod` 90 or 180 for shapes that repeat | square to the table, a board along x |
| `parallel(a, b, tol=5, axis=long)` | a's and b's long sides parallel (`axis=x`: their local x axes) | cutlery, tools, books |

### Arrangements

| statement | holds when | use |
|---|---|---|
| `beside(a, b, side, gap=0, max)` | a's footprint on b's `side` with between `gap` and `max` of table between the two footprints | "the fork 2 to 5 cm left of the plate" |
| `row(m1, m2, ..., axis=+y, gap=0.01, tol=0.01, order, spacing)` | the members on one line, in the order listed along `axis` (`+y`: the list runs right to left, `-y` left to right, `+x` near to far), footprints at least `gap` apart; `order=height` / `width` / `length` / `mass` (`-` descending) sorts them first; `spacing=even` keeps every centre spacing within `tol` of the mean | drinks on a shelf, cans in a line |
| `align(m1, ..., axis=x, edge=center, tol=0.005)` | equal x (or y) of the centres or of the footprints' `min` / `max` edge | flush backs, spines in line |
| `evenly_spaced(m1, m2, m3, ..., axis=y, tol=0.005)` | every centre spacing within `tol` of the mean spacing | spice jars, test tubes |
| `grid(m1, ..., cols, dx, dy, tol=0.01)` | rows of `cols` members, member k at `(row dx, col dy)` from the first; a free shared pitch without `dx`, `dy` | trays, racks, egg cartons |
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
| `max_move(a, d)`, `max_turn(a, deg)` | a moves at most `d` / turns at most `deg` from where it started | repairing a shipped or human-made scene |
| `stack(m1, m2, ...)` | each member rests on the one before it, centred within `tol` | towers, pallet layers |
| `balanced_on(a, b, margin=0.01)` | a's centre of mass (from the asset's physics when it has it) over b's top, `margin` in from its edge, in b's own frame | stability of a stack |
| any statement but `no_penetration`, `fixed`, `on_support`, `upright`, `place`, `clearance`, `stack`, with `soft=w` | a preference of weight `w` in (0, 1] | "ideally", "if possible" |

### The statements of v0 (unchanged)

`no_penetration(*, margin)`, `fixed(a)`, `on_support(a, b)`, `upright(a)`, `within(a, b, inset)`,
`inside(a, b)`, `left_of`, `right_of`, `in_front_of`, `behind` `(a, b, gap)`, `min_distance` and
`near` `(a, b, r)`, `place(a, x, y, yaw)`, `prefer(a, w)`, `minimize displacement(*)`. A body named in
`on_support` or in a `stack` keeps that support over one `on_support(*, ...)` gives it.

## Recipes

Each of these is a program that passes the rubric of the matching task of the comparison
(`experiments/dsl_vs_direct/tasks.py`); the shared lines are

```
no_penetration(*, margin=0.005)   on_support(*, table)   upright(*)   within(*, table.top, inset=0.02)
```

**A place setting.**
```
at(plate, x=0.55, y=0.0, tol=0.005)
yaw(fork, deg=0, axis=long)   yaw(knife, deg=0, axis=long)   parallel(spoon, knife)
symmetric(fork, knife, plate, axis=y)
beside(fork, plate, side=left, gap=0.02, max=0.05)   beside(knife, plate, side=right, gap=0.02, max=0.05)
beside(spoon, knife, side=right, gap=0.02)   beside(mug, fork, side=left, gap=0.03)   facing(mug, axis=-x, tol=10)
in_front_of(glass, plate, gap=0.12)   near(glass, knife, r=0.14)
```

**Drinks in a row, shortest at the left.**
```
row(can_3, can_5, bottle_1, can_1, can_2, can_4, axis=-y, gap=0.025)   evenly_spaced(can_3, can_5, bottle_1, can_1, can_2, can_4, axis=y)
at(*, x=0.75, tol=0.004)
```

**Sorting a heap into zones.**
```
within(fruit_1, table.top, y0=0.27)  ...   cluster(fruit_1, fruit_2, fruit_3, fruit_4, fruit_5, r=0.10)
within(can_1, table.top, y1=-0.27)  ...   row(can_1, can_2, can_3, can_4, axis=+x, gap=0.025)
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
row(box_1, box_2, box_3, axis=+y, gap=0.02)   row(box_4, box_5, box_6, axis=+y, gap=0.02)   in_front_of(box_4, box_1, gap=0.30)
```

**A tower.**
```
stack(box_large, box_medium, box_small, tol=0.005)   within(box_large, table.top, side=right, frac=0.5)   reach(box_large, r=0.68)
within(apple, table.top, side=left, frac=0.5)  ...
```

## Defaults

Omitted keywords take recorded defaults (`simready/dsl/schema.py`: `DEFAULTS`, `OP_DEFAULTS`), which
the certificate lists under `provenance.tolerances.statement_defaults`.

## Limits

- A disjunction (`keep_out`, `approach`) picks its side from the current poses. Combined with a rigid
  structure (a fixed-pitch `grid` that would have to leave a keep-out region as a whole), the choice
  can stall; the scene stays penetration-free and the predicate reports the shortfall.
- Footprints are the projections of a body's mesh on the plane: corridors, regions and gaps do not
  know heights (a flat sheet in a corridor blocks it).
- `balanced_on` takes b's top as its footprint rectangle in b's own frame.
- `max_move`, `max_turn` and `attach` are measured from the poses the program was compiled on; after
  a settle they are measured from the poses the settle started from.
