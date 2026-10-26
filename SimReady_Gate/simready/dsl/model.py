"""SimReady DSL: a typed program of constraints and objectives over bodies.

Text form (one statement per line, '#' comments, `*` = every free body):

    program
      no_penetration(*, margin=0.01)
      fixed(table)
      on_support(mug, table)      upright(mug)
      within(mug, table.top, inset=0.02)
      on_support(book_*, shelf.level2)        # a fixture's level, from 1 at the bottom (the scene summary lists them)
      flush(book_*, shelf.level2, side=near, gap=0.01)   # the footprints' near edges 1 cm inside the level's front edge
      place(mug, x=0.45, y=0.10, yaw=30)      # a target pose in the shrunken scale-space (soft)
      inside(mug, tray)
      left_of(mug, bowl, axis=y, gap=0.05)
      min_distance(mug, bowl, r=0.10)
      near(mug, bowl, r=0.25)
      minimize displacement(*)
      prefer(bowl, pose=intent, w=0.1)
      # v1: the robot and the task
      reach(mug, r=0.75, min=0.30)            # centre within the arm's annulus about the base (x=, y=; default 0, 0)
      at(plate, x=0.55, y=0.0, tol=0.01)      # the centre in a window (a hard position; place is a soft pull)
      clearance(mug, gap=0.03)                # every other body at least 3 cm from the mug's mesh
      approach(mug, length=0.20, width=0.10)  # a free corridor from the mug toward the robot
      facing(mug, axis=-x, tol=20)            # the mug's local -x (its handle) points at the robot (or at b / x=, y=)
      yaw(book_*, deg=180, axis=front)        # each book's own front side (its spine, from its data) points along -x
      yaw(box, deg=0, mod=90, tol=3)          yaw(board, deg=90, axis=long)       parallel(fork, knife)   # long sides
      # v1: arrangements (members in the order given)
      row(can_1, can_2, can_3, axis=y, gap=0.02)     align(book_1, book_2, axis=x, edge=max)
      row(can_*, start=left, order=height, spacing=even, mid=0)  # every can, sorted by the compiler, shortest at the left
      evenly_spaced(can_1, can_2, can_3, axis=y)     grid(c1, c2, c3, c4, cols=2, dx=0.12, dy=0.12)
      cluster(orange, lime, lemon, r=0.12)           symmetric(fork, knife, plate, axis=y)
      between(cup, plate, bottle, axis=y, gap=0.05)  centered(cup, saucer)
      offset(fork, plate, dy=0.15, frame=local)      attach(lid, pot)
      # v1: regions, edits, stacking, preferences
      within(apple, table.top, side=left, frac=0.5)  keep_out(*, oven, side=near, depth=0.30)
      max_move(*, d=0.03)   max_turn(*, deg=10)      stack(box_1, box_2, box_3)   balanced_on(box_2, box_1)
      beside(fork, plate, side=left, gap=0.02, max=0.05)   # footprints 2-5 cm apart, the fork on the plate's left
      left_of(mug, bowl, gap=0.10, soft=0.5)         # soft=: a preference, reported but not required
    gate
      G2: min_gap >= 1e-3
      G5: v_max <= 0.05, dx <= 0.01

Every statement compiles to per-step QP rows (see compile.py) and to a final
predicate check, so the repair and the verifier enforce the same object.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NAME_RE = re.compile(r"^[A-Za-z_][\w\-]*(\.[A-Za-z_]\w*)?$")
# '*' and name patterns (can_*, fruit_?): free bodies, in natural order (can_2 before can_10)
PATTERN_RE = re.compile(r"^[\w\-*?]*[*?][\w\-*?]*$")


def is_wildcard(tok: str) -> bool:
    """'*' or a name pattern: it stands for free bodies, not for one named body."""
    return "*" in tok or "?" in tok


REGION_KW = {"side", "frac", "depth", "x0", "x1", "y0", "y1"}
KEYWORDS = {"no_penetration": {"margin"}, "within": {"inset", "soft"} | REGION_KW, "inside": {"inset", "soft"},
            "place": {"x", "y", "yaw", "w"},
            "left_of": {"gap", "axis", "soft"}, "right_of": {"gap", "axis", "soft"}, "in_front_of": {"gap", "axis", "soft"},
            "behind": {"gap", "axis", "soft"}, "min_distance": {"r", "soft"}, "near": {"r", "soft"}, "prefer": {"pose", "w"},
            # v1
            "reach": {"r", "min", "x", "y", "soft"}, "clearance": {"gap"}, "at": {"x", "y", "tol", "soft"},
            "beside": {"side", "gap", "max", "soft"},
            "approach": {"length", "width", "toward", "x", "y", "soft"},
            "facing": {"axis", "tol", "x", "y", "soft"}, "yaw": {"deg", "tol", "mod", "axis", "soft"}, "parallel": {"mod", "tol", "axis", "soft"},
            "row": {"axis", "start", "gap", "tol", "order", "spacing", "mid", "edge", "soft"}, "align": {"axis", "edge", "tol", "soft"},
            "evenly_spaced": {"axis", "tol", "soft"}, "grid": {"cols", "dx", "dy", "tol", "soft"}, "cluster": {"r", "soft"},
            "symmetric": {"axis", "tol", "soft"}, "between": {"axis", "gap", "soft"}, "centered": {"tol", "soft"},
            "offset": {"dx", "dy", "frame", "tol", "soft"}, "attach": {"tol", "soft"},
            "keep_out": {"gap", "soft"} | REGION_KW, "flush": {"side", "gap", "tol", "soft"}, "max_move": {"d", "soft"}, "max_turn": {"deg", "soft"},
            "stack": {"tol"}, "balanced_on": {"margin", "soft"}}
NUMERIC_KW = {"margin", "inset", "x", "y", "yaw", "w", "gap", "r", "min", "max", "length", "width", "tol", "deg", "mod", "cols",
              "dx", "dy", "frac", "depth", "x0", "x1", "y0", "y1", "d", "soft", "mid"}
SIGNED_KW = {"x", "y", "yaw", "deg", "dx", "dy", "x0", "x1", "y0", "y1", "mid"}
# keywords whose value is a word, and the words each statement accepts
WORDS = {"axis": {"left_of": {"x", "y"}, "right_of": {"x", "y"}, "in_front_of": {"x", "y"}, "behind": {"x", "y"},
                  "align": {"x", "y"}, "evenly_spaced": {"x", "y"}, "symmetric": {"x", "y"}, "between": {"x", "y"},
                  "row": {"x", "y"}, "facing": {"x", "y", "+x", "-x", "+y", "-y", "front", "handle"},
                  "yaw": {"x", "y", "+x", "-x", "+y", "-y", "long", "short", "front", "handle"}, "parallel": {"x", "y", "long", "short"}},
         "edge": {"align": {"min", "center", "max"}, "row": {"min", "center", "max"}},
         "order": {"row": {"height", "-height", "width", "-width", "length", "-length", "mass", "-mass"}},
         "spacing": {"row": {"even"}},
         "start": {"row": {"left", "right", "near", "far"}},
         "side": {**{op: {"left", "right", "near", "far", "center"} for op in ("within", "keep_out")},
                  "beside": {"left", "right", "near", "far"}, "flush": {"left", "right", "near", "far"}},
         "toward": {"approach": {"robot", "near", "far", "left", "right"}},
         "frame": {"offset": {"world", "local"}},
         "pose": {"prefer": {"intent"}}}
STATEMENTS = {
    "no_penetration", "place", "fixed", "on_support", "upright", "within", "inside",
    "left_of", "right_of", "in_front_of", "behind", "min_distance", "near",
    "minimize", "prefer",
    "reach", "clearance", "approach", "facing", "yaw", "parallel", "at", "beside",
    "row", "align", "evenly_spaced", "grid", "cluster", "symmetric", "between", "centered", "offset", "attach",
    "keep_out", "flush", "max_move", "max_turn", "stack", "balanced_on",
}
# positional arguments each statement takes: (min, max); None = unbounded
ARITY = {"no_penetration": (1, 1), "place": (1, 1), "fixed": (1, None), "on_support": (2, 2), "upright": (1, 1),
         "within": (1, 2), "inside": (2, 2), "left_of": (2, 2), "right_of": (2, 2), "in_front_of": (2, 2),
         "behind": (2, 2), "min_distance": (2, 2), "near": (2, 2), "minimize": (2, 2), "prefer": (1, 1),
         "reach": (1, 1), "clearance": (1, 1), "at": (1, 1), "beside": (2, 2), "approach": (1, 1), "facing": (1, 2), "yaw": (1, 1), "parallel": (2, 2),
         "row": (2, None), "align": (2, None), "evenly_spaced": (3, None), "grid": (2, None), "cluster": (2, None),
         "symmetric": (3, 3), "between": (3, 3), "centered": (2, 2), "offset": (2, 2), "attach": (2, 2),
         "keep_out": (1, 2), "flush": (2, 2), "max_move": (1, 1), "max_turn": (1, 1), "stack": (2, None), "balanced_on": (2, 2)}
# statements over a list of distinct bodies, in the order given; '*' and patterns among the members
# stand for the free bodies they match that the statement does not name itself
GROUP = {"row", "align", "evenly_spaced", "grid", "cluster", "stack"}
# the line a row runs along, from the end its first member is at
START_LINE = {"left": "y", "right": "y", "near": "x", "far": "x"}
# statements whose first argument is one named body ('*' or a pattern would put every body at one spot)
ONE_BODY = {"place", "centered", "offset", "symmetric", "between"}
# keywords a statement cannot do without
REQUIRED = {"reach": {"r"}, "clearance": {"gap"}, "beside": {"side"}, "flush": {"side"}, "yaw": {"deg"}, "grid": {"cols"}, "cluster": {"r"},
            "max_move": {"d"}, "max_turn": {"deg"}, "min_distance": {"r"}, "near": {"r"}}
# gate sections, their fields and the only operator each field takes
GATES = {"G2": {"min_gap": ">="}, "G5": {"v_max": "<=", "dx": "<="}}


@dataclass
class Statement:
    name: str
    args: list            # positional tokens (str)
    kw: dict              # key -> value (float or str)
    line: int = 0

    def __repr__(self):
        kw = ", ".join(f"{k}={v}" for k, v in self.kw.items())
        return f"{self.name}({', '.join(self.args)}{', ' if kw and self.args else ''}{kw})"

    @property
    def soft(self) -> bool:
        """A statement with `soft=` is a preference: weighted rows, a reported but not required predicate."""
        return "soft" in self.kw


@dataclass
class Program:
    statements: list = field(default_factory=list)
    gate: dict = field(default_factory=dict)
    source: str = ""

    def of(self, name: str):
        return [s for s in self.statements if s.name == name]


_num = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _val(tok: str):
    tok = tok.strip()
    if _num.match(tok):
        return float(tok)
    return tok


def _split_args(s: str):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out]


def check_statement(name: str, args: list, kw: dict, where: str = "") -> None:
    """The rules a statement must meet whichever way it was written (text or JSON); raises
    SyntaxError naming the statement."""
    at = f"{where}: " if where else ""
    lo_n, hi_n = ARITY[name]
    if name in GROUP and any(is_wildcard(a) for a in args):
        lo_n = 1                              # counted once the wildcards are expanded on a scene
    if len(args) < lo_n or (hi_n is not None and len(args) > hi_n):
        want = f"{lo_n}" if lo_n == hi_n else (f"at least {lo_n}" if hi_n is None else f"{lo_n}-{hi_n}")
        raise SyntaxError(f"{at}{name} takes {want} argument(s), got {len(args)}")
    missing = REQUIRED.get(name, set()) - set(kw)
    if missing:
        raise SyntaxError(f"{at}{name} needs {', '.join(sorted(k + '=' for k in missing))}")
    if name in GROUP:
        if len(set(args)) != len(args):
            raise SyntaxError(f"{at}{name} lists a body twice")
    elif any(is_wildcard(a) for a in args[1:]):
        raise SyntaxError(f"{at}'*' and name patterns are only allowed as the first argument or among a group's members")
    elif name in ONE_BODY and args and is_wildcard(args[0]):
        raise SyntaxError(f"{at}{name} takes one named body, not '{args[0]}'")
    if name in ("symmetric", "between") and len(set(args)) != 3:
        raise SyntaxError(f"{at}{name} needs three different bodies")
    for k, v in kw.items():
        if k in WORDS and v not in WORDS[k].get(name, set()):
            raise SyntaxError(f"{at}{name}: {k}= must be one of {sorted(WORDS[k].get(name, set()))}, got '{v}'")
    if name == "within" and len(args) == 1 and not ({"x0", "x1", "y0", "y1"} & set(kw)):
        raise SyntaxError(f"{at}within without a region body needs bounds (x0=, x1=, y0=, y1=)")
    if name == "keep_out" and len(args) == 1 and not ({"x0", "x1", "y0", "y1"} & set(kw)):
        raise SyntaxError(f"{at}keep_out without a region body needs bounds (x0=, x1=, y0=, y1=)")
    if ("frac" in kw or "depth" in kw) and "side" not in kw:
        raise SyntaxError(f"{at}{name}: frac= and depth= go with side=")
    if "frac" in kw and "depth" in kw:
        raise SyntaxError(f"{at}{name}: give frac= (a part of the region) or depth= (a strip beside it), not both")
    if "start" in kw and "axis" in kw and START_LINE[kw["start"]] != kw["axis"]:
        raise SyntaxError(f"{at}{name}: a line along {kw['axis']} starts at {' or '.join(e for e, a in START_LINE.items() if a == kw['axis'])}")
    if name == "beside" and "max" in kw and kw["max"] < kw.get("gap", 0.0):
        raise SyntaxError(f"{at}beside: max= is smaller than gap=")
    if "side" in kw and len(args) < 2:
        raise SyntaxError(f"{at}{name}: side= needs a region body")
    if "frac" in kw and not 0.0 < kw["frac"] <= 1.0:
        raise SyntaxError(f"{at}frac= must lie in (0, 1]")
    if "soft" in kw and not 0.0 < kw["soft"] <= 1.0:
        raise SyntaxError(f"{at}soft= must lie in (0, 1]")
    if "mod" in kw and kw["mod"] not in (90.0, 180.0, 360.0):
        raise SyntaxError(f"{at}mod= must be 90, 180 or 360")
    if "cols" in kw and (kw["cols"] < 1 or kw["cols"] != int(kw["cols"])):
        raise SyntaxError(f"{at}cols= must be a positive whole number")
    if name == "max_turn" and kw.get("deg", 0.0) < 0:
        raise SyntaxError(f"{at}max_turn: deg= must be non-negative")
    if name == "reach" and kw.get("min", 0.0) > kw["r"]:
        raise SyntaxError(f"{at}reach: min= exceeds r=")
    if name == "at" and not ({"x", "y"} & set(kw)):
        raise SyntaxError(f"{at}at needs x= or y= (or both)")
    if name == "facing" and len(args) == 2 and ({"x", "y"} & set(kw)):
        raise SyntaxError(f"{at}facing takes a target body or x=, y=, not both")
    for lo_k, hi_k in (("x0", "x1"), ("y0", "y1")):
        if lo_k in kw and hi_k in kw and kw[lo_k] >= kw[hi_k]:
            raise SyntaxError(f"{at}{lo_k}= must be smaller than {hi_k}=")


def parse_program(text: str) -> Program:
    prog = Program(source=text)
    section = "program"
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0]
        if head in ("program", "gate", "scene"):
            section = head
            continue
        if section == "gate":
            m = re.match(r"^(G\d+)\s*:\s*(.*)$", line)
            if not m or m.group(1) not in GATES:
                raise SyntaxError(f"line {ln}: gate line must be one of {sorted(GATES)}: '{line}'")
            gate = prog.gate.setdefault(m.group(1), {})
            for term in m.group(2).split(","):
                mm = re.match(r"^\s*(\w+)\s*(>=|<=|==|=)\s*(\S+)\s*$", term)
                if not mm or mm.group(1) not in GATES[m.group(1)]:
                    raise SyntaxError(f"line {ln}: cannot parse gate term '{term.strip()}' (fields: {sorted(GATES[m.group(1)])})")
                if mm.group(2) != GATES[m.group(1)][mm.group(1)]:
                    raise SyntaxError(f"line {ln}: {mm.group(1)} takes '{GATES[m.group(1)][mm.group(1)]}'")
                v = _val(mm.group(3))
                if not isinstance(v, float) or not math.isfinite(v) or v < 0:
                    raise SyntaxError(f"line {ln}: gate value must be a finite non-negative number: '{term.strip()}'")
                if mm.group(1) in gate:
                    raise SyntaxError(f"line {ln}: duplicate gate field {mm.group(1)}")
                gate[mm.group(1)] = (mm.group(2), v)
            continue
        # statements may be several per line, separated by two or more spaces
        for chunk in re.split(r"\s{2,}", line):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r"^(minimize)\s+(\w+)\((.*)\)\s*$", chunk)
            if m:
                args = [m.group(2)] + _split_args(m.group(3))
                if args != ["displacement", "*"]:
                    raise SyntaxError(f"line {ln}: only 'minimize displacement(*)' is supported")
                prog.statements.append(Statement("minimize", args, {}, ln))
                continue
            m = re.match(r"^(\w+)\((.*)\)\s*$", chunk)
            if not m:
                raise SyntaxError(f"line {ln}: cannot parse '{chunk}'")
            name, body = m.group(1), m.group(2)
            if name not in STATEMENTS:
                raise SyntaxError(f"line {ln}: unknown statement '{name}'")
            args, kw = [], {}
            for a in _split_args(body):
                if "(" in a or ")" in a:
                    raise SyntaxError(f"line {ln}: '{a}' - statements on one line must be separated by two spaces")
                if "=" in a and not a.startswith("*"):
                    k, v = a.split("=", 1)
                    k = k.strip()
                    if k not in KEYWORDS.get(name, set()):
                        raise SyntaxError(f"line {ln}: {name} does not take '{k}=' (allowed: {sorted(KEYWORDS.get(name, set()))})")
                    if k in kw:
                        raise SyntaxError(f"line {ln}: duplicate keyword '{k}='")
                    val = _val(v)
                    if k in NUMERIC_KW:
                        if not isinstance(val, float) or not math.isfinite(val):
                            raise SyntaxError(f"line {ln}: {k}= must be a finite number, got '{v.strip()}'")
                        if k not in SIGNED_KW and val < 0:
                            raise SyntaxError(f"line {ln}: {k}= must be non-negative")
                    elif k in WORDS and val not in WORDS[k].get(name, set()):
                        raise SyntaxError(f"line {ln}: {name}: {k}= must be one of {sorted(WORDS[k].get(name, set()))}")
                    kw[k] = val
                else:
                    a = a.strip()
                    if not (NAME_RE.match(a) or PATTERN_RE.match(a)):
                        raise SyntaxError(f"line {ln}: bad body name '{a}'")
                    args.append(a)
            if name == "minimize" and args != ["displacement", "*"]:
                raise SyntaxError(f"line {ln}: only 'minimize displacement(*)' is supported")
            try:
                check_statement(name, args, kw)
            except SyntaxError as e:
                raise SyntaxError(f"line {ln}: {e}: '{chunk}'") from None
            prog.statements.append(Statement(name, args, kw, ln))
    return prog
