"""SimReady DSL v0: a typed program of constraints and objectives over bodies.

Text form (one statement per line, '#' comments, `*` = every free body):

    program
      no_penetration(*, margin=0.01)
      fixed(table)
      on_support(mug, table)      upright(mug)
      within(mug, table.top, inset=0.02)
      place(mug, x=0.45, y=0.10, yaw=30)      # a target pose in the shrunken scale-space (soft)
      inside(mug, tray)
      left_of(mug, bowl, axis=y, gap=0.05)
      min_distance(mug, bowl, r=0.10)
      near(mug, bowl, r=0.25)
      minimize displacement(*)
      prefer(bowl, pose=intent, w=0.1)
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
KEYWORDS = {"no_penetration": {"margin"}, "within": {"inset"}, "inside": {"inset"}, "place": {"x", "y", "yaw", "w"},
            "left_of": {"gap", "axis"}, "right_of": {"gap", "axis"}, "in_front_of": {"gap", "axis"}, "behind": {"gap", "axis"},
            "min_distance": {"r"}, "near": {"r"}, "prefer": {"pose", "w"}}
NUMERIC_KW = {"margin", "inset", "x", "y", "yaw", "w", "gap", "r"}
SIGNED_KW = {"x", "y", "yaw"}
STATEMENTS = {
    "no_penetration", "place", "fixed", "on_support", "upright", "within", "inside",
    "left_of", "right_of", "in_front_of", "behind", "min_distance", "near",
    "minimize", "prefer",
}
# positional arguments each statement takes: (min, max); None = unbounded
ARITY = {"no_penetration": (1, 1), "place": (1, 1), "fixed": (1, None), "on_support": (2, 2), "upright": (1, 1),
         "within": (2, 2), "inside": (2, 2), "left_of": (2, 2), "right_of": (2, 2), "in_front_of": (2, 2),
         "behind": (2, 2), "min_distance": (2, 2), "near": (2, 2), "minimize": (2, 2), "prefer": (1, 1)}
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
                    elif k == "axis" and val not in ("x", "y"):
                        raise SyntaxError(f"line {ln}: axis must be x or y")
                    elif k == "pose" and val != "intent":
                        raise SyntaxError(f"line {ln}: only pose=intent is supported")
                    kw[k] = val
                else:
                    a = a.strip()
                    if a != "*" and not NAME_RE.match(a):
                        raise SyntaxError(f"line {ln}: bad body name '{a}'")
                    args.append(a)
            lo_n, hi_n = ARITY[name]
            if len(args) < lo_n or (hi_n is not None and len(args) > hi_n):
                want = f"{lo_n}" if lo_n == hi_n else (f"at least {lo_n}" if hi_n is None else f"{lo_n}-{hi_n}")
                raise SyntaxError(f"line {ln}: {name} takes {want} argument(s), got {len(args)}: '{chunk}'")
            if name == "minimize" and args != ["displacement", "*"]:
                raise SyntaxError(f"line {ln}: only 'minimize displacement(*)' is supported")
            prog.statements.append(Statement(name, args, kw, ln))
    return prog
