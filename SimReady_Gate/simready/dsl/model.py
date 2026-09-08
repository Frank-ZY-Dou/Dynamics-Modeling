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

import re
from dataclasses import dataclass, field

NAME_RE = re.compile(r"^[A-Za-z_][\w\-]*(\.[A-Za-z_]\w*)?$")
KEYWORDS = {"no_penetration": {"margin"}, "within": {"inset"}, "inside": {"inset"}, "place": {"x", "y", "yaw", "w"},
            "left_of": {"gap", "axis"}, "right_of": {"gap", "axis"}, "in_front_of": {"gap", "axis"}, "behind": {"gap", "axis"},
            "min_distance": {"r"}, "near": {"r"}, "prefer": {"pose", "w"}}
STATEMENTS = {
    "no_penetration", "place", "fixed", "on_support", "upright", "within", "inside",
    "left_of", "right_of", "in_front_of", "behind", "min_distance", "near",
    "minimize", "prefer",
}


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
            if m:
                for term in m.group(2).split(","):
                    mm = re.match(r"^\s*(\w+)\s*(>=|<=|==|=)\s*([-+.\deE]+)\s*$", term)
                    if mm:
                        prog.gate.setdefault(m.group(1), {})[mm.group(1)] = (mm.group(2), float(mm.group(3)))
            else:
                prog.gate.setdefault("certify", []).append(line)
            continue
        # statements may be several per line, separated by two or more spaces
        for chunk in re.split(r"\s{2,}", line):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r"^(minimize)\s+(\w+)\((.*)\)\s*$", chunk)
            if m:
                prog.statements.append(Statement("minimize", [m.group(2)] + _split_args(m.group(3)), {}, ln))
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
                    kw[k] = _val(v)
                else:
                    a = a.strip()
                    if a != "*" and not NAME_RE.match(a):
                        raise SyntaxError(f"line {ln}: bad body name '{a}'")
                    args.append(a)
            prog.statements.append(Statement(name, args, kw, ln))
    return prog
