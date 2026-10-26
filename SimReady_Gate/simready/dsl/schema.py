"""JSON schema + validation for text2function output (an LLM-produced DSL program).

The model returns a JSON object; `validate_program_json` type-checks it against
the scene (every body exists, containers/supports are tagged as such, numbers in
range) and turns it into DSL text for the deterministic compiler. Anything the
model left implicit is filled with a recorded default so the certificate can
show what was assumed.
"""
from __future__ import annotations

import fnmatch
import json
import math
import re

import numpy as np

from ..scene.model import Scene, support_levels
from .model import check_statement, is_wildcard, GROUP, PATTERN_RE

AXIS_WORDS = ("x", "y", "+x", "-x", "+y", "-y", "long", "short", "front", "handle")   # the axis words of any statement
OPS = ["no_penetration", "place", "fixed", "on_support", "upright", "within", "inside",
       "left_of", "right_of", "in_front_of", "behind", "min_distance", "near", "minimize", "prefer",
       "reach", "clearance", "approach", "facing", "yaw", "parallel", "at", "beside",
       "row", "align", "evenly_spaced", "grid", "ring", "cluster", "symmetric", "between", "centered", "offset", "attach",
       "keep_out", "flush", "max_move", "max_turn", "stack", "balanced_on"]

PROGRAM_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SimReadyProgram",
    "type": "object",
    "required": ["statements"],
    "properties": {
        "intent": {"type": "string", "description": "one-sentence restatement of the layout intent"},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op": {"type": "string", "enum": OPS},
                    "a": {"type": "string", "description": "primary body, * for every free body, or a pattern (can_*) for the free bodies it matches"},
                    "b": {"type": "string", "description": "reference body / support / container / region; <fixture>.level<K> is one of a fixture's levels"},
                    "c": {"type": "string", "description": "third body (symmetric: the centre; between: the other end)"},
                    "members": {"type": "array", "items": {"type": "string"}, "description": "bodies of a group statement, in order; * or a pattern (can_*) stands for the free bodies it matches"},
                    "margin": {"type": "number", "minimum": 0}, "gap": {"type": "number", "minimum": 0},
                    "x": {"type": "number"}, "y": {"type": "number"}, "yaw": {"type": "number"},
                    "inset": {"type": "number", "minimum": 0}, "r": {"type": "number", "minimum": 0},
                    "axis": {"type": "string", "enum": list(AXIS_WORDS)}, "w": {"type": "number", "minimum": 0},
                    "min": {"type": "number", "minimum": 0}, "max": {"type": "number", "minimum": 0}, "length": {"type": "number", "minimum": 0},
                    "width": {"type": "number", "minimum": 0}, "toward": {"type": "string", "enum": ["robot", "near", "far", "left", "right"]},
                    "tol": {"type": "number", "minimum": 0}, "deg": {"type": "number"}, "mod": {"type": "number", "enum": [90, 180, 360]},
                    "cols": {"type": "integer", "minimum": 1}, "dx": {"type": "number"}, "dy": {"type": "number"},
                    "frame": {"type": "string", "enum": ["world", "local"]}, "edge": {"type": "string", "enum": ["min", "center", "max"]},
                    "order": {"type": "string", "enum": ["height", "-height", "width", "-width", "length", "-length", "mass", "-mass"]},
                    "spacing": {"type": "string", "enum": ["even"]},
                    "start": {"type": "string", "enum": ["left", "right", "near", "far"], "description": "row: the end the first member is at (left or right for a line along y, near or far along x)"},
                    "mid": {"type": "number", "description": "row: where along its axis the midpoint of the two end members' centres lies"},
                    "angle": {"type": "number", "description": "ring: the angle of the first member about the centre, degrees from +x"},
                    "face": {"type": "string", "enum": ["in", "out"], "description": "ring: each member's axis toward or away from the centre"},
                    "side": {"type": "string", "enum": ["left", "right", "near", "far", "center"]},
                    "frac": {"type": "number", "minimum": 0, "maximum": 1}, "depth": {"type": "number", "minimum": 0},
                    "x0": {"type": "number"}, "x1": {"type": "number"}, "y0": {"type": "number"}, "y1": {"type": "number"},
                    "d": {"type": "number", "minimum": 0}, "soft": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "description": "which words in the request justify this statement"}
                },
                "additionalProperties": False
            }
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "gate": {"type": "object", "properties": {
            "min_gap": {"type": "number"}, "settle_v_max": {"type": "number"}, "settle_dx": {"type": "number"}}}
    },
    "additionalProperties": False
}

DEFAULTS = {"margin": 0.005, "inset": 0.02, "gap": 0.05}
# defaults of the v1 statements, applied by the compiler when a keyword is omitted (recorded in certificates)
OP_DEFAULTS = {
    "reach": {"min": 0.0, "x": 0.0, "y": 0.0},
    "at": {"tol": 0.01},
    "beside": {"gap": 0.0},
    "approach": {"length": 0.20, "width": 0.10, "toward": "robot", "x": 0.0, "y": 0.0},
    "facing": {"axis": "+x", "tol": 15.0},
    "yaw": {"tol": 5.0, "mod": 360.0, "axis": "x"},
    "parallel": {"mod": 180.0, "tol": 5.0, "axis": "long"},
    "row": {"axis": "y", "gap": 0.01, "tol": 0.01, "edge": "center"},
    "align": {"axis": "x", "edge": "center", "tol": 0.005},
    "evenly_spaced": {"axis": "y", "tol": 0.005},
    "grid": {"tol": 0.01},
    "ring": {"tol": 0.01, "axis": "+x"},
    "symmetric": {"axis": "y", "tol": 0.01},
    "between": {"axis": "y", "gap": 0.0},
    "centered": {"tol": 0.01},
    "offset": {"dx": 0.0, "dy": 0.0, "frame": "world", "tol": 0.01},
    "attach": {"tol": 0.01, "deg": 2.0},
    "keep_out": {"gap": 0.0},
    "flush": {"gap": 0.0, "tol": 0.005},
    "region": {"frac": 0.5},
    "stack": {"tol": 0.01},
    "balanced_on": {"margin": 0.01},
}
NAME_RE = re.compile(r"^[A-Za-z_][\w\-]*(\.[A-Za-z_]\w*)?$")


class ProgramError(ValueError):
    pass


FIELDS = {
    "no_penetration": {"a", "margin"}, "fixed": {"a"}, "on_support": {"a", "b"}, "upright": {"a"},
    "within": {"a", "b", "inset", "side", "frac", "depth", "x0", "x1", "y0", "y1", "soft"},
    "inside": {"a", "b", "inset", "soft"}, "place": {"a", "x", "y", "yaw", "w"},
    "left_of": {"a", "b", "gap", "axis", "soft"}, "right_of": {"a", "b", "gap", "axis", "soft"},
    "in_front_of": {"a", "b", "gap", "axis", "soft"}, "behind": {"a", "b", "gap", "axis", "soft"},
    "min_distance": {"a", "b", "r", "soft"}, "near": {"a", "b", "r", "soft"}, "minimize": {"a"}, "prefer": {"a", "w"},
    "reach": {"a", "r", "min", "x", "y", "soft"}, "clearance": {"a", "gap"}, "at": {"a", "x", "y", "tol", "soft"},
    "beside": {"a", "b", "side", "gap", "max", "soft"},
    "approach": {"a", "length", "width", "toward", "x", "y", "soft"},
    "facing": {"a", "b", "axis", "tol", "x", "y", "soft"}, "yaw": {"a", "deg", "tol", "mod", "axis", "soft"},
    "parallel": {"a", "b", "mod", "tol", "axis", "soft"},
    "row": {"members", "axis", "start", "gap", "tol", "order", "spacing", "mid", "edge", "soft"}, "align": {"members", "axis", "edge", "tol", "soft"},
    "evenly_spaced": {"members", "axis", "tol", "soft"}, "grid": {"members", "cols", "dx", "dy", "tol", "soft"},
    "ring": {"members", "r", "x", "y", "angle", "face", "axis", "tol", "soft"},
    "cluster": {"members", "r", "soft"}, "symmetric": {"a", "b", "c", "axis", "tol", "soft"},
    "between": {"a", "b", "c", "axis", "gap", "soft"}, "centered": {"a", "b", "tol", "soft"},
    "offset": {"a", "b", "dx", "dy", "frame", "tol", "soft"}, "attach": {"a", "b", "tol", "soft"},
    "keep_out": {"a", "b", "gap", "side", "frac", "depth", "x0", "x1", "y0", "y1", "soft"},
    "flush": {"a", "b", "side", "gap", "tol", "soft"},
    "max_move": {"a", "d", "soft"}, "max_turn": {"a", "deg", "soft"}, "stack": {"members", "tol"},
    "balanced_on": {"a", "b", "margin", "soft"},
}
NUMERIC = ("gap", "inset", "r", "margin", "w", "x", "y", "yaw", "min", "max", "length", "width", "tol", "deg", "mod", "cols",
           "dx", "dy", "frac", "depth", "x0", "x1", "y0", "y1", "d", "mid", "angle", "soft")
SIGNED = ("x", "y", "yaw", "deg", "dx", "dy", "x0", "x1", "y0", "y1", "mid", "angle")
WORD_FIELDS = ("axis", "edge", "side", "toward", "frame", "order", "spacing", "start", "face")
# the order keywords are written in the DSL text of a v1 statement
KW_ORDER = ("side", "r", "min", "gap", "max", "length", "width", "toward", "axis", "start", "order", "spacing", "mid", "edge", "cols", "dx", "dy", "frame", "deg", "mod", "angle", "face",
            "tol", "inset", "frac", "depth", "x0", "x1", "y0", "y1", "x", "y", "d", "margin", "soft")


def _fmt(v):
    """A number as the DSL text writes it (integers without a fraction, floats by repr)."""
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) < 1e15):
        return str(int(v))
    return repr(float(v))


def validate_program_json(obj: dict, scene: Scene) -> tuple[str, list[str]]:
    """Return (dsl_text, notes). Raises ProgramError with a fix-it message."""
    _check_shape(obj)
    names = set(scene.names()); free = {b.name for b in scene.free()}
    tags = {b.name: b.tags for b in scene.bodies}
    notes = list(obj.get("assumptions", []))
    lines = ["program"]
    seen_nopen = False
    for i, st in enumerate(obj["statements"]):
        op = st["op"]; a = st.get("a"); b = st.get("b")

        def need(x, role):
            if x is None:
                raise ProgramError(f"statement {i} ({op}): missing {role}")
            if not isinstance(x, str):
                raise ProgramError(f"statement {i} ({op}): {role} must be a body name")
            if is_wildcard(x):
                if role not in ("a", "member"):
                    raise ProgramError(f"statement {i} ({op}): '*' and name patterns are only allowed in a and in members")
                if not PATTERN_RE.match(x):
                    raise ProgramError(f"statement {i} ({op}): pattern '{x}' cannot be written in the DSL")
                if x != "*" and not any(fnmatch.fnmatchcase(n, x) for n in free):
                    raise ProgramError(f"statement {i} ({op}): '{x}' matches no free body. Free: {sorted(free)}")
                return
            if x.split(".")[0] not in names:
                raise ProgramError(f"statement {i} ({op}): unknown body '{x}'. Known: {sorted(names)}")
            if not NAME_RE.match(x):
                raise ProgramError(f"statement {i} ({op}): body name '{x}' cannot be written in the DSL")
        if b is not None and a == b:
            raise ProgramError(f"statement {i}: {op} needs two different bodies")
        if op == "no_penetration":
            seen_nopen = True
            lines.append(f"  no_penetration(*, margin={st.get('margin', DEFAULTS['margin'])})")
        elif op == "fixed":
            need(a, "a"); lines.append(f"  fixed({a})")
        elif op == "on_support":
            need(a, "a"); need(b, "support")
            if "support" not in tags.get(b.split(".")[0], set()) and "fixture" not in tags.get(b.split(".")[0], set()) and b in free:
                notes.append(f"{b} used as a support but is a free body; it will be treated as movable")
            lines.append(f"  on_support({a}, {b})")
        elif op == "upright":
            need(a, "a"); lines.append(f"  upright({a})")
        elif op == "place":
            need(a, "a")
            if is_wildcard(a) or "x" not in st or "y" not in st:
                raise ProgramError(f"statement {i}: place needs one body and x, y")
            yaw = f", yaw={st['yaw']}" if "yaw" in st else ""
            w = f", w={st['w']}" if "w" in st else ""
            lines.append(f"  place({a}, x={st['x']}, y={st['y']}{yaw}{w})")
        elif op == "inside":
            need(a, "a"); need(b, "container")
            if "container" not in tags.get(b, set()):
                notes.append(f"{b} is not tagged as a container; inside() uses its bounding footprint")
            kws = [f"inset={st['inset']}"] if "inset" in st else []
            if "soft" in st:
                kws.append(f"soft={_fmt(st['soft'])}")
            lines.append(f"  inside({a}, {b}{''.join(', ' + k for k in kws)})")
        elif op in ("left_of", "right_of", "in_front_of", "behind"):
            need(a, "a"); need(b, "b")
            if a == b:
                raise ProgramError(f"statement {i}: {op} needs two different bodies")
            axis = f", axis={st['axis']}" if st.get("axis") in ("x", "y") else ""
            if "axis" in st and st["axis"] not in ("x", "y"):
                raise ProgramError(f"statement {i} ({op}): axis must be x or y")
            soft = f", soft={_fmt(st['soft'])}" if "soft" in st else ""
            lines.append(f"  {op}({a}, {b}, gap={st.get('gap', DEFAULTS['gap'])}{axis}{soft})")
        elif op in ("min_distance", "near"):
            need(a, "a"); need(b, "b")
            if "r" not in st:
                raise ProgramError(f"statement {i} ({op}): needs r")
            soft = f", soft={_fmt(st['soft'])}" if "soft" in st else ""
            lines.append(f"  {op}({a}, {b}, r={st['r']}{soft})")
        elif op == "minimize":
            lines.append("  minimize displacement(*)")
        elif op == "prefer":
            need(a, "a"); lines.append(f"  prefer({a}, pose=intent, w={st.get('w', 0.1)})")
        elif op in FIELDS:
            lines.append("  " + _v1_line(i, op, st, need, tags, notes))
        else:
            raise ProgramError(f"statement {i}: unknown op {op}")
    if not seen_nopen:
        lines.insert(1, f"  no_penetration(*, margin={DEFAULTS['margin']})")
        notes.append("no_penetration added by default")
    g = obj.get("gate") or {}
    if g:
        lines.append("gate")
        if "min_gap" in g:
            lines.append(f"  G2: min_gap >= {g['min_gap']}")
        if "settle_v_max" in g or "settle_dx" in g:
            parts = []
            if "settle_v_max" in g:
                parts.append(f"v_max <= {g['settle_v_max']}")
            if "settle_dx" in g:
                parts.append(f"dx <= {g['settle_dx']}")
            lines.append("  G5: " + ", ".join(parts))
    return "\n".join(lines), notes


def _v1_line(i, op, st, need, tags, notes):
    """DSL text of a v1 statement (and `within` with a region), after the checks the text parser
    applies, so that an error names the JSON statement rather than a line of generated text."""
    if op in GROUP:
        members = st.get("members")
        if not isinstance(members, list) or not members:
            raise ProgramError(f"statement {i} ({op}): needs 'members', a list of body names")
        for m in members:
            need(m, "member")
            if "." in m:
                raise ProgramError(f"statement {i} ({op}): members are bodies, got '{m}'")
        args = list(members)
    else:
        need(st.get("a"), "a")
        args = [st["a"]]
        if "b" in st:
            need(st["b"], "b")
            args.append(st["b"])
        elif op in ("parallel", "centered", "offset", "attach", "balanced_on", "symmetric", "between", "flush"):
            raise ProgramError(f"statement {i} ({op}): missing b")
        if op in ("symmetric", "between"):
            if "c" not in st:
                raise ProgramError(f"statement {i} ({op}): missing c")
            need(st["c"], "c")
            args.append(st["c"])
        elif "c" in st:
            raise ProgramError(f"statement {i} ({op}): field 'c' does not belong to {op}")
    if op == "within" and len(args) == 2 and "inset" not in st and not (set(st) & {"side", "x0", "x1", "y0", "y1"}):
        st = dict(st, inset=DEFAULTS["inset"])
    kw = {k: st[k] for k in KW_ORDER if k in st}
    try:
        check_statement(op, args, {k: (float(v) if k in NUMERIC else v) for k, v in kw.items()})
    except SyntaxError as e:
        raise ProgramError(f"statement {i} ({op}): {e}") from None
    if op == "stack" and any("support" in tags.get(m, set()) for m in args[1:]):
        notes.append(f"stack({', '.join(args)}): a member above the first is tagged as a support; it is stacked like any free body")
    text = ", ".join(args + [f"{k}={_fmt(v) if k in NUMERIC else v}" for k, v in kw.items()])
    return f"{op}({text})"


def _number(v, label, signed=False):
    """Every number that reaches the DSL text is a finite int/float (bools, strings, NaN and
    infinities are rejected here, before anything is interpolated into a statement)."""
    try:
        finite = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))
    except OverflowError:          # an integer too large for a float
        finite = False
    if not finite:
        raise ProgramError(f"{label} must be a finite number")
    if not signed and v < 0:
        raise ProgramError(f"{label} must be non-negative")


def _check_shape(obj):
    """Structural validation that does not need jsonschema: types, ranges, allowed keys.
    Every field is checked for the op it belongs to; a value of the wrong type is an error, never
    something that is passed through into the program text."""
    if not isinstance(obj, dict) or not isinstance(obj.get("statements"), list):
        raise ProgramError("program must be an object with a 'statements' list")
    extra = set(obj) - {"statements", "intent", "assumptions", "gate"}
    if extra:
        raise ProgramError(f"unknown program fields: {sorted(extra)}")
    if "intent" in obj and not isinstance(obj["intent"], str):
        raise ProgramError("intent must be a string")
    if "assumptions" in obj and (not isinstance(obj["assumptions"], list)
                                 or not all(isinstance(s, str) for s in obj["assumptions"])):
        raise ProgramError("assumptions must be a list of strings")
    for i, st in enumerate(obj["statements"]):
        if not isinstance(st, dict) or not isinstance(st.get("op"), str) or st["op"] not in FIELDS:
            raise ProgramError(f"statement {i}: 'op' must be one of {sorted(FIELDS)}")
        op = st["op"]
        unknown = set(st) - FIELDS[op] - {"op", "reason"}
        if unknown:
            raise ProgramError(f"statement {i} ({op}): fields {sorted(unknown)} do not belong to {op}")
        for k in ("a", "b", "c", "reason") + WORD_FIELDS:
            if k in st and not isinstance(st[k], str):
                raise ProgramError(f"statement {i} ({op}): '{k}' must be a string")
        if "members" in st and (not isinstance(st["members"], list) or not all(isinstance(m, str) for m in st["members"])):
            raise ProgramError(f"statement {i} ({op}): 'members' must be a list of body names")
        for k in NUMERIC:
            if k in st:
                _number(st[k], f"statement {i} ({op}): '{k}'", signed=k in SIGNED)
        if "axis" in st and st["axis"] not in AXIS_WORDS:
            raise ProgramError(f"statement {i}: axis must be one of {', '.join(AXIS_WORDS)}")
    g = obj.get("gate")
    if g is not None:
        if not isinstance(g, dict):
            raise ProgramError("gate must be an object")
        for k, v in g.items():
            if k not in ("min_gap", "settle_v_max", "settle_dx"):
                raise ProgramError(f"unknown gate field '{k}'")
            _number(v, f"gate.{k}")


def api_schema(scene=None, ops=None):
    """PROGRAM_SCHEMA in the subset the structured-outputs API accepts: no $schema, no numeric
    ranges (checked by _check_shape instead), additionalProperties false on every object, and,
    with a scene, body names constrained to the scene's names plus '*' and the patterns of its numbered
    names (name_patterns); with `ops`, restricted to
    those statements (program_schema)."""
    sch = program_schema(ops)
    sch.pop("$schema", None)

    def walk(node):
        if isinstance(node, dict):
            for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                node.pop(k, None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(sch)
    stp = sch["properties"]["statements"]["items"]["properties"]
    if "mod" in stp:
        stp["mod"] = {"type": "number"}                 # numeric enums are checked by _check_shape
    if scene is not None:
        names = list(scene.names())
        stems = name_patterns(scene)
        stp["a"] = {"type": "string", "enum": names + ["*"] + stems}
        stp["b"] = {"type": "string", "enum": names + [n + ".top" for n in names] + level_refs(scene)}
        if "c" in stp:
            stp["c"] = {"type": "string", "enum": names}
        if "members" in stp:
            stp["members"] = {"type": "array", "items": {"type": "string", "enum": names + ["*"] + stems}}
    return sch


def level_refs(scene: Scene) -> list:
    """`name.levelK` of every fixture with more than one level (scene.model.support_levels), for the schema's references."""
    out = []
    for b in scene.fixed_bodies():
        if "support" in b.tags:
            levels = support_levels(scene, b)
            if len(levels) > 1:
                out += [f"{b.name}.level{lv.index}" for lv in levels]
    return out


def name_patterns(scene: Scene) -> list:
    """The patterns a scene's numbered names suggest: can_* for can_1, can_2, ... (two or more free
    bodies), for the name enums of a schema bound to that scene."""
    stems = {}
    for b in scene.free():
        m = re.match(r"^(.*?)\d+$", b.name)
        if m and m.group(1):
            stems[m.group(1)] = stems.get(m.group(1), 0) + 1
    return [f"{p}*" for p, k in sorted(stems.items()) if k >= 2]


def op_schema(scene=None, ops=None) -> dict:
    """A program schema with one branch per statement (anyOf on `op`): each branch takes only its
    statement's fields and requires the ones it cannot do without, so a structured answer cannot
    give a statement another statement's field. Body names are the scene's when a scene is given."""
    from .model import REQUIRED, GROUP, WORDS
    flat = api_schema(scene, ops)
    props = flat["properties"]["statements"]["items"]["properties"]
    ops = list(ops or OPS)
    table = V0_FIELDS if set(ops) <= set(V0_FIELDS) else FIELDS
    branches = []
    for op in ops:
        fields = {"op": {"type": "string", "const": op}}
        for f in sorted(table[op]):
            fields[f] = props[f]
            if f in WORDS and op in WORDS[f]:
                fields[f] = {"type": "string", "enum": sorted(WORDS[f][op])}     # the words this statement takes
        if "reason" in props:
            fields["reason"] = props["reason"]
        need = ["op"] + (["members"] if op in GROUP else ["a"] if "a" in table[op] and op not in ("no_penetration", "minimize") else [])
        if op in ("on_support", "inside", "left_of", "right_of", "in_front_of", "behind", "min_distance", "near", "parallel",
                  "centered", "offset", "attach", "balanced_on", "beside", "symmetric", "between", "flush"):
            need.append("b")
        if op in ("symmetric", "between"):
            need.append("c")
        if op == "place":
            need += ["x", "y"]
        need += sorted(k for k in REQUIRED.get(op, set()) if k in table[op])
        branches.append({"type": "object", "properties": fields, "required": need, "additionalProperties": False})
    flat["properties"]["statements"]["items"] = {"anyOf": branches}
    return flat


def scene_summary(scene: Scene) -> str:
    """Compact scene description handed to the model."""
    rows = []
    for b in scene.bodies:
        lo, hi = b.world_aabb()
        ext = hi - lo
        kind = ','.join(sorted(b.tags)) or 'object'
        label = b.meta.get("label")
        line = (f"- {b.name}: {'FIXED ' if b.fixed else ''}{kind}{f' ({label})' if label else ''}; "
                f"size {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m; at ({b.center[0]:.2f}, {b.center[1]:.2f}, {b.center[2]:.2f})")
        levels = support_levels(scene, b) if b.fixed and "support" in b.tags else []
        if len(levels) > 1:
            line += "; levels from the bottom: " + ", ".join(
                f"{b.name}.level{lv.index} z={lv.z:.3f} x[{lv.x0:.2f}, {lv.x1:.2f}] y[{lv.y0:.2f}, {lv.y1:.2f}] "
                + ("open above" if math.isinf(lv.headroom) else f"headroom {lv.headroom:.3f} m") for lv in levels)
        elif b.fixed and "support" in b.tags:
            line += f"; top rect x[{lo[0]:.2f}, {hi[0]:.2f}] y[{lo[1]:.2f}, {hi[1]:.2f}] at z={scene.support_height(b):.3f}"
        if not b.fixed:
            # the body's own axes: what yaw, facing and parallel are about
            loc = np.ptp(b.verts, axis=0)
            yaw = math.degrees(math.atan2(b.rotation[1, 0], b.rotation[0, 0]))
            line += f"; local size {loc[0]:.2f}x{loc[1]:.2f}x{loc[2]:.2f} m; heading {yaw:.0f} deg"
        mass = (b.meta.get("physics") or {}).get("mass")
        if not b.fixed and mass:
            line += f"; mass {mass:.2f} kg"
        if b.meta.get("grasp_axis"):
            line += f"; handle/grasp side {b.meta['grasp_axis']}"
        if b.meta.get("front_axis"):
            line += f"; front side {b.meta['front_axis']}"
        rows.append(line)
    return "\n".join(rows)


LANGUAGE = """Statements (a, b, c are body names; members is a list of names, in order; in a and in members,
* is every free body and a pattern such as can_* or fruit_? every free body whose name matches, in
natural order (can_2 before can_10), leaving out bodies the statement names itself):
  no_penetration(margin)                 nothing interpenetrates; margin is the clearance the solver keeps
  fixed(a)                               a does not move
  on_support(a, b)   upright(a)          a rests on b (b.levelK: on level K of a fixture, from 1 at the bottom,
                                         which must leave a room above for a's height); a keeps its resting
                                         orientation (no roll or pitch)
  within(a, b, inset)                    a's footprint inside b (b.top: b's top rectangle; b.levelK: that level's
                                         rectangle). A part of b:
                                         side=left|right|near|far|center with frac (default 0.5); a strip beside
                                         b: side with depth. x0, x1, y0, y1 bound the region (any subset; alone
                                         they define it)
  inside(a, b)                           a inside container b
  keep_out(a, b, gap)                    a's footprint stays outside a region (region fields as for within)
  flush(a, b, side, gap, tol)            a's footprint edge on side (near|far|left|right) gap inside the same edge of
                                         b's rectangle (b, b.top or b.levelK), within tol (default 0.005)
  beside(a, b, side, gap, max)           a's footprint on b's side (left|right|near|far) with between gap and
                                         max of table between the two footprints (max optional)
  left_of | right_of | in_front_of | behind (a, b, gap)    relations of the centres, gap between centres
  near(a, b, r) | min_distance(a, b, r)  centre distance at most / at least r
  at(a, x, y, tol)                       a's centre at x and/or y, within tol (default 0.01)
  reach(a, r, min)                       a's centre within r of the robot base (and at least min from it);
                                         x, y move the base
  clearance(a, gap)                      every other body at least gap from a's surface (room for a gripper)
  approach(a, length, width, toward)     nothing in the corridor of that length and width from a toward the
                                         robot (toward=robot, default) or toward near|far|left|right
  facing(a, b, axis, tol)                a's local axis (+x, -x, +y, -y, or front / handle: the side the scene
                                         summary gives) points at b, or at x, y, or by default at the robot,
                                         within tol degrees
  yaw(a, deg, tol, mod, axis)            the heading of a's local +x is deg (axis=long: of its long side, a
                                         line, so deg and deg+180 are the same; axis=front or handle: of that
                                         side); mod 90 or 180 for shapes that repeat when turned
  parallel(a, b, tol)                    a's and b's long sides parallel (axis=x: their local x axes)
  row(members, axis, start, gap, order, spacing, mid, edge)  on one line along axis (x or y, default y), in the
                                         order listed from the end start names (left|right along y, near|far
                                         along x; default right, near), footprints at least gap apart;
                                         order=height|width|length|mass (- for descending) sorts the members
                                         first, from the bodies' sizes and masses;
                                         spacing=even makes every centre spacing equal to the mean within tol;
                                         mid puts the midpoint of the two end members' centres at that y (x for
                                         an x row), whichever members the order puts at the ends; edge=min|max
                                         lines up the footprints' near (min) or far (max) edges across the row
                                         instead of the centres (spines in line)
  align(members, axis, edge)             equal x (or y) of the centres, or of the footprints' min or max edge
  evenly_spaced(members, axis, tol)      every centre spacing along axis within tol of the mean spacing
  grid(members, cols, dx, dy)            rows of cols members: member k at (row*dx, col*dy) from the first;
                                         without dx, dy the pitch is free but shared
  ring(members, r, x, y, angle, face, axis)  the members evenly spaced on a circle of radius r about (x, y)
                                         (default: their own centre), counter-clockwise in the order listed
                                         from angle (degrees from +x; default: the angle that moves them
                                         least); face=in|out turns each one's axis (default +x; front,
                                         handle) toward or away from the centre
  cluster(members, r)                    every member within r of the group's centre
  symmetric(a, b, c, axis)               a and b mirror images about c along axis (equal other coordinate)
  between(a, b, c, axis, gap)            a's centre between b's and c's along axis, gap from each
  centered(a, b, tol)                    a's centre over b's centre (a cup on its saucer)
  offset(a, b, dx, dy, frame)            a's centre at (dx, dy) from b's, in the scene frame or, with
                                         frame=local, in b's own frame
  attach(a, b)                           a keeps its present pose relative to b
  stack(members)                         each member rests centred on the one before it
  balanced_on(a, b, margin)              a's centre of mass over b's top, margin from its edge
  max_move(a, d)  max_turn(a, deg)       a moves at most d metres / turns at most deg from where it is
  place(a, x, y, yaw)                    a target pose in the shrunken scale-space (a soft pull)
  minimize                               minimum displacement
Any statement other than no_penetration, fixed, on_support, upright, place, clearance and stack takes
soft (0 to 1]: a preference with that weight, which may yield; it is reported but not required."""


V0_OPS = ["no_penetration", "place", "fixed", "on_support", "upright", "within", "inside", "left_of", "right_of",
          "in_front_of", "behind", "min_distance", "near", "minimize", "prefer"]
LANGUAGE_V0 = """Statements (a, b are body names):
  no_penetration(margin)                 nothing interpenetrates; margin is the clearance the solver keeps
  fixed(a)                               a does not move
  on_support(a, b)   upright(a)          a rests on b; a keeps its resting orientation (no roll or pitch)
  within(a, b, inset)                    a's footprint inside b (b.top: b's top rectangle), inset from its edge
  inside(a, b)                           a inside container b
  beside(a, b, side, gap, max)           a's footprint on b's side (left|right|near|far) with between gap and
                                         max of table between the two footprints (max optional)
  left_of | right_of | in_front_of | behind (a, b, gap)    relations of the centres, gap between centres
  near(a, b, r) | min_distance(a, b, r)  centre distance at most / at least r
  place(a, x, y, yaw)                    a target pose in the shrunken scale-space (a soft pull)
  prefer(a, w)                           a pull toward a's present pose
  minimize                               minimum displacement"""


# the fields each statement took before v1 (no regions, no preferences)
V0_FIELDS = {"no_penetration": {"a", "margin"}, "fixed": {"a"}, "on_support": {"a", "b"}, "upright": {"a"},
             "within": {"a", "b", "inset"}, "inside": {"a", "b", "inset"}, "place": {"a", "x", "y", "yaw", "w"},
             "left_of": {"a", "b", "gap", "axis"}, "right_of": {"a", "b", "gap", "axis"}, "in_front_of": {"a", "b", "gap", "axis"},
             "behind": {"a", "b", "gap", "axis"}, "min_distance": {"a", "b", "r"}, "near": {"a", "b", "r"}, "minimize": {"a"},
             "prefer": {"a", "w"}}


def program_schema(ops=None) -> dict:
    """PROGRAM_SCHEMA restricted to some statements (their ops and the fields they take; the v0
    statements with their v0 fields)."""
    import copy
    sch = copy.deepcopy(PROGRAM_SCHEMA)
    if ops is None:
        return sch
    item = sch["properties"]["statements"]["items"]
    item["properties"]["op"]["enum"] = list(ops)
    table = V0_FIELDS if set(ops) <= set(V0_FIELDS) else FIELDS
    keep = {"op", "reason"}.union(*(table[o] for o in ops))
    item["properties"] = {k: v for k, v in item["properties"].items() if k in keep}
    return sch


V1_NOTES = """
- Table between two objects ("2 cm between the plate and the fork") is a gap between footprints:
  beside, row and keep_out measure footprints; left_of, near, min_distance and reach measure centres.
- To sort by height, width, length or mass, give row an order; the compiler sorts from the sizes.
  A row already aligns its members; spacing=even spaces them evenly in that order and mid centres
  it. Do not repeat a row's members in align or evenly_spaced."""


def system_prompt(ops=None, language: str | None = None, with_schema: bool = True, reasons: bool = True) -> str:
    """The fixed part of the text-to-program prompt: frame, rules, the statements and the schema
    (restricted to `ops` when given, with that statement list). Without `with_schema` the schema is
    left out, for a caller that hands it to the model as the structured-output format instead;
    without `reasons` the justifications are optional (an answer with only the program)."""
    language = language or (LANGUAGE if ops is None else LANGUAGE_V0)
    place_rule = ("Say what must hold and let the solver find the poses. " if ops is None else "")
    wild_rule = ("A group of similar bodies is a pattern: members=[\"can_*\",\n  \"bottle_1\"] lists every can, then the bottle."
                 if ops is None else "`*` means every free body (first argument only).")
    return f"""You translate a layout request into a SimReady program: a list of geometric constraints and
objectives over named bodies. Output ONLY a JSON object matching the schema below.

Frame: metres and degrees, z up. The robot base is at (0, 0) and looks along +x: +x is away from the
robot ("far"), -x toward it ("near"), +y is left, -y right. RoboLab convention: left_of(a,b) means
a.y >= b.y + gap; in_front_of(a,b) means a.x >= b.x + gap.

Rules:
- Use only bodies listed in the scene. {wild_rule}
- Every free body that should rest on a surface gets on_support(a=<body>, b=<support>) and, if it
  should keep its resting orientation (an asset that rests lying flat, such as a remote control,
  stays flat), upright(a=<body>).
- {place_rule}place(a, x, y, yaw) puts a body's reference centre at (x, y) on its support at the
  configured shrink scale, where overlaps may remain, and keeps pulling it there as the scale is
  restored; use it for a rough position the request implies and no statement expresses. Do not place
  bodies on top of each other{"; use stack or on_support for that" if ops is None else ""}.
- Keep within(a=*, b=<support>.top) unless the request lets bodies leave the support.{V1_NOTES if ops is None else ""}
- Do not invent numbers; when the request is silent, omit the field and it takes a recorded default.
- {"Put the words that justify each statement in `reason`. List anything you assumed in `assumptions`." if reasons else
   "Answer with the statements only: `reason` and `assumptions` are optional, leave them out."}

{language}
{("" if not with_schema else chr(10) + "Schema:" + chr(10) + json.dumps(program_schema(ops)) + chr(10))}"""


def user_prompt(request: str, scene: Scene) -> str:
    return f"""Scene:
{scene_summary(scene)}

Request: {request}
"""


def prompt_for(request: str, scene: Scene) -> str:
    return system_prompt() + "\n" + user_prompt(request, scene)
