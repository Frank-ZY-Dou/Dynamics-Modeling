"""JSON schema + validation for text2function output (an LLM-produced DSL program).

The model returns a JSON object; `validate_program_json` type-checks it against
the scene (every body exists, containers/supports are tagged as such, numbers in
range) and turns it into DSL text for the deterministic compiler. Anything the
model left implicit is filled with a recorded default so the certificate can
show what was assumed.
"""
from __future__ import annotations

import json
import math
import re

from ..scene.model import Scene

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
                    "op": {"type": "string", "enum": [
                        "no_penetration", "place", "fixed", "on_support", "upright", "within", "inside",
                        "left_of", "right_of", "in_front_of", "behind", "min_distance", "near",
                        "minimize", "prefer"]},
                    "a": {"type": "string", "description": "primary body, or * for every free body"},
                    "b": {"type": "string", "description": "reference body / support / container / region"},
                    "margin": {"type": "number", "minimum": 0}, "gap": {"type": "number", "minimum": 0},
                    "x": {"type": "number"}, "y": {"type": "number"}, "yaw": {"type": "number"},
                    "inset": {"type": "number", "minimum": 0}, "r": {"type": "number", "minimum": 0},
                    "axis": {"type": "string", "enum": ["x", "y"]}, "w": {"type": "number", "minimum": 0},
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
NAME_RE = re.compile(r"^[A-Za-z_][\w\-]*(\.[A-Za-z_]\w*)?$")


class ProgramError(ValueError):
    pass


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
            if x == "*" and role != "a":
                raise ProgramError(f"statement {i} ({op}): '*' is only allowed as the first argument")
            if x != "*" and x.split(".")[0] not in names:
                raise ProgramError(f"statement {i} ({op}): unknown body '{x}'. Known: {sorted(names)}")
            if x != "*" and not NAME_RE.match(x):
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
            if "support" not in tags.get(b, set()) and "fixture" not in tags.get(b, set()) and b in free:
                notes.append(f"{b} used as a support but is a free body; it will be treated as movable")
            lines.append(f"  on_support({a}, {b})")
        elif op == "upright":
            need(a, "a"); lines.append(f"  upright({a})")
        elif op == "place":
            need(a, "a")
            if a == "*" or "x" not in st or "y" not in st:
                raise ProgramError(f"statement {i}: place needs one body and x, y")
            yaw = f", yaw={st['yaw']}" if "yaw" in st else ""
            w = f", w={st['w']}" if "w" in st else ""
            lines.append(f"  place({a}, x={st['x']}, y={st['y']}{yaw}{w})")
        elif op == "within":
            need(a, "a"); need(b, "region")
            lines.append(f"  within({a}, {b}, inset={st.get('inset', DEFAULTS['inset'])})")
        elif op == "inside":
            need(a, "a"); need(b, "container")
            if "container" not in tags.get(b, set()):
                notes.append(f"{b} is not tagged as a container; inside() uses its bounding footprint")
            lines.append(f"  inside({a}, {b}, inset={st['inset']})" if "inset" in st else f"  inside({a}, {b})")
        elif op in ("left_of", "right_of", "in_front_of", "behind"):
            need(a, "a"); need(b, "b")
            if a == b:
                raise ProgramError(f"statement {i}: {op} needs two different bodies")
            axis = f", axis={st['axis']}" if st.get("axis") in ("x", "y") else ""
            lines.append(f"  {op}({a}, {b}, gap={st.get('gap', DEFAULTS['gap'])}{axis})")
        elif op in ("min_distance", "near"):
            need(a, "a"); need(b, "b")
            if "r" not in st:
                raise ProgramError(f"statement {i} ({op}): needs r")
            lines.append(f"  {op}({a}, {b}, r={st['r']})")
        elif op == "minimize":
            lines.append("  minimize displacement(*)")
        elif op == "prefer":
            need(a, "a"); lines.append(f"  prefer({a}, pose=intent, w={st.get('w', 0.1)})")
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


FIELDS = {
    "no_penetration": {"a", "margin"}, "fixed": {"a"}, "on_support": {"a", "b"}, "upright": {"a"},
    "within": {"a", "b", "inset"}, "inside": {"a", "b", "inset"}, "place": {"a", "x", "y", "yaw", "w"},
    "left_of": {"a", "b", "gap", "axis"}, "right_of": {"a", "b", "gap", "axis"},
    "in_front_of": {"a", "b", "gap", "axis"}, "behind": {"a", "b", "gap", "axis"},
    "min_distance": {"a", "b", "r"}, "near": {"a", "b", "r"}, "minimize": {"a"}, "prefer": {"a", "w"},
}
NUMERIC = ("gap", "inset", "r", "margin", "w", "x", "y", "yaw")
SIGNED = ("x", "y", "yaw")


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
        for k in ("a", "b", "reason"):
            if k in st and not isinstance(st[k], str):
                raise ProgramError(f"statement {i} ({op}): '{k}' must be a string")
        for k in NUMERIC:
            if k in st:
                _number(st[k], f"statement {i} ({op}): '{k}'", signed=k in SIGNED)
        if "axis" in st and st["axis"] not in ("x", "y"):
            raise ProgramError(f"statement {i}: axis must be x or y")
    g = obj.get("gate")
    if g is not None:
        if not isinstance(g, dict):
            raise ProgramError("gate must be an object")
        for k, v in g.items():
            if k not in ("min_gap", "settle_v_max", "settle_dx"):
                raise ProgramError(f"unknown gate field '{k}'")
            _number(v, f"gate.{k}")


def api_schema(scene=None):
    """PROGRAM_SCHEMA in the subset the structured-outputs API accepts: no $schema, no numeric
    ranges (checked by _check_shape instead), additionalProperties false on every object, and,
    with a scene, body names constrained to the scene's names plus '*'."""
    import copy
    sch = copy.deepcopy(PROGRAM_SCHEMA)
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
    if scene is not None:
        names = list(scene.names())
        stp = sch["properties"]["statements"]["items"]["properties"]
        stp["a"] = {"type": "string", "enum": names + ["*"]}
        stp["b"] = {"type": "string", "enum": names + [n + ".top" for n in names]}
    return sch


def scene_summary(scene: Scene) -> str:
    """Compact scene description handed to the model."""
    rows = []
    for b in scene.bodies:
        lo, hi = b.world_aabb()
        ext = hi - lo
        line = (f"- {b.name}: {'FIXED ' if b.fixed else ''}{','.join(sorted(b.tags)) or 'object'}; "
                f"size {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m; at ({b.center[0]:.2f}, {b.center[1]:.2f}, {b.center[2]:.2f})")
        if b.fixed and "support" in b.tags:
            line += f"; top rect x[{lo[0]:.2f}, {hi[0]:.2f}] y[{lo[1]:.2f}, {hi[1]:.2f}] at z={scene.support_height(b):.3f}"
        rows.append(line)
    return "\n".join(rows)


def prompt_for(request: str, scene: Scene) -> str:
    return f"""You translate a layout request into a SimReady program: a list of geometric constraints and
objectives over named bodies. Output ONLY a JSON object matching the schema below.

Rules:
- Use only bodies listed in the scene. `*` means every free body.
- Every free body that should rest on a surface gets on_support(a=<body>, b=<support>) and, if it
  should keep its resting orientation (no roll or pitch; an asset that rests lying flat, such as a
  remote control, stays flat), upright(a=<body>).
- Frame convention (RoboLab): left_of(a,b) means a.y >= b.y + gap; in_front_of(a,b) means a.x >= b.x + gap
  (the robot looks along +x). Distances in metres.
- place(a, x, y, yaw) is a placement in the shrunken scale-space: it puts body a's reference centre at
  (x, y) on its support while every body is small and nothing touches, and keeps pulling it there as
  the scale is restored; contacts and the other statements can still move it. Use it to lay a scene
  out from your understanding of the request; coordinates in the scene frame, yaw in degrees about z,
  and the supports' top rectangles are listed with the scene. Do not place bodies on top of each other.
- Do not invent numbers; when the request is silent, omit the field and it takes a recorded default.
- Put the words that justify each statement in `reason`. List anything you assumed in `assumptions`.

Scene:
{scene_summary(scene)}

Request: {request}

Schema:
{json.dumps(PROGRAM_SCHEMA)}
"""
