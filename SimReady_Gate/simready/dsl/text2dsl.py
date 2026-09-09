"""text2function: natural language -> validated DSL program.

Backends:
  - "claude": Anthropic SDK (needs ANTHROPIC_API_KEY); see anthropic_backend().
  - "prompt": returns the prompt so a hosting agent (the skill in skills/) answers it and
    feeds the JSON back through `finish(json_text, scene)`.
"""
from __future__ import annotations

import json
import math
import os

from ..scene.model import Scene
from .schema import prompt_for, validate_program_json, ProgramError


MAX_JSON_CHARS = 2_000_000


def _pairs(items):
    obj = {}
    for k, v in items:
        if k in obj:
            raise ProgramError(f"duplicate JSON key '{k}'")
        obj[k] = v
    return obj


def _constant(token):
    raise ProgramError(f"'{token}' is not a valid JSON number")


def _float(token):
    v = float(token)
    if not math.isfinite(v):
        raise ProgramError(f"number out of range: {token}")
    return v


_DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)


def strict_json_loads(text: str):
    """json.loads that rejects NaN/Infinity, numbers that overflow to infinity and duplicate keys."""
    if not isinstance(text, str):
        raise ProgramError("program text must be a string")
    if len(text) > MAX_JSON_CHARS:
        raise ProgramError(f"program text exceeds {MAX_JSON_CHARS} characters")
    try:
        return _DECODER.decode(text)
    except json.JSONDecodeError as e:
        raise ProgramError(f"invalid JSON: {e}") from e


def extract_json(text: str) -> dict:
    """First top-level JSON object in `text` (models sometimes add prose or code fences around
    it), decoded by the standard library from each candidate '{' with `raw_decode`, so escaped
    quotes and braces inside strings are handled by the JSON grammar itself."""
    if not isinstance(text, str):
        raise ProgramError("model output must be text")
    if len(text) > MAX_JSON_CHARS:
        raise ProgramError(f"model output exceeds {MAX_JSON_CHARS} characters")
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ProgramError("no JSON object in model output")


def finish(json_text: str, scene: Scene):
    obj = extract_json(json_text)
    dsl, notes = validate_program_json(obj, scene)
    return dsl, notes, obj


def text2dsl(request: str, scene: Scene, backend: str = "auto", model: str | None = None):
    """Return (dsl_text, notes, raw_json) or, for backend='prompt', (None, prompt, None)."""
    if backend == "auto":
        backend = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "prompt"
    prompt = prompt_for(request, scene)
    if backend == "prompt":
        return None, prompt, None
    if backend == "claude":
        from .anthropic_backend import complete  # lazy: optional dependency
        text = complete(prompt, scene=scene, model=model)
        return finish(text, scene)
    raise ValueError(backend)
