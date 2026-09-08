"""text2function: natural language -> validated DSL program.

Backends:
  - "claude": Anthropic SDK (needs ANTHROPIC_API_KEY); see anthropic_backend().
  - "prompt": returns the prompt so a hosting agent (the skill in skills/) answers it and
    feeds the JSON back through `finish(json_text, scene)`.
"""
from __future__ import annotations

import json
import os
import re

from ..scene.model import Scene
from .schema import prompt_for, validate_program_json, ProgramError


def extract_json(text: str) -> dict:
    """First balanced top-level JSON object in `text` (models sometimes add prose around it)."""
    start = text.find("{")
    while start >= 0:
        depth = 0; in_str = False; esc = False
        for k in range(start, len(text)):
            ch = text[k]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:k + 1])
                    except json.JSONDecodeError:
                        break
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
