"""Claude backend for text2function (Anthropic Python SDK 1.x).

Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
profile; nothing is hardcoded. The response is constrained to PROGRAM_SCHEMA with
structured outputs, so the validator only has to type-check against the scene.
"""
from __future__ import annotations

import anthropic

from .schema import PROGRAM_SCHEMA, api_schema

DEFAULT_MODEL = "claude-opus-5"
SYSTEM = ("You are the text2function step of a simulation-readiness gate for robot scenes. "
          "You translate layout requests into geometric constraint programs. You never judge "
          "geometry yourself; deterministic tools verify and repair the scene afterwards.")


def complete(prompt: str, model: str | None = None, max_tokens: int = 16000, *, scene=None) -> str:
    """Return the JSON text of the program. Raises RuntimeError on a refusal.
    `scene` restricts the body names the schema accepts to the scene's names."""
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"format": {"type": "json_schema", "schema": api_schema(scene)}},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        cat = getattr(getattr(response, "stop_details", None), "category", None)
        raise RuntimeError(f"model declined the request (category={cat})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the model ran out of output tokens before finishing the program; split the request")
    texts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    if not texts:
        raise RuntimeError(f"no text in the response (stop_reason={response.stop_reason})")
    return texts[0]
