"""One isolated `claude -p` call per request: the protocol of the comparison.

Every call is a new process, started by this program with:
  --safe-mode                 CLAUDE.md, skills, plugins, hooks, MCP servers, custom agents and
                              output styles off; authentication and model selection as usual
  --no-session-persistence    the session is not written to disk; nothing can be resumed
  --tools ""                  no built-in tool: the model reads the message and answers
  --strict-mcp-config         no MCP server (none is configured for the call)
  --system-prompt-file        the condition's fixed system prompt replaces Claude Code's own
  --model, --effort           fixed for a whole run
  --json-schema               the condition's answer format (structured output)
  --output-format json        the answer with its token usage, cost and timing
in a new, empty working directory outside any repository, with an environment built from an
allowlist (HOME, PATH, locale) plus CLAUDE_CODE_DISABLE_AUTO_MEMORY=1: nothing from the session
that runs the experiment (its CLAUDE_* variables, its messaging socket) reaches the call. The user
message is built from the task files alone; no history crosses trials, and a later round of a trial
states its earlier answers in the message itself. Every call's input and output is kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ENV_KEEP = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "SHELL", "TMPDIR")
FLAGS = ["--safe-mode", "--no-session-persistence", "--tools", "", "--strict-mcp-config"]


def isolated_env() -> dict:
    env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return env


def cli_version() -> str:
    try:
        return subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=60, env=isolated_env()).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def call(system: str, user: str, model: str, effort: str, schema: dict | None = None, work_root: str | None = None,
         timeout: float = 1800.0, extra_flags: list | None = None, isolated: bool = True) -> dict:
    """Run one call; returns the answer (`structured` when a schema was given, else `text`), the
    usage the CLI reports and the wall time. `isolated=False` is only for the leak test's control."""
    root = Path(work_root or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    cwd = Path(tempfile.mkdtemp(prefix="trial_", dir=root))          # empty: nothing for the call to find
    sysdir = Path(tempfile.mkdtemp(prefix="sys_", dir=root))
    try:
        sys_path = sysdir / "system.txt"
        sys_path.write_text(system)
        argv = ["claude", "-p"] + (FLAGS if isolated else []) + ["--system-prompt-file", str(sys_path), "--model", model,
                                                                  "--effort", effort, "--output-format", "json"]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema)]
        argv += list(extra_flags or [])
        t0 = time.time()
        p = subprocess.run(argv, input=user, capture_output=True, text=True, cwd=str(cwd),
                           env=isolated_env() if isolated else dict(os.environ), timeout=timeout)
        wall = time.time() - t0
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
        shutil.rmtree(sysdir, ignore_errors=True)
    out = {"argv": [a if a != str(sys_path) else "<system.txt>" for a in argv], "wall_s": wall, "returncode": p.returncode,
           "system_sha256": hashlib.sha256(system.encode()).hexdigest()[:16], "stderr": p.stderr[-2000:]}
    try:
        d = json.loads(p.stdout)
    except ValueError:
        out.update({"ok": False, "error": "no JSON from the CLI", "stdout": p.stdout[-4000:]})
        return out
    u = d.get("usage") or {}
    out.update({"ok": not d.get("is_error", False), "text": d.get("result"), "structured": d.get("structured_output"),
                "num_turns": d.get("num_turns"), "duration_ms": d.get("duration_ms"), "duration_api_ms": d.get("duration_api_ms"),
                "cost_usd": d.get("total_cost_usd"), "stop_reason": d.get("stop_reason"), "subtype": d.get("subtype"),
                "model_usage": {m: {k: v.get(k) for k in ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "thinkingTokens")}
                                for m, v in (d.get("modelUsage") or {}).items()},
                "usage": {"input": u.get("input_tokens", 0), "cache_creation": u.get("cache_creation_input_tokens", 0),
                          "cache_read": u.get("cache_read_input_tokens", 0), "output": u.get("output_tokens", 0),
                          "thinking": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0)}})
    return out
