"""CLI-harness adapters for delegated endpoints.

A delegated call hands one prompt to an agentic CLI harness (codex
exec) running hermetically — ephemeral, user config ignored, app/MCP
connectors stripped, local sandbox read-only — and returns the final
answer plus the full JSONL transcript. From motif's perspective the
interior is thinking: recording granularity follows effect radius, and
the hermetic invocation's residual ceiling is external-read plus
content generation (attested by the 2026-07-16 capability canary; see
notes/2026-07-16-canary-results.md).

This module is a leaf: stdlib only, no motif imports. llm.py calls
run_codex() and converts the AdapterResult into lifecycle events.
Treat the JSONL event vocabulary as an adapter protocol, not a stable
one — the CLI version is recorded in every result's fingerprint.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from dataclasses import dataclass, field

# Patchable in tests: point at a fake harness script. The real binary
# is resolved from PATH once, lazily.
CODEX_BIN: list[str] = ["codex"]

# The hermetic invocation. Every element is load-bearing (canary,
# 2026-07-16): --ephemeral + --ignore-user-config for statelessness;
# -s read-only governs LOCAL shell; mcp_servers={} strips the ChatGPT
# app connectors, WITHOUT WHICH THE HARNESS HAS LIVE REMOTE-WRITE
# TOOLS (GitHub issue/PR mutation) regardless of the sandbox.
CODEX_HERMETIC_ARGS: tuple[str, ...] = (
    "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
    "-s", "read-only", "--json", "-c", "mcp_servers={}",
)

_cli_version: str | None = None


@dataclass(slots=True)
class AdapterResult:
    """What one harness invocation produced — the raw facts, untyped
    by motif semantics. llm.py decides what they mean."""
    text: str | None            # final agent message, None if none arrived
    usage: dict                 # normalized token usage (may be empty)
    transcript: str             # raw JSONL, complete
    stderr: str
    exit_code: int | None       # None if killed on deadline
    failure: str | None         # normalized failure kind, None on success
    fingerprint: dict = field(default_factory=dict)


async def _cli_version_of(binary: list[str]) -> str:
    global _cli_version
    if _cli_version is None or binary != CODEX_BIN:
        try:
            proc = await asyncio.create_subprocess_exec(
                *binary, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            version = out.decode(errors="replace").strip()
        except Exception:
            version = "unknown"
        if binary == CODEX_BIN:
            _cli_version = version
        return version
    return _cli_version


def _normalize_usage(events: list[dict]) -> dict:
    """Token usage from the LAST turn.completed — counts are
    cumulative snapshots, never summed across events."""
    usage: dict = {}
    for event in events:
        if event.get("type") == "turn.completed" and event.get("usage"):
            u = event["usage"]
            usage = {
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": (u.get("output_tokens", 0)
                                  + u.get("reasoning_output_tokens", 0)),
                "cache_read_tokens": u.get("cached_input_tokens", 0),
                "cache_creation_tokens": 0,
                # Subscription lane: zero marginal dollars is the true
                # billed cost; the tokens still count as resource use.
                "reported_cost": 0.0,
                "billing_basis": "subscription",
                "usage_source": "codex_jsonl",
            }
    return usage


def strictify_schema(schema: dict) -> dict:
    """OpenAI strict structured output rejects any object schema
    without "additionalProperties": false — supply it recursively.
    Semantically a no-op for extraction (extra properties were never
    wanted); schemas strict mode still rejects fall through to the
    schema-in-prompt rung of the ladder."""
    def walk(node):
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            if out.get("type") == "object" or "properties" in out:
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node
    return walk(schema)


def _classify_failure(exit_code: int | None, stderr: str,
                      text: str | None, events: list[dict]) -> str | None:
    if exit_code is None:
        return "deadline_exceeded"
    turn_errors = " ".join(
        str(e.get("error", e.get("message", "")))
        for e in events if e.get("type") in ("turn.failed", "error"))
    if "invalid_json_schema" in turn_errors:
        return "schema_rejected"
    low = (stderr + " " + turn_errors).lower()
    if exit_code != 0 or turn_errors:
        if "login" in low or "auth" in low or "401" in low:
            return "auth_failed"
        if "rate limit" in low or "429" in low or "quota" in low:
            return "rate_limited" if "rate" in low else "quota_exhausted"
        return "process_failed"
    if text is None:
        return "no_answer"
    return None


async def run_codex(
    prompt: str,
    *,
    model: str = "",
    effort: str | None = None,
    search: bool = False,
    schema: dict | None = None,
    timeout: float = 600.0,
    max_output_bytes: int = 4_000_000,
    extra_args: tuple[str, ...] = (),
) -> AdapterResult:
    """One hermetic codex invocation. The prompt goes through stdin
    (argv "-"): a prompt in argv with stdin left open makes the CLI
    block on "reading additional input" (canary finding).

    Runs in a fresh empty temp cwd, in its own process group; on
    deadline the whole group is killed. The controller (this process)
    keeps its normal filesystem access — only the harness's tools are
    sandboxed.
    """
    argv = list(CODEX_BIN) + list(CODEX_HERMETIC_ARGS)
    if model:
        argv += ["-m", model]
    if effort:
        argv += ["-c", f"model_reasoning_effort={effort}"]
    if search:
        argv += ["-c", "tools.web_search=true"]
    argv += list(extra_args)

    tmpdir = tempfile.mkdtemp(prefix="motif-delegated-")
    schema_path = None
    try:
        if schema is not None:
            schema_path = os.path.join(tmpdir, "output-schema.json")
            with open(schema_path, "w") as f:
                json.dump(schema, f)
            argv += ["--output-schema", schema_path]

        argv.append("-")  # prompt arrives on stdin

        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=tmpdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group — killable as a tree
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=timeout)
            exit_code: int | None = proc.returncode
        except (TimeoutError, asyncio.TimeoutError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            stdout, stderr = await proc.communicate()  # reap
            exit_code = None

        transcript = stdout.decode(errors="replace")
        if len(transcript) > max_output_bytes:
            transcript = transcript[:max_output_bytes] + "\n[truncated]"

        events: list[dict] = []
        for line in transcript.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # non-JSON diagnostics interleave; keep going

        text = None
        for event in events:
            if (event.get("type") == "item.completed"
                    and event.get("item", {}).get("type") == "agent_message"):
                text = event["item"].get("text")

        stderr_text = stderr.decode(errors="replace")
        return AdapterResult(
            text=text,
            usage=_normalize_usage(events),
            transcript=transcript,
            stderr=stderr_text[:20_000],
            exit_code=exit_code,
            failure=_classify_failure(exit_code, stderr_text, text, events),
            fingerprint={
                "adapter": "codex",
                "cli_version": await _cli_version_of(list(CODEX_BIN)),
                "argv": [a for a in argv[:-1]],  # everything but the prompt marker
                "hermetic": True,
                "cwd": "ephemeral-tmpdir",
            },
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
