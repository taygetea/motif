"""LLM calling layer.

Three verbs: complete(), extract(), act().
Msg in, text or structured data or action out. render() is implicit.

Observer support: call observe() to attach callbacks that receive
every LLM call's inputs and outputs. The pipeline stays pure.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Callable, Any

from dotenv import load_dotenv
import anthropic

from .prompt import Msg, render

# Load .env from the project root (or any parent). Does nothing if no .env exists.
load_dotenv()

# Detect SDK support for output_config (added in ~0.80) once at import time,
# not per-call via try/except TypeError which would mask unrelated TypeErrors.
_HAS_OUTPUT_CONFIG = hasattr(anthropic, "NOT_GIVEN")  # proxy for modern SDK
try:
    from packaging.version import Version
    _HAS_OUTPUT_CONFIG = Version(anthropic.__version__) >= Version("0.80")
except Exception:
    pass  # packaging not available — use the hasattr heuristic

_client: anthropic.AsyncAnthropic | None = None
_observers: list[Callable] = []


def observe(*observers: Callable):
    """Attach observer callbacks. Each receives (verb, msg, result, model, meta)."""
    _observers.extend(observers)


def clear_observers():
    """Remove all observers."""
    _observers.clear()


def _notify(verb: str, msg: Msg, result: Any, model: str, meta: dict):
    for obs in _observers:
        try:
            obs(verb, msg, result, model, meta)
        except Exception:
            pass  # observers should not break the pipeline


_max_retries = 3


def configure(*, max_retries: int | None = None, model: str | None = None):
    """Configure the LLM client. Call before first API use.

        llm.configure(model="claude-opus-4-6")     # upgrade default
        llm.configure(max_retries=5)
    """
    global _max_retries, _client, DEFAULT_MODEL
    if max_retries is not None:
        _max_retries = max_retries
        _client = None  # force re-creation with new settings
    if model is not None:
        DEFAULT_MODEL = model


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # anthropic SDK reads ANTHROPIC_API_KEY from env automatically.
        # max_retries handles 429, 529, and transient 500s with backoff.
        _client = anthropic.AsyncAnthropic(max_retries=_max_retries)
    return _client


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5"  # used by flow.py for structural decisions
DEFAULT_MAX_TOKENS = 16000


async def complete(
    msg: Msg,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    backend: str = "anthropic",
    meta: dict | None = None,
) -> str:
    """Msg in, text out."""
    client = _get_client()
    payload = render(msg, backend=backend)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": payload["messages"],
    }
    if "system" in payload:
        kwargs["system"] = payload["system"]
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = await client.messages.create(**kwargs)

    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    result = "\n".join(text_parts)

    _notify("complete", msg, result, model, meta or {})
    return result


async def extract(
    msg: Msg,
    schema: dict,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    backend: str = "anthropic",
    meta: dict | None = None,
) -> dict:
    """Msg in, structured data out.

    Uses output_config with json_schema for structured output.
    Falls back to forced tool use on older SDK versions.
    """
    client = _get_client()
    payload = render(msg, backend=backend)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": payload["messages"],
    }
    if "system" in payload:
        kwargs["system"] = payload["system"]
    if temperature is not None:
        kwargs["temperature"] = temperature

    if _HAS_OUTPUT_CONFIG:
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }
        response = await client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "text":
                result = json.loads(block.text)
                _notify("extract", msg, result, model, meta or {})
                return result
        raise ValueError("No text block in structured response")

    else:
        # Older SDK — forced tool use as structured output
        tool_name = "structured_output"
        kwargs["tools"] = [{
            "name": tool_name,
            "description": "Record your structured assessment.",
            "input_schema": schema,
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        response = await client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                result = block.input
                _notify("extract", msg, result, model, meta or {})
                return result
        raise ValueError("No tool_use block in structured response")


# --- act: the third verb ---

@dataclass(slots=True)
class ToolRequest:
    """A single tool call from the model."""
    id: str
    name: str
    input: dict


@dataclass(slots=True)
class ActResult:
    """What act() returned — either a final answer or tool calls.

    Check .done to see if the model finished, or .tool_calls for
    what it wants to invoke next. stop_reason is the raw API signal:
    "end_turn", "tool_use", or "max_tokens" (truncated).
    """
    text: str | None = None
    tool_calls: list[ToolRequest] = field(default_factory=list)
    stop_reason: str | None = None

    @property
    def done(self) -> bool:
        return not self.tool_calls


async def act(
    msg: Msg,
    tools: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    backend: str = "anthropic",
    meta: dict | None = None,
) -> ActResult:
    """Msg in, action out. The third verb.

    Sends the Msg with a tool list. The model either:
    - Returns final text (ActResult.done is True, .text has the answer)
    - Returns tool calls (ActResult.done is False, .tool_calls has them)

    The caller executes the tools, appends tool_use/tool_result segments
    to the Msg with |, and calls act() again. The agent loop is just
    this verb in a while loop.

        result = await act(prompt, tools=TOOL_SCHEMAS)
        if result.done:
            print(result.text)
        else:
            for call in result.tool_calls:
                output = execute(call.name, call.input)
                prompt = prompt | tool_use(call.id, call.name, call.input) \\
                                | tool_result(call.id, output)
    """
    client = _get_client()
    payload = render(msg, backend=backend)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": payload["messages"],
        "tools": tools,
    }
    if "system" in payload:
        kwargs["system"] = payload["system"]
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = await client.messages.create(**kwargs)

    # Collect text and tool calls from the response
    text_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolRequest(
                id=block.id,
                name=block.name,
                input=block.input,
            ))

    text = "\n".join(text_parts) if text_parts else None
    result = ActResult(
        text=text,
        tool_calls=tool_calls,
        stop_reason=getattr(response, "stop_reason", None),
    )

    _notify("act", msg, result, model, meta or {})
    return result
