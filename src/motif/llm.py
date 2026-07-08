"""LLM calling layer.

Three verbs: complete(), extract(), act().
Msg in, text or structured data or action out. render() is implicit.

Observer support: call observe() to attach callbacks that receive
every LLM call's inputs and outputs. The pipeline stays pure.
"""

from __future__ import annotations

import asyncio
import os
import json
from dataclasses import dataclass, field
from typing import Callable, Any

from dotenv import load_dotenv
import anthropic
import httpx  # transitive dependency of the anthropic SDK — no new top-level dep

from .prompt import Msg, render
from .graph import current_node

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


# Pricing per million tokens (input, output, cache_read, cache_write).
# Cache reads are cheaper; cache writes cost extra on top of input.
# Updated for current Anthropic pricing as of early 2026.
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4-6":      (15.00, 75.00, 1.50, 18.75),
    "claude-opus-4-5":      (15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-4-6":    (3.00,  15.00, 0.30, 3.75),
    "claude-sonnet-4-5":    (3.00,  15.00, 0.30, 3.75),
    "claude-haiku-4-5":     (0.80,  4.00,  0.08, 1.00),
}


class CostTracker:
    """LLM observer that tracks token usage and cost.

        tracker = CostTracker()
        llm.observe(tracker)
        # ... run pipeline ...
        print(tracker)   # Cost: $0.42 (12,340 in / 3,210 out)
        tracker.cost     # 0.42
        tracker.reset()

    Attaches to llm.observe(), not flow.observe(). Pricing is
    looked up by model name; unknown models track tokens but not cost.
    """

    def __init__(self):
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.cache_creation_tokens: int = 0
        self.calls: int = 0
        self._cost: float = 0.0

    def __call__(self, verb: str, msg: Any, result: Any, model: str, meta: dict):
        inp = meta.get("input_tokens", 0)
        out = meta.get("output_tokens", 0)
        cache_read = meta.get("cache_read_tokens", 0)
        cache_create = meta.get("cache_creation_tokens", 0)

        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_create
        self.calls += 1

        # Look up pricing — strip date suffixes for matching
        base_model = model
        for name, prices in _PRICING.items():
            if model.startswith(name) or name.startswith(model):
                base_model = name
                break

        if base_model in _PRICING:
            p_in, p_out, p_cache_read, p_cache_write = _PRICING[base_model]
            # Cache reads replace regular input tokens in billing
            regular_input = inp - cache_read - cache_create
            self._cost += (
                regular_input * p_in / 1_000_000
                + out * p_out / 1_000_000
                + cache_read * p_cache_read / 1_000_000
                + cache_create * p_cache_write / 1_000_000
            )

    @property
    def cost(self) -> float:
        return round(self._cost, 4)

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.calls = 0
        self._cost = 0.0

    def __repr__(self):
        return (f"Cost: ${self.cost:.4f} "
                f"({self.input_tokens:,} in / {self.output_tokens:,} out / "
                f"{self.calls} calls)")


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
        _client = anthropic.AsyncAnthropic(
            max_retries=_max_retries,
            timeout=600.0,  # 10 minutes — prevents SDK forcing streaming
        )
    return _client


def _usage(response) -> dict:
    """Extract token usage from an API response for observer meta."""
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
    }


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CHEAP_MODEL = "claude-haiku-4-5"  # used by flow.py for structural decisions
DEFAULT_MAX_TOKENS = 32000

# Sentinel for "use whatever DEFAULT_MODEL is at call time"
_UNSET = object()


# --- Endpoints and roles ---
#
# Cost is a property of the run, not the program. Pipelines name what a
# call is FOR (a role); a profile binds roles to endpoints per deployment.
# Swapping Anthropic API / OpenRouter / a local model is a profile change,
# never a pipeline edit.

@dataclass(frozen=True, slots=True)
class Endpoint:
    """A model bound to the place it runs.

    base_url=None means the Anthropic SDK transport (ANTHROPIC_API_KEY).
    Any base_url means an OpenAI-compatible /chat/completions endpoint —
    OpenRouter, a local llama.cpp server, vLLM, etc.

        Endpoint("deepseek/deepseek-v4-flash",
                 base_url="https://openrouter.ai/api/v1",
                 key_env="OPENROUTER_API_KEY",
                 extra={"provider": {"order": ["DeepSeek"]},
                        "reasoning": {"enabled": False}})

        Endpoint("gemma-4-26b-a4b", base_url="http://localhost:11500/v1")

    extra is merged into the request body (openai transport only) —
    provider routing, reasoning toggles, whatever the endpoint speaks.
    """
    model: str
    base_url: str | None = None
    key_env: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoleRef:
    """A lazy reference to a role. Resolution happens at call time inside
    the verbs, so profiles can be bound (or swapped) any time before the
    call — including after modules with role-valued defaults are imported.
    """
    name: str

    def __repr__(self):
        return f"role({self.name!r})"


def role(name: str) -> RoleRef:
    """Name what a call is for; the current profile decides what runs it.

        brief = await llm.complete(prompt, model=llm.role("content"))
        vote  = await llm.extract(prompt, schema=S, model=llm.role("structure"))
    """
    return RoleRef(name)


_profile: dict[str, str | Endpoint] = {}


def use_profile(profile: dict[str, str | Endpoint]):
    """Bind role names to models/endpoints for this deployment.

        llm.use_profile({
            "structure": "claude-haiku-4-5",
            "content":   "claude-opus-4-5",
            "swarm":     Endpoint("deepseek/deepseek-v4-flash",
                                  base_url="https://openrouter.ai/api/v1",
                                  key_env="OPENROUTER_API_KEY"),
        })

    Unbound roles fall back: "content" → DEFAULT_MODEL,
    "structure" → DEFAULT_CHEAP_MODEL. Other unbound names raise at call time.
    """
    _profile.clear()
    _profile.update(profile)


def _endpoint(model: str | Endpoint | RoleRef) -> Endpoint:
    """Resolve role → binding → Endpoint. A bare model string is an
    Anthropic-transport endpoint."""
    if isinstance(model, RoleRef):
        if model.name in _profile:
            model = _profile[model.name]
        elif model.name == "content":
            model = DEFAULT_MODEL
        elif model.name == "structure":
            model = DEFAULT_CHEAP_MODEL
        else:
            raise KeyError(
                f"Role {model.name!r} is not bound — call "
                f"llm.use_profile({{{model.name!r}: ...}}) "
                f"(only 'content' and 'structure' have defaults)")
    if isinstance(model, Endpoint):
        return model
    return Endpoint(model=model)


# --- OpenAI-compatible transport (httpx) ---

_http: httpx.AsyncClient | None = None

_RETRY_STATUSES = {429, 500, 502, 503, 529}

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=600.0)
    return _http


def _openai_headers(ep: Endpoint) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(ep.key_env) if ep.key_env else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _openai_url(ep: Endpoint) -> str:
    return ep.base_url.rstrip("/") + "/chat/completions"


async def _openai_request(ep: Endpoint, body: dict) -> dict:
    """POST to an OpenAI-compatible endpoint with simple backoff retry."""
    body = {**body, **ep.extra}
    for attempt in range(_max_retries + 1):
        resp = await _get_http().post(
            _openai_url(ep), json=body, headers=_openai_headers(ep))
        if resp.status_code in _RETRY_STATUSES and attempt < _max_retries:
            await asyncio.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code >= 400:
            # Surface the provider's error body — status alone is useless
            # for debugging routing/parameter-support failures.
            raise httpx.HTTPStatusError(
                f"{resp.status_code} from {_openai_url(ep)}: {resp.text[:500]}",
                request=resp.request, response=resp)
        return resp.json()
    raise RuntimeError("unreachable")  # loop always returns or raises


def _usage_openai(data: dict) -> dict:
    u = data.get("usage") or {}
    meta = {
        "input_tokens": u.get("prompt_tokens", 0),
        "output_tokens": u.get("completion_tokens", 0),
        "cache_read_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "cache_creation_tokens": 0,
    }
    if "cost" in u:  # OpenRouter reports actual dollars — pass through
        meta["reported_cost"] = u["cost"]
    return meta


def _strip_fences(text: str) -> str:
    """Strip a markdown code fence if the whole payload is wrapped in one."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.index("\n") if "\n" in t else len(t)
        t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t


def _tools_to_openai(tools: list[dict]) -> list[dict]:
    """Anthropic tool schemas ({name, description, input_schema}) →
    OpenAI function-calling format."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object"}),
        },
    } for t in tools]


async def complete(
    msg: Msg,
    *,
    model: str | Endpoint | RoleRef = _UNSET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    streaming: bool = False,
    meta: dict | None = None,
) -> str:
    """Msg in, text out.

    model may be a model name (Anthropic transport) or an Endpoint
    (OpenAI-compatible transport). The transport decides the render format.

    streaming=True emits per-chunk notifications to observers (for live display)
    while still returning the complete text. Same result, richer observation.
    """
    ep = _endpoint(DEFAULT_MODEL if model is _UNSET else model)

    if streaming:
        # Use stream() internally, collect the result
        chunks = []
        async for chunk in stream(msg, model=ep, max_tokens=max_tokens,
                                   temperature=temperature, meta=meta):
            chunks.append(chunk)
        return "".join(chunks)

    if ep.base_url:
        payload = render(msg, backend="openai")
        body = {"model": ep.model, "max_tokens": max_tokens,
                "messages": payload["messages"]}
        if temperature is not None:
            body["temperature"] = temperature
        data = await _openai_request(ep, body)
        result = data["choices"][0]["message"].get("content") or ""
        _notify("complete", msg, result, ep.model, {
            **(meta or {}),
            **_usage_openai(data),
        })
        return result

    client = _get_client()
    payload = render(msg, backend="anthropic")

    kwargs = {
        "model": ep.model,
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

    _notify("complete", msg, result, ep.model, {
        **(meta or {}),
        **_usage(response),
    })
    return result


async def stream(
    msg: Msg,
    *,
    model: str | Endpoint | RoleRef = _UNSET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    meta: dict | None = None,
):
    """Msg in, async iterator of text chunks out.

    Same as complete() but yields tokens as they arrive.
    The full text is also notified to observers after the stream ends.

        async for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
    """
    ep = _endpoint(DEFAULT_MODEL if model is _UNSET else model)
    _meta = meta or {}
    node = current_node()  # graph node from flow context, if any
    full_text = []

    if ep.base_url:
        payload = render(msg, backend="openai")
        body = {"model": ep.model, "max_tokens": max_tokens,
                "messages": payload["messages"], "stream": True,
                "stream_options": {"include_usage": True},
                **ep.extra}
        if temperature is not None:
            body["temperature"] = temperature

        usage_meta = {}
        async with _get_http().stream(
            "POST", _openai_url(ep), json=body, headers=_openai_headers(ep),
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode(errors="replace")[:500]
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} from {ep.base_url}: {detail}",
                    request=resp.request, response=resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if chunk.get("usage"):
                    usage_meta = _usage_openai(chunk)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    full_text.append(text)
                    if node:
                        node.append_output(text)
                    _notify("chunk", msg, text, ep.model, _meta)
                    yield text

        result = "".join(full_text)
        _notify("stream", msg, result, ep.model, {**_meta, **usage_meta})
        return

    client = _get_client()
    payload = render(msg, backend="anthropic")

    kwargs = {
        "model": ep.model,
        "max_tokens": max_tokens,
        "messages": payload["messages"],
    }
    if "system" in payload:
        kwargs["system"] = payload["system"]
    if temperature is not None:
        kwargs["temperature"] = temperature

    async with client.messages.stream(**kwargs) as s:
        async for text in s.text_stream:
            full_text.append(text)
            if node:
                node.append_output(text)
            _notify("chunk", msg, text, ep.model, _meta)
            yield text
        response = await s.get_final_message()

    result = "".join(full_text)
    _notify("stream", msg, result, ep.model, {**_meta, **_usage(response)})


async def extract(
    msg: Msg,
    schema: dict,
    *,
    model: str | Endpoint | RoleRef = _UNSET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    meta: dict | None = None,
) -> dict:
    """Msg in, structured data out.

    Anthropic transport: output_config with json_schema (forced tool use
    on older SDK versions). OpenAI-compatible transport: response_format
    json_schema (supported by OpenRouter, llama.cpp, vLLM).
    """
    ep = _endpoint(DEFAULT_MODEL if model is _UNSET else model)

    if ep.base_url:
        payload = render(msg, backend="openai")
        body = {"model": ep.model, "max_tokens": max_tokens,
                "messages": payload["messages"]}
        if temperature is not None:
            body["temperature"] = temperature

        # Not every endpoint supports json_schema (OpenRouter calls it
        # "structured_outputs" and 404s a pinned provider without it).
        # Degrade invisibly: json_schema → json_object + schema in prompt
        # → bare prompt. The caller just gets their dict.
        instructed = payload["messages"] + [{
            "role": "user",
            "content": ("Respond with only a JSON object matching this "
                        "schema:\n" + json.dumps(schema)),
        }]
        attempts = [
            {**body, "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True,
                                "schema": schema}}},
            {**body, "messages": instructed,
             "response_format": {"type": "json_object"}},
            {**body, "messages": instructed},
        ]
        data = None
        for i, attempt in enumerate(attempts):
            try:
                data = await _openai_request(ep, attempt)
                break
            except httpx.HTTPStatusError as e:
                unsupported = e.response.status_code in (400, 404)
                if not unsupported or i == len(attempts) - 1:
                    raise
        content = data["choices"][0]["message"].get("content") or ""
        result = json.loads(_strip_fences(content))
        _notify("extract", msg, result, ep.model,
                {**(meta or {}), **_usage_openai(data)})
        return result

    model = ep.model
    client = _get_client()
    payload = render(msg, backend="anthropic")

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
                _notify("extract", msg, result, model, {**(meta or {}), **_usage(response)})
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
                _notify("extract", msg, result, model, {**(meta or {}), **_usage(response)})
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
    model: str | Endpoint | RoleRef = _UNSET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
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
    ep = _endpoint(DEFAULT_MODEL if model is _UNSET else model)

    if ep.base_url:
        payload = render(msg, backend="openai")
        body = {"model": ep.model, "max_tokens": max_tokens,
                "messages": payload["messages"],
                "tools": _tools_to_openai(tools)}
        if temperature is not None:
            body["temperature"] = temperature
        data = await _openai_request(ep, body)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        tool_calls = [
            ToolRequest(
                id=tc["id"],
                name=tc["function"]["name"],
                input=json.loads(tc["function"].get("arguments") or "{}"),
            )
            for tc in (message.get("tool_calls") or [])
        ]
        finish = choice.get("finish_reason")
        result = ActResult(
            text=message.get("content") or None,
            tool_calls=tool_calls,
            stop_reason=_FINISH_TO_STOP.get(finish, finish),
        )
        _notify("act", msg, result, ep.model,
                {**(meta or {}), **_usage_openai(data)})
        return result

    model = ep.model
    client = _get_client()
    payload = render(msg, backend="anthropic")

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

    _notify("act", msg, result, model, {**(meta or {}), **_usage(response)})
    return result
