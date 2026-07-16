"""LLM calling layer.

Three verbs: complete(), extract(), act().
Msg in, text or structured data or action out. render() is implicit.

Observation: every verb invocation emits call-lifecycle events —
CallStarted → CallChunk* → (CallCompleted | CallFailed) — through
observe_calls(). Each call has its own identity (call_id), so two
identical concurrent resamples are two distinct facts. The legacy
(verb, msg, result, model, meta) signature remains available through
observe(), derived from the same events. The pipeline stays pure;
this module knows nothing about the computation graph — record.py
projects these events into graph nodes from above.
"""

from __future__ import annotations

import asyncio
import os
import json
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Any

from dotenv import load_dotenv
import anthropic
import httpx  # transitive dependency of the anthropic SDK — no new top-level dep

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


# --- Call-lifecycle events: the observation seam ---
#
# Pure Layer-2 facts about verb invocations. One CallStarted per call,
# zero or more CallChunks (streaming), then exactly one of
# CallCompleted / CallFailed. call_id is the per-call identity —
# best_of's five judgments are five ids, not one shared node.
#
# Events are emitted inline, in the calling task, so an observer can
# read task-local context (the graph projection reads current_node()).

@dataclass(frozen=True, slots=True)
class CallStarted:
    """A verb invocation began. msg is the actual input Msg, retained —
    replay, lineage, and the loom read it from here."""
    call_id: str
    verb: str            # "complete" | "stream" | "extract" | "act"
    msg: Msg
    declared: Any        # the model param as passed: str | Endpoint | RoleRef
    endpoint: Endpoint   # the resolved endpoint that will serve the call
    params: dict         # max_tokens, temperature, schema, tools — the call's facts
    meta: dict           # author-supplied meta (fan passes {"node": <label>})


@dataclass(frozen=True, slots=True)
class CallChunk:
    """One streamed text chunk."""
    call_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CallCompleted:
    """The call returned. stop_reason uses the anthropic vocabulary on
    every transport ("length" arrives as "max_tokens"); a completed
    call with stop_reason "max_tokens" is a truncation — the verb may
    still raise Truncated after this event, so cost is always seen."""
    call_id: str
    result: Any          # str | dict | ActResult — what the verb returns
    usage: dict          # token counts; may carry reported_cost
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CallFailed:
    """The call raised. usage is non-empty when the transport billed
    before failing (e.g. a truncated extract that cannot parse)."""
    call_id: str
    error: str
    exception: BaseException | None = None
    usage: dict = field(default_factory=dict)


_call_observers: list[Callable] = []

# The graph projection slot — set by record.py at import, deliberately
# not an entry in the observer registry: clear_observers() must not be
# able to silence the graph, and exactly one projection may exist (a
# second would double-record every call).
_projection: Callable | None = None


def _new_call_id() -> str:
    return uuid.uuid4().hex[:12]


def _emit(event) -> None:
    if _projection is not None:
        try:
            _projection(event)
        except Exception:
            pass  # the record must not break the pipeline
    for obs in _call_observers:
        try:
            obs(event)
        except Exception:
            pass  # observers must not break the pipeline


def observe_calls(*observers: Callable):
    """Attach observers to the call-lifecycle seam. Each receives
    CallStarted / CallChunk / CallCompleted / CallFailed events."""
    _call_observers.extend(observers)


class _LegacyAdapter:
    """Presents the legacy (verb, msg, result, model, meta) observer
    signature on top of call-lifecycle events. Stateful: correlates by
    call_id. Chunks become ("chunk", ...) notifications; failures were
    never notified in the legacy protocol and still aren't."""

    __slots__ = ("fn", "_started")

    def __init__(self, fn: Callable):
        self.fn = fn
        self._started: dict[str, CallStarted] = {}

    def __call__(self, event):
        if isinstance(event, CallStarted):
            self._started[event.call_id] = event
        elif isinstance(event, CallChunk):
            started = self._started.get(event.call_id)
            if started is not None:
                self.fn("chunk", started.msg, event.text,
                        started.endpoint.model, dict(started.meta))
        elif isinstance(event, CallCompleted):
            started = self._started.pop(event.call_id, None)
            if started is not None:
                self.fn(started.verb, started.msg, event.result,
                        started.endpoint.model,
                        {**started.meta, **event.usage})
        elif isinstance(event, CallFailed):
            self._started.pop(event.call_id, None)


def observe(*observers: Callable):
    """Attach legacy observer callbacks. Each receives
    (verb, msg, result, model, meta) — derived from the call-lifecycle
    events by an adapter. New code should prefer observe_calls()."""
    _call_observers.extend(_LegacyAdapter(o) for o in observers)


def clear_observers():
    """Remove all observers (both signatures). The graph projection is
    not an observer and survives."""
    _call_observers.clear()


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
        if verb == "chunk":
            return  # per-chunk notifications are not calls

        inp = meta.get("input_tokens", 0)
        out = meta.get("output_tokens", 0)
        cache_read = meta.get("cache_read_tokens", 0)
        cache_create = meta.get("cache_creation_tokens", 0)

        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_create
        self.calls += 1

        # Providers that report actual billed cost (OpenRouter) beat any
        # table lookup — use the real number and skip estimation.
        if "reported_cost" in meta:
            self._cost += meta["reported_cost"] or 0.0
            return

        # Pricing lookup: exact id or a suffixed variant of a table entry
        # ("claude-haiku-4-5-20260101"). Bare prefix matching in both
        # directions billed lookalike ids against the wrong row.
        base_model = model
        for name in _PRICING:
            if model == name or model.startswith(name + "-"):
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


_profile: ContextVar[dict[str, str | Endpoint] | None] = ContextVar(
    "motif_profile", default=None)


def use_profile(profile: dict[str, str | Endpoint]):
    """Bind role names to models/endpoints for this run.

        llm.use_profile({
            "structure": "claude-haiku-4-5",
            "content":   "claude-opus-4-5",
            "swarm":     Endpoint("deepseek/deepseek-v4-flash",
                                  base_url="https://openrouter.ai/api/v1",
                                  key_env="OPENROUTER_API_KEY"),
        })

    Unbound roles fall back: "content" → DEFAULT_MODEL,
    "structure" → DEFAULT_CHEAP_MODEL. Other unbound names raise at call time.

    The binding is a ContextVar, not a process global: it applies to the
    current async context and any tasks created after the call. Concurrent
    runs that bind different profiles cannot switch each other's models.
    """
    _profile.set(dict(profile))


def _endpoint(model: str | Endpoint | RoleRef) -> Endpoint:
    """Resolve role → binding → Endpoint. A bare model string is an
    Anthropic-transport endpoint."""
    if isinstance(model, RoleRef):
        bound = _profile.get() or {}
        if model.name in bound:
            model = bound[model.name]
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


class Truncated(RuntimeError):
    """The response hit max_tokens and is only partial.

    Raised by complete() and stream() so truncation cannot masquerade as
    a successful result (allow_truncation=True opts back in to partial
    text), and by extract() unconditionally — a JSON document cut mid-
    stream is never a valid partial, so there is nothing to opt in to.
    The raw partial text is available as `.partial`.
    """

    def __init__(self, message: str, partial: str = ""):
        super().__init__(message)
        self.partial = partial


def _truncated(max_tokens: int, partial: str) -> Truncated:
    return Truncated(
        f"Response hit max_tokens={max_tokens} — this is a partial answer, "
        f"not a complete one. Raise max_tokens, or pass allow_truncation=True "
        f"to accept partial output (available on the exception as .partial).",
        partial)


def _extract_truncated(max_tokens: int, partial: str) -> Truncated:
    return Truncated(
        f"extract() response hit max_tokens={max_tokens} — the structured "
        f"output was cut mid-JSON and cannot be parsed. Raise max_tokens or "
        f"ask for less. The raw partial text is on the exception as .partial.",
        partial)


async def complete(
    msg: Msg,
    *,
    model: str | Endpoint | RoleRef = _UNSET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    streaming: bool = False,
    allow_truncation: bool = False,
    meta: dict | None = None,
) -> str:
    """Msg in, text out.

    model may be a model name (Anthropic transport) or an Endpoint
    (OpenAI-compatible transport). The transport decides the render format.

    streaming=True emits per-chunk notifications to observers (for live display)
    while still returning the complete text. Same result, richer observation.

    A response cut off by max_tokens raises Truncated (observers are still
    notified first, so cost is tracked). Pass allow_truncation=True to get
    the partial text back as an ordinary return instead.
    """
    declared = DEFAULT_MODEL if model is _UNSET else model
    ep = _endpoint(declared)

    if streaming:
        # Delegate to stream(), collect the result. The lifecycle is
        # stream()'s (verb "stream") — no double emission from here.
        chunks = []
        async for chunk in stream(msg, model=declared, max_tokens=max_tokens,
                                   temperature=temperature,
                                   allow_truncation=allow_truncation, meta=meta):
            chunks.append(chunk)
        return "".join(chunks)

    call_id = _new_call_id()
    _emit(CallStarted(call_id, "complete", msg, declared, ep,
                      {"max_tokens": max_tokens, "temperature": temperature},
                      dict(meta or {})))
    try:
        if ep.base_url:
            payload = render(msg, backend="openai")
            body = {"model": ep.model, "max_tokens": max_tokens,
                    "messages": payload["messages"]}
            if temperature is not None:
                body["temperature"] = temperature
            data = await _openai_request(ep, body)
            choice = data["choices"][0]
            result = choice["message"].get("content") or ""
            usage = _usage_openai(data)
            finish = choice.get("finish_reason")
            stop_reason = _FINISH_TO_STOP.get(finish, finish)
        else:
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

            result = "\n".join(block.text for block in response.content
                               if block.type == "text")
            usage = _usage(response)
            stop_reason = getattr(response, "stop_reason", None)
    except Exception as e:
        _emit(CallFailed(call_id, str(e), e))
        raise

    _emit(CallCompleted(call_id, result, usage, stop_reason))
    if stop_reason == "max_tokens" and not allow_truncation:
        raise _truncated(max_tokens, result)
    return result


async def stream(
    msg: Msg,
    *,
    model: str | Endpoint | RoleRef = _UNSET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
    allow_truncation: bool = False,
    meta: dict | None = None,
):
    """Msg in, async iterator of text chunks out.

    Same as complete() but yields tokens as they arrive.
    The full text is also notified to observers after the stream ends.
    A stream cut off by max_tokens raises Truncated after the last chunk
    (all chunks were already yielded; observers were notified) unless
    allow_truncation=True.

        async for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
    """
    declared = DEFAULT_MODEL if model is _UNSET else model
    ep = _endpoint(declared)
    _meta = dict(meta or {})
    full_text = []

    call_id = _new_call_id()
    _emit(CallStarted(call_id, "stream", msg, declared, ep,
                      {"max_tokens": max_tokens, "temperature": temperature},
                      _meta))
    settled = False  # CallCompleted or CallFailed has been emitted

    try:
        if ep.base_url:
            payload = render(msg, backend="openai")
            body = {"model": ep.model, "max_tokens": max_tokens,
                    "messages": payload["messages"], "stream": True,
                    "stream_options": {"include_usage": True},
                    **ep.extra}
            if temperature is not None:
                body["temperature"] = temperature

            usage_meta = {}
            finish_reason = None
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
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        full_text.append(text)
                        _emit(CallChunk(call_id, text))
                        yield text

            result = "".join(full_text)
            _emit(CallCompleted(call_id, result, usage_meta,
                                _FINISH_TO_STOP.get(finish_reason, finish_reason)))
            settled = True
            if finish_reason == "length" and not allow_truncation:
                raise _truncated(max_tokens, result)
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
                _emit(CallChunk(call_id, text))
                yield text
            response = await s.get_final_message()

        result = "".join(full_text)
        stop_reason = getattr(response, "stop_reason", None)
        _emit(CallCompleted(call_id, result, _usage(response), stop_reason))
        settled = True
        if stop_reason == "max_tokens" and not allow_truncation:
            raise _truncated(max_tokens, result)
    except GeneratorExit:
        # The consumer abandoned the stream mid-flight (break / close).
        # Every started call settles exactly once — record the truth.
        if not settled:
            _emit(CallFailed(call_id, "stream abandoned before completion"))
        raise
    except Exception as e:
        if not settled:  # Truncated raises after CallCompleted — don't double-emit
            _emit(CallFailed(call_id, str(e), e))
        raise


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
    declared = DEFAULT_MODEL if model is _UNSET else model
    ep = _endpoint(declared)

    call_id = _new_call_id()
    _emit(CallStarted(call_id, "extract", msg, declared, ep,
                      {"max_tokens": max_tokens, "temperature": temperature,
                       "schema": schema},
                      dict(meta or {})))
    settled = False  # truncation emits its own CallFailed, with usage
    try:
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
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if choice.get("finish_reason") == "length":
                # Cut mid-JSON: fail legibly before json.loads turns this
                # into an opaque JSONDecodeError. The transport billed us —
                # the failure event carries the usage.
                exc = _extract_truncated(max_tokens, content)
                _emit(CallFailed(call_id, str(exc), exc,
                                 usage=_usage_openai(data)))
                settled = True
                raise exc
            result = json.loads(_strip_fences(content))
            usage = _usage_openai(data)
        else:
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

            if _HAS_OUTPUT_CONFIG:
                kwargs["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
                response = await client.messages.create(**kwargs)
                if getattr(response, "stop_reason", None) == "max_tokens":
                    partial = "\n".join(b.text for b in response.content
                                        if b.type == "text")
                    exc = _extract_truncated(max_tokens, partial)
                    _emit(CallFailed(call_id, str(exc), exc,
                                     usage=_usage(response)))
                    settled = True
                    raise exc
                result = None
                for block in response.content:
                    if block.type == "text":
                        result = json.loads(block.text)
                        break
                if result is None:
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
                if getattr(response, "stop_reason", None) == "max_tokens":
                    partial = str(next(
                        (b.input for b in response.content
                         if b.type == "tool_use"), ""))
                    exc = _extract_truncated(max_tokens, partial)
                    _emit(CallFailed(call_id, str(exc), exc,
                                     usage=_usage(response)))
                    settled = True
                    raise exc
                result = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == tool_name:
                        result = block.input
                        break
                if result is None:
                    raise ValueError("No tool_use block in structured response")
            usage = _usage(response)
    except Exception as e:
        if not settled:
            _emit(CallFailed(call_id, str(e), e))
        raise

    _emit(CallCompleted(call_id, result, usage))
    return result


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
    declared = DEFAULT_MODEL if model is _UNSET else model
    ep = _endpoint(declared)

    call_id = _new_call_id()
    _emit(CallStarted(call_id, "act", msg, declared, ep,
                      {"max_tokens": max_tokens, "temperature": temperature,
                       "tools": tools},
                      dict(meta or {})))
    try:
        if ep.base_url:
            payload = render(msg, backend="openai")
            body = {"model": ep.model, "max_tokens": max_tokens,
                    "messages": payload["messages"]}
            if tools:  # some providers 400 on an empty tools array
                body["tools"] = _tools_to_openai(tools)
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
            usage = _usage_openai(data)
        else:
            client = _get_client()
            payload = render(msg, backend="anthropic")

            kwargs = {
                "model": ep.model,
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

            result = ActResult(
                text="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls,
                stop_reason=getattr(response, "stop_reason", None),
            )
            usage = _usage(response)
    except Exception as e:
        _emit(CallFailed(call_id, str(e), e))
        raise

    # act() never raises on truncation: stop_reason "max_tokens" is a
    # representable result the agent loop handles (see flow.agent).
    _emit(CallCompleted(call_id, result, usage, result.stop_reason))
    return result
