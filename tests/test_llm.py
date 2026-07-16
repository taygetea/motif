"""Tests for llm.py — endpoints, roles, and the OpenAI-compatible transport.

The openai transport is tested at the HTTP boundary via httpx.MockTransport,
so the real request-building and response-parsing code runs. The anthropic
path is tested against a fake SDK client at the same altitude.
"""

import json
import os

import httpx
import pytest

import motif.llm as llm
from motif.llm import Endpoint, RoleRef, role, use_profile, _endpoint
from motif import system, user


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate profile and transport state per test."""
    token = llm._profile.set(None)
    llm.clear_observers()
    yield
    llm._profile.reset(token)
    llm.clear_observers()
    llm._http = None


def _mock_http(handler):
    """Install an httpx client whose transport is the given handler."""
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chat_response(content="hello", tool_calls=None, finish="stop",
                   usage=None, model="test-model"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-1",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


EP = Endpoint("test-model", base_url="https://example.test/v1",
              key_env="TEST_LLM_KEY", extra={"provider": {"order": ["X"]}})


# --- Endpoint and role resolution ---

class TestResolution:
    def test_bare_string_is_anthropic_endpoint(self):
        ep = _endpoint("claude-opus-4-5")
        assert ep.model == "claude-opus-4-5"
        assert ep.base_url is None

    def test_endpoint_passes_through(self):
        assert _endpoint(EP) is EP

    def test_role_default_content(self):
        assert _endpoint(role("content")).model == llm.DEFAULT_MODEL

    def test_role_default_structure(self):
        assert _endpoint(role("structure")).model == llm.DEFAULT_CHEAP_MODEL

    def test_role_unbound_raises(self):
        with pytest.raises(KeyError):
            _endpoint(role("swarm"))

    def test_profile_binds_role_to_endpoint(self):
        use_profile({"swarm": EP})
        assert _endpoint(role("swarm")) is EP

    def test_profile_binds_role_to_string(self):
        use_profile({"content": "claude-haiku-4-5"})
        assert _endpoint(role("content")).model == "claude-haiku-4-5"

    def test_roleref_is_lazy(self):
        """A RoleRef created before the profile resolves against the
        profile current at call time — that's the whole point."""
        ref = role("content")
        use_profile({"content": "late-bound-model"})
        assert _endpoint(ref).model == "late-bound-model"

    def test_roleref_repr(self):
        assert repr(role("swarm")) == "role('swarm')"


# --- complete() over the openai transport ---

class TestOpenAIComplete:
    @pytest.mark.asyncio
    async def test_returns_content(self, monkeypatch):
        monkeypatch.setenv("TEST_LLM_KEY", "sk-test-123")
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response("the answer"))

        _mock_http(handler)
        result = await llm.complete(system("sys") | user("q"), model=EP)

        assert result == "the answer"
        assert seen["url"] == "https://example.test/v1/chat/completions"
        assert seen["auth"] == "Bearer sk-test-123"
        assert seen["body"]["model"] == "test-model"
        # extra merged into body (provider routing etc.)
        assert seen["body"]["provider"] == {"order": ["X"]}
        # openai render: system is a message, not a top-level key
        assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}

    @pytest.mark.asyncio
    async def test_none_content_returns_empty(self):
        """Reasoning models can return content: None — never crash."""
        _mock_http(lambda r: httpx.Response(200, json=_chat_response(None)))
        result = await llm.complete(user("q"), model=EP)
        assert result == ""

    @pytest.mark.asyncio
    async def test_usage_reaches_observers(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response(
            usage={"prompt_tokens": 42, "completion_tokens": 7,
                   "cost": 1.5e-06,
                   "prompt_tokens_details": {"cached_tokens": 12}})))
        seen = {}
        llm.observe(lambda verb, msg, res, model, meta: seen.update(meta, verb=verb))
        await llm.complete(user("q"), model=EP)
        assert seen["input_tokens"] == 42
        assert seen["output_tokens"] == 7
        assert seen["cache_read_tokens"] == 12
        assert seen["reported_cost"] == 1.5e-06
        assert seen["verb"] == "complete"

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_chat_response("ok"))

        _mock_http(handler)
        result = await llm.complete(user("q"), model=EP)
        assert result == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_4xx_raises(self):
        _mock_http(lambda r: httpx.Response(400, json={"error": "bad"}))
        with pytest.raises(httpx.HTTPStatusError):
            await llm.complete(user("q"), model=EP)

    @pytest.mark.asyncio
    async def test_truncation_raises(self):
        """finish_reason='length' must not masquerade as a complete answer."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("partial ans", finish="length")))
        with pytest.raises(llm.Truncated) as exc:
            await llm.complete(user("q"), model=EP, max_tokens=8)
        assert exc.value.partial == "partial ans"
        assert "max_tokens=8" in str(exc.value)
        assert "allow_truncation" in str(exc.value)  # the fix is in the message

    @pytest.mark.asyncio
    async def test_truncation_allowed_returns_partial(self):
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("partial ans", finish="length")))
        result = await llm.complete(user("q"), model=EP, allow_truncation=True)
        assert result == "partial ans"

    @pytest.mark.asyncio
    async def test_truncation_still_notifies_observers(self):
        """Cost was incurred — observers must see the call before the raise."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("partial", finish="length")))
        seen = {}
        llm.observe(lambda verb, msg, res, model, meta: seen.update(meta, verb=verb))
        with pytest.raises(llm.Truncated):
            await llm.complete(user("q"), model=EP)
        assert seen["verb"] == "complete"
        assert seen["output_tokens"] == 5


class TestProfileIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_tasks_keep_their_own_profiles(self):
        """use_profile in one task must not switch another task's models."""
        import asyncio

        async def run(model_name):
            use_profile({"content": model_name})
            await asyncio.sleep(0)  # yield so the tasks interleave
            return _endpoint(role("content")).model

        a, b = await asyncio.gather(run("model-a"), run("model-b"))
        assert (a, b) == ("model-a", "model-b")


# --- extract() over the openai transport ---

class TestOpenAIExtract:
    SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}}

    @pytest.mark.asyncio
    async def test_parses_json_content(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response('{"x": 3}'))

        _mock_http(handler)
        result = await llm.extract(user("q"), self.SCHEMA, model=EP)
        assert result == {"x": 3}
        rf = seen["body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == self.SCHEMA

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        """Some models fence their JSON despite response_format."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response('```json\n{"x": 9}\n```')))
        result = await llm.extract(user("q"), self.SCHEMA, model=EP)
        assert result == {"x": 9}

    @pytest.mark.asyncio
    async def test_degrades_json_schema_to_json_object(self):
        """Endpoints without structured_outputs (e.g. pinned DeepSeek on
        OpenRouter → 404) get json_object + schema-in-prompt, invisibly."""
        bodies = []

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            rf = body.get("response_format", {})
            if rf.get("type") == "json_schema":
                return httpx.Response(404, json={"error": {"message": "No endpoints found"}})
            return httpx.Response(200, json=_chat_response('{"x": 5}'))

        _mock_http(handler)
        result = await llm.extract(user("q"), self.SCHEMA, model=EP)
        assert result == {"x": 5}
        # second attempt used json_object and put the schema in the prompt
        assert bodies[1]["response_format"] == {"type": "json_object"}
        assert "schema" in bodies[1]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_non_parameter_errors_still_raise(self):
        """Only 400/404 trigger degradation — auth errors surface."""
        _mock_http(lambda r: httpx.Response(401, json={"error": "bad key"}))
        with pytest.raises(httpx.HTTPStatusError):
            await llm.extract(user("q"), self.SCHEMA, model=EP)

    @pytest.mark.asyncio
    async def test_truncation_raises_before_parse(self):
        """finish_reason='length' on structured output raises Truncated
        with the raw partial — not an opaque JSONDecodeError."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response('{"x": 1, "y"', finish="length")))
        with pytest.raises(llm.Truncated) as exc:
            await llm.extract(user("q"), self.SCHEMA, model=EP, max_tokens=16)
        assert exc.value.partial == '{"x": 1, "y"'
        assert "max_tokens=16" in str(exc.value)


# --- act() over the openai transport ---

class TestOpenAIAct:
    TOOLS = [{"name": "search", "description": "Search the web",
              "input_schema": {"type": "object",
                               "properties": {"q": {"type": "string"}}}}]

    @pytest.mark.asyncio
    async def test_tool_calls_parsed(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response(
                None, finish="tool_calls",
                tool_calls=[{"id": "tc1", "type": "function",
                             "function": {"name": "search",
                                          "arguments": '{"q": "motif"}'}}]))

        _mock_http(handler)
        result = await llm.act(user("find it"), self.TOOLS, model=EP)

        assert not result.done
        assert result.tool_calls[0].id == "tc1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].input == {"q": "motif"}  # real dict, not a string
        assert result.stop_reason == "tool_use"  # mapped to anthropic vocabulary
        # tools converted to openai function format on the wire
        assert seen["body"]["tools"][0]["type"] == "function"
        assert seen["body"]["tools"][0]["function"]["name"] == "search"
        assert seen["body"]["tools"][0]["function"]["parameters"] == \
            self.TOOLS[0]["input_schema"]

    @pytest.mark.asyncio
    async def test_final_text(self):
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("done", finish="stop")))
        result = await llm.act(user("q"), self.TOOLS, model=EP)
        assert result.done
        assert result.text == "done"
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_empty_tools_omitted_from_body(self):
        """Some providers 400 on an empty tools array — omit the key."""
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response("ok"))

        _mock_http(handler)
        await llm.act(user("q"), [], model=EP)
        assert "tools" not in seen["body"]

    @pytest.mark.asyncio
    async def test_length_maps_to_max_tokens(self):
        """agent() checks stop_reason == 'max_tokens' — mapping must hold."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("partial", finish="length")))
        result = await llm.act(user("q"), self.TOOLS, model=EP)
        assert result.stop_reason == "max_tokens"


# --- stream() over the openai transport (SSE) ---

class TestOpenAIStream:
    @pytest.mark.asyncio
    async def test_streams_chunks_and_usage(self):
        sse = (
            'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
            'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}\n\n'
            'data: [DONE]\n\n'
        )

        def handler(request):
            body = json.loads(request.content)
            assert body["stream"] is True
            return httpx.Response(200, content=sse.encode(),
                                  headers={"content-type": "text/event-stream"})

        _mock_http(handler)
        events = []
        llm.observe(lambda verb, msg, res, model, meta:
                    events.append((verb, res, meta)))

        chunks = [c async for c in llm.stream(user("q"), model=EP)]
        assert chunks == ["hel", "lo"]

        final = [e for e in events if e[0] == "stream"]
        assert final[0][1] == "hello"
        assert final[0][2]["input_tokens"] == 3

    @pytest.mark.asyncio
    async def test_complete_streaming_true_uses_sse(self):
        sse = ('data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
               'data: [DONE]\n\n')
        _mock_http(lambda r: httpx.Response(200, content=sse.encode()))
        result = await llm.complete(user("q"), model=EP, streaming=True)
        assert result == "ok"


# --- anthropic path still parses correctly (first direct coverage) ---

class _FakeBlock:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeUsage:
    input_tokens = 11
    output_tokens = 4
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _FakeResponse:
    def __init__(self, blocks):
        self.content = blocks
        self.usage = _FakeUsage()
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


class TestAnthropicPath:
    @pytest.mark.asyncio
    async def test_complete_joins_text_blocks(self, monkeypatch):
        fake = _FakeClient(_FakeResponse([
            _FakeBlock("text", text="part one"),
            _FakeBlock("text", text="part two"),
        ]))
        monkeypatch.setattr(llm, "_client", fake)
        result = await llm.complete(system("s") | user("q"), model="claude-haiku-4-5")
        assert result == "part one\npart two"
        # anthropic render: system is a top-level key
        assert fake.messages.last_kwargs["system"][0]["text"] == "s"
        assert fake.messages.last_kwargs["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_act_parses_tool_use_blocks(self, monkeypatch):
        fake = _FakeClient(_FakeResponse([
            _FakeBlock("tool_use", id="t1", name="search", input={"q": "x"}),
        ]))
        monkeypatch.setattr(llm, "_client", fake)
        result = await llm.act(user("q"), tools=[{"name": "search",
                                                  "input_schema": {}}])
        assert not result.done
        assert result.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_extract_truncation_raises_output_config_path(self, monkeypatch):
        """A structured response cut at max_tokens is never parseable —
        Truncated, with the raw text as .partial."""
        response = _FakeResponse([_FakeBlock("text", text='{"partial')])
        response.stop_reason = "max_tokens"
        monkeypatch.setattr(llm, "_client", _FakeClient(response))
        monkeypatch.setattr(llm, "_HAS_OUTPUT_CONFIG", True)
        with pytest.raises(llm.Truncated) as exc:
            await llm.extract(user("q"), {"type": "object"},
                              model="claude-haiku-4-5", max_tokens=8)
        assert exc.value.partial == '{"partial'

    @pytest.mark.asyncio
    async def test_extract_truncation_raises_forced_tool_path(self, monkeypatch):
        response = _FakeResponse([
            _FakeBlock("tool_use", id="t1", name="structured_output",
                       input={"half": "done"}),
        ])
        response.stop_reason = "max_tokens"
        monkeypatch.setattr(llm, "_client", _FakeClient(response))
        monkeypatch.setattr(llm, "_HAS_OUTPUT_CONFIG", False)
        with pytest.raises(llm.Truncated) as exc:
            await llm.extract(user("q"), {"type": "object"},
                              model="claude-haiku-4-5", max_tokens=8)
        assert "half" in exc.value.partial


# --- CostTracker ---

class TestCostTracker:
    def test_known_model_cost(self):
        t = llm.CostTracker()
        t("complete", None, None, "claude-haiku-4-5",
          {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        # haiku: $0.80/M in + $4.00/M out
        assert t.cost == pytest.approx(4.80)

    def test_date_suffix_matches(self):
        t = llm.CostTracker()
        t("complete", None, None, "claude-haiku-4-5-20260101",
          {"input_tokens": 1_000_000, "output_tokens": 0})
        assert t.cost == pytest.approx(0.80)

    def test_unknown_model_tracks_tokens_not_cost(self):
        t = llm.CostTracker()
        t("complete", None, None, "deepseek/deepseek-v4-flash",
          {"input_tokens": 500, "output_tokens": 100})
        assert t.input_tokens == 500
        assert t.cost == 0.0

    def test_reported_cost_beats_table(self):
        """Providers that report billed dollars (OpenRouter) are believed
        verbatim — even for models in the pricing table."""
        t = llm.CostTracker()
        t("complete", None, None, "claude-haiku-4-5",
          {"input_tokens": 1_000_000, "output_tokens": 0,
           "reported_cost": 0.05})
        assert t.cost == pytest.approx(0.05)  # not the table's $0.80
