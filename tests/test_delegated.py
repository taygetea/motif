"""Delegated endpoints: the codex adapter at the real subprocess
boundary (fake_codex.py stands in for the CLI — real process spawn,
real stdin, real JSONL parsing), plus the requires= preflight, the
lifecycle events with attachments, and the graph record.
"""

import asyncio
import json
import os
import sys

import pytest

import motif.llm as llm
import motif.delegated as delegated
from motif import (
    graph, system, user, tool_use,
    Endpoint, DelegatedEndpoint, DelegationFailed, CallStarted,
    CallCompleted, CallFailed,
)

FAKE = [sys.executable, os.path.join(os.path.dirname(__file__), "fake_codex.py")]


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr(delegated, "CODEX_BIN", FAKE)
    monkeypatch.setattr(delegated, "_cli_version", None)
    monkeypatch.setattr(llm, "_delegated_semaphores", {})
    llm.clear_observers()
    yield
    llm.clear_observers()


def _scenario(monkeypatch, name):
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", name)


EP = DelegatedEndpoint.codex(search=True)


def _events():
    seen = []
    llm.observe_calls(seen.append)
    return seen


# --- The adapter itself ---

class TestAdapter:
    @pytest.mark.asyncio
    async def test_prompt_crosses_stdin_and_flags_are_hermetic(self, monkeypatch, tmp_path):
        """fake_codex asserts the hermetic flag set and stdin transport
        internally (it crashes without them); the log proves what
        actually crossed the boundary."""
        _scenario(monkeypatch, "ok")
        log = tmp_path / "invocation.json"
        monkeypatch.setenv("FAKE_CODEX_LOG", str(log))

        result = await delegated.run_codex("hello harness", search=True)

        assert result.failure is None
        assert result.text.startswith("fake answer to: hello harness")
        seen = json.loads(log.read_text())
        assert seen["prompt"] == "hello harness"
        assert "tools.web_search=true" in seen["argv"]
        assert "motif-delegated-" in seen["cwd"]  # fresh ephemeral cwd

    @pytest.mark.asyncio
    async def test_usage_normalized_from_last_turn(self, monkeypatch):
        _scenario(monkeypatch, "ok")
        result = await delegated.run_codex("q")
        assert result.usage["input_tokens"] == 1000
        assert result.usage["output_tokens"] == 60  # output + reasoning
        assert result.usage["cache_read_tokens"] == 400
        assert result.usage["reported_cost"] == 0.0
        assert result.usage["billing_basis"] == "subscription"

    @pytest.mark.asyncio
    async def test_deadline_kills_and_classifies(self, monkeypatch):
        _scenario(monkeypatch, "hang")
        result = await delegated.run_codex("q", timeout=1.0)
        assert result.failure == "deadline_exceeded"
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_failure_classification(self, monkeypatch):
        _scenario(monkeypatch, "autherr")
        assert (await delegated.run_codex("q")).failure == "auth_failed"
        _scenario(monkeypatch, "ratelimit")
        assert (await delegated.run_codex("q")).failure == "rate_limited"
        _scenario(monkeypatch, "noanswer")
        assert (await delegated.run_codex("q")).failure == "no_answer"

    @pytest.mark.asyncio
    async def test_non_json_noise_is_tolerated(self, monkeypatch):
        _scenario(monkeypatch, "garbage")
        result = await delegated.run_codex("q")
        assert result.failure is None
        assert result.text.startswith("fake answer")

    @pytest.mark.asyncio
    async def test_fingerprint_pins_the_invocation(self, monkeypatch):
        _scenario(monkeypatch, "ok")
        result = await delegated.run_codex("q")
        assert result.fingerprint["adapter"] == "codex"
        assert result.fingerprint["cli_version"] == "fake-codex 0.0.1"
        assert result.fingerprint["hermetic"] is True


# --- The verbs over a delegated endpoint ---

class TestDelegatedVerbs:
    @pytest.mark.asyncio
    async def test_complete_records_one_call_with_attachments(self, monkeypatch):
        _scenario(monkeypatch, "ok")
        events = _events()
        msg = system("Be terse.") | user("What is a monoid?")
        text = await llm.complete(msg, model=EP, requires={"web_search"})

        assert text.startswith("fake answer")
        assert [type(e) for e in events] == [CallStarted, CallCompleted]
        kinds = {a.kind for a in events[1].attachments}
        assert {"harness_transcript", "flat_prompt", "attestation"} <= kinds

        node = graph.root_nodes()[0]
        assert node.kind == "llm_call"
        assert node.state == "complete"
        assert node.meta["model"] == "delegated:codex"
        assert {a.kind for a in node.attachments} == kinds
        flat = next(a for a in node.attachments if a.kind == "flat_prompt")
        assert "<instructions>" in flat.data and "Be terse." in flat.data

    @pytest.mark.asyncio
    async def test_extract_round_trips_schema(self, monkeypatch):
        _scenario(monkeypatch, "schema")
        data = await llm.extract(user("q"),
                                 {"type": "object",
                                  "properties": {"answer": {"type": "integer"}}},
                                 model=EP)
        assert data["answer"] == 42  # schema file reached the harness

    @pytest.mark.asyncio
    async def test_extract_degrades_when_strict_mode_rejects(self, monkeypatch):
        """OpenAI strict mode can reject a schema even after
        strictification — the ladder falls to schema-in-prompt,
        invisibly, in the same lifecycle."""
        _scenario(monkeypatch, "schemareject")
        events = _events()
        data = await llm.extract(user("q"), {"type": "object"}, model=EP)
        assert data == {"answer": 43}
        # one lifecycle despite two harness invocations
        assert [type(e) for e in events] == [CallStarted, CallCompleted]

    def test_strictify_adds_additional_properties_recursively(self):
        schema = {"type": "object",
                  "properties": {"items": {"type": "array",
                                           "items": {"type": "object",
                                                     "properties": {}}}}}
        strict = delegated.strictify_schema(schema)
        assert strict["additionalProperties"] is False
        assert strict["properties"]["items"]["items"]["additionalProperties"] is False
        assert "additionalProperties" not in schema  # original untouched

    @pytest.mark.asyncio
    async def test_failure_raises_with_kind_and_attaches_transcript(self, monkeypatch):
        _scenario(monkeypatch, "autherr")
        events = _events()
        with pytest.raises(DelegationFailed) as exc:
            await llm.complete(user("q"), model=EP)
        assert exc.value.kind == "auth_failed"
        failed = events[-1]
        assert type(failed) is CallFailed
        kinds = {a.kind for a in failed.attachments}
        assert "harness_stderr" in kinds and "harness_transcript" in kinds
        assert graph.root_nodes()[0].state == "error"

    @pytest.mark.asyncio
    async def test_history_bearing_msg_fails_loudly(self):
        msg = user("q") | tool_use("t1", "search", {"q": "x"})
        with pytest.raises(ValueError, match="silently drop"):
            await llm.complete(msg, model=EP)

    @pytest.mark.asyncio
    async def test_act_and_stream_reject_delegated(self):
        with pytest.raises(TypeError, match="flow.agent"):
            await llm.act(user("q"), [], model=EP)
        with pytest.raises(TypeError, match="complete"):
            async for _ in llm.stream(user("q"), model=EP):
                pass

    @pytest.mark.asyncio
    async def test_cost_tracker_counts_subscription_tokens_at_zero_dollars(self, monkeypatch):
        _scenario(monkeypatch, "ok")
        tracker = llm.CostTracker()
        llm.observe_calls(tracker)
        await llm.complete(user("q"), model=EP)
        assert tracker.calls == 1
        assert tracker.input_tokens == 1000
        assert tracker.cost == 0.0  # reported_cost 0: subscription lane

    @pytest.mark.asyncio
    async def test_admission_respects_max_concurrency(self, monkeypatch):
        """With max_concurrency=1 the two hanging calls serialize —
        the second's deadline covers the first's full run."""
        _scenario(monkeypatch, "ok")
        ep = DelegatedEndpoint.codex(search=False, max_concurrency=1)
        in_flight = {"now": 0, "peak": 0}
        real = delegated.run_codex

        async def counting(*a, **kw):
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            try:
                return await real(*a, **kw)
            finally:
                in_flight["now"] -= 1

        monkeypatch.setattr(delegated, "run_codex", counting)
        await asyncio.gather(
            llm.complete(user("a"), model=ep),
            llm.complete(user("b"), model=ep),
        )
        assert in_flight["peak"] == 1


# --- requires= preflight ---

class TestRequires:
    @pytest.mark.asyncio
    async def test_missing_capability_fails_before_spending(self, monkeypatch):
        called = {"n": 0}

        async def boom(*a, **kw):
            called["n"] += 1

        monkeypatch.setattr(delegated, "run_codex", boom)
        no_search = DelegatedEndpoint.codex(search=False)
        with pytest.raises(ValueError, match="web_search"):
            await llm.complete(user("q"), model=no_search,
                               requires={"web_search"})
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_plain_endpoint_fails_requires(self):
        with pytest.raises(ValueError, match="capabilities"):
            await llm.complete(user("q"), model=Endpoint("some-model"),
                               requires={"web_search"})

    @pytest.mark.asyncio
    async def test_api_endpoint_can_declare_capabilities(self, monkeypatch):
        """An OpenRouter :online model can honestly claim web_search —
        requires= passes and the ordinary transport proceeds."""
        import httpx
        online = Endpoint("some-model:online",
                          base_url="https://example.test/v1",
                          capabilities=frozenset({"web_search"}))
        llm._http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})))
        try:
            result = await llm.complete(user("q"), model=online,
                                        requires={"web_search"})
            assert result == "ok"
        finally:
            llm._http = None
