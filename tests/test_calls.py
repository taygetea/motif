"""Tests for the call-lifecycle protocol.

Three layers, tested at their own altitude:
  - llm.py emits CallStarted → CallChunk* → (CallCompleted | CallFailed)
    with per-call identity — tested at the HTTP boundary (MockTransport),
    so real request/response code runs.
  - The legacy (verb, msg, result, model, meta) observer signature is
    derived from those events by an adapter — its fidelity is mostly
    proven by test_llm.py passing unchanged; the failure-suppression
    contract is pinned here.
  - record.py folds events into graph nodes — tested both through real
    verb calls and as a pure fold over hand-built events.
"""

import json

import httpx
import pytest

import motif  # installs the graph projection
import motif.llm as llm
import motif.flow as flow
from motif import graph, user, system, Endpoint, role
from motif.llm import CallStarted, CallChunk, CallCompleted, CallFailed
from motif.show import narrate
from motif.graph import Node


@pytest.fixture(autouse=True)
def _clean_state():
    token = llm._profile.set(None)
    llm.clear_observers()
    yield
    llm._profile.reset(token)
    llm.clear_observers()
    llm._http = None


def _mock_http(handler):
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chat_response(content="hello", finish="stop", usage=None):
    return {
        "id": "gen-1",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


EP = Endpoint("test-model", base_url="https://example.test/v1")

SSE = ('data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
       'data: {"choices": [{"delta": {"content": "lo"}}], '
       '"usage": {"prompt_tokens": 3, "completion_tokens": 2}}\n\n'
       'data: [DONE]\n\n')


def _events():
    """Attach a call observer, return the list it appends to."""
    seen = []
    llm.observe_calls(seen.append)
    return seen


# --- Event emission from the verbs ---

class TestEventEmission:
    @pytest.mark.asyncio
    async def test_complete_emits_started_then_completed(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("ans")))
        events = _events()
        msg = system("s") | user("q")
        await llm.complete(msg, model=EP, max_tokens=99)

        assert [type(e) for e in events] == [CallStarted, CallCompleted]
        started, completed = events
        assert started.call_id == completed.call_id
        assert started.verb == "complete"
        assert started.msg is msg          # the actual Msg, retained
        assert started.declared is EP
        assert started.endpoint is EP
        assert started.params["max_tokens"] == 99
        assert completed.result == "ans"
        assert completed.usage["input_tokens"] == 10
        assert completed.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_declared_role_survives_resolution(self):
        """The event carries both what the author declared (the role)
        and what actually ran (the endpoint)."""
        _mock_http(lambda r: httpx.Response(200, json=_chat_response()))
        llm.use_profile({"swarm": EP})
        events = _events()
        await llm.complete(user("q"), model=role("swarm"))
        started = events[0]
        assert started.declared == role("swarm")
        assert started.endpoint is EP

    @pytest.mark.asyncio
    async def test_failure_emits_failed(self):
        _mock_http(lambda r: httpx.Response(401, json={"error": "bad key"}))
        events = _events()
        with pytest.raises(httpx.HTTPStatusError):
            await llm.complete(user("q"), model=EP)

        assert [type(e) for e in events] == [CallStarted, CallFailed]
        assert events[1].call_id == events[0].call_id
        assert "401" in events[1].error
        assert isinstance(events[1].exception, httpx.HTTPStatusError)

    @pytest.mark.asyncio
    async def test_truncation_completes_then_raises(self):
        """A truncated call DID complete at the transport — the event
        carries the cost and stop_reason; the verb raises after."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("part", finish="length")))
        events = _events()
        with pytest.raises(llm.Truncated):
            await llm.complete(user("q"), model=EP)

        assert [type(e) for e in events] == [CallStarted, CallCompleted]
        assert events[1].stop_reason == "max_tokens"  # anthropic vocabulary
        assert events[1].result == "part"

    @pytest.mark.asyncio
    async def test_extract_params_carry_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        _mock_http(lambda r: httpx.Response(200, json=_chat_response('{"x": 3}')))
        events = _events()
        result = await llm.extract(user("q"), schema, model=EP)
        assert result == {"x": 3}
        assert events[0].verb == "extract"
        assert events[0].params["schema"] is schema
        assert events[1].result == {"x": 3}

    @pytest.mark.asyncio
    async def test_extract_truncation_fails_with_usage(self):
        """A truncated extract is a CallFailed — there is no valid
        partial dict — but the transport billed us, so the failure
        event carries the usage. Emitted exactly once."""
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response('{"x"', finish="length")))
        events = _events()
        with pytest.raises(llm.Truncated):
            await llm.extract(user("q"), {"type": "object"}, model=EP)
        assert [type(e) for e in events] == [CallStarted, CallFailed]
        assert events[1].usage["output_tokens"] == 5
        assert isinstance(events[1].exception, llm.Truncated)

    @pytest.mark.asyncio
    async def test_extract_parse_failure_emits_failed(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("not json")))
        events = _events()
        with pytest.raises(json.JSONDecodeError):
            await llm.extract(user("q"), {"type": "object"}, model=EP)
        assert [type(e) for e in events] == [CallStarted, CallFailed]

    @pytest.mark.asyncio
    async def test_act_completes_with_actresult(self):
        _mock_http(lambda r: httpx.Response(
            200, json=_chat_response("done", finish="stop")))
        events = _events()
        tools = [{"name": "t", "input_schema": {}}]
        await llm.act(user("q"), tools, model=EP)
        assert events[0].verb == "act"
        assert events[0].params["tools"] is tools
        assert events[1].result.text == "done"
        assert events[1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_emits_chunks_between(self):
        _mock_http(lambda r: httpx.Response(200, content=SSE.encode()))
        events = _events()
        chunks = [c async for c in llm.stream(user("q"), model=EP)]
        assert chunks == ["hel", "lo"]

        assert [type(e) for e in events] == \
            [CallStarted, CallChunk, CallChunk, CallCompleted]
        assert {e.call_id for e in events} == {events[0].call_id}
        assert [e.text for e in events[1:3]] == ["hel", "lo"]
        assert events[3].result == "hello"
        assert events[3].usage["input_tokens"] == 3

    @pytest.mark.asyncio
    async def test_stream_is_lazy(self):
        """The call starts when consumption starts — creating the
        generator alone is not a call."""
        _mock_http(lambda r: httpx.Response(200, content=SSE.encode()))
        events = _events()
        gen = llm.stream(user("q"), model=EP)
        assert events == []
        await gen.__anext__()
        assert type(events[0]) is CallStarted
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_stream_abandoned_emits_failed(self):
        """Every started call settles exactly once, even when the
        consumer walks away mid-stream."""
        _mock_http(lambda r: httpx.Response(200, content=SSE.encode()))
        events = _events()
        gen = llm.stream(user("q"), model=EP)
        await gen.__anext__()
        await gen.aclose()

        assert type(events[-1]) is CallFailed
        assert "abandoned" in events[-1].error

    @pytest.mark.asyncio
    async def test_resamples_have_distinct_identity(self):
        """Two identical calls are two facts — the whole point."""
        _mock_http(lambda r: httpx.Response(200, json=_chat_response()))
        events = _events()
        msg = user("same prompt")
        await llm.complete(msg, model=EP)
        await llm.complete(msg, model=EP)
        ids = {e.call_id for e in events}
        assert len(ids) == 2


# --- Legacy adapter contract ---

class TestLegacyAdapter:
    @pytest.mark.asyncio
    async def test_failures_are_not_notified(self):
        """The legacy protocol never notified failures; the adapter
        preserves that (and must not leak started-state)."""
        _mock_http(lambda r: httpx.Response(500, json={"error": "boom"}))
        calls = []
        llm.observe(lambda *a: calls.append(a))
        with pytest.raises(httpx.HTTPStatusError):
            await llm.complete(user("q"), model=EP)
        assert calls == []
        # adapter state cleaned up: no dangling started entries
        adapter = llm._call_observers[-1]
        assert adapter._started == {}

    @pytest.mark.asyncio
    async def test_five_tuple_shape(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("ans")))
        seen = {}

        def legacy(verb, msg, result, model, meta):
            seen.update(verb=verb, msg=msg, result=result, model=model, meta=meta)

        llm.observe(legacy)
        msg = user("q")
        await llm.complete(msg, model=EP, meta={"node": "x"})
        assert seen["verb"] == "complete"
        assert seen["msg"] is msg
        assert seen["result"] == "ans"
        assert seen["model"] == "test-model"  # resolved model string, as before
        assert seen["meta"]["node"] == "x"
        assert seen["meta"]["input_tokens"] == 10


# --- The graph projection, through real verbs ---

class TestProjection:
    @pytest.mark.asyncio
    async def test_bare_complete_becomes_root_node(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("ans")))
        msg = user("q")
        await llm.complete(msg, model=EP)

        roots = graph.root_nodes()
        assert len(roots) == 1
        node = roots[0]
        assert node.kind == "llm_call"
        assert node.title == "complete"
        assert node.state == "complete"
        assert node.output == "ans"
        assert node.msg is msg                      # retained input
        assert node.meta["endpoint"] == "test-model"
        assert node.meta["usage"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_node_id_is_call_id(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response()))
        events = _events()
        await llm.complete(user("q"), model=EP)
        assert graph.root_nodes()[0].id == events[0].call_id

    @pytest.mark.asyncio
    async def test_role_declaration_labels_node(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("{}")))
        llm.use_profile({"structure": EP})
        await llm.extract(user("q"), {"type": "object"}, model=role("structure"))
        node = graph.root_nodes()[0]
        assert node.meta["model"] == "role:structure"
        assert node.meta["endpoint"] == "test-model"

    @pytest.mark.asyncio
    async def test_failed_call_is_an_error_node(self):
        _mock_http(lambda r: httpx.Response(401, json={"error": "no"}))
        with pytest.raises(httpx.HTTPStatusError):
            await llm.complete(user("q"), model=EP)
        node = graph.root_nodes()[0]
        assert node.state == "error"
        assert "401" in node.error

    @pytest.mark.asyncio
    async def test_resamples_are_distinct_nodes(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response()))
        msg = user("same")
        await llm.complete(msg, model=EP)
        await llm.complete(msg, model=EP)
        roots = graph.root_nodes()
        assert len(roots) == 2
        assert roots[0].id != roots[1].id

    @pytest.mark.asyncio
    async def test_flow_call_records_beneath_annotation(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("out")))
        await flow.call(user("q"), title="synthesis", model=EP)

        roots = graph.root_nodes()
        assert len(roots) == 1
        annotation = roots[0]
        assert annotation.kind == "call"
        records = [c for c in annotation.children if c.kind == "llm_call"]
        assert len(records) == 1
        assert records[0].output == "out"

    @pytest.mark.asyncio
    async def test_fan_items_carry_their_call_records(self):
        _mock_http(lambda r: httpx.Response(200, json=_chat_response("r")))
        await flow.fan([{"name": "a"}, {"name": "b"}],
                       lambda item: user(item["name"]),
                       title="work", model=EP)

        fan_node = graph.root_nodes()[0]
        assert fan_node.kind == "fan"
        assert [c.kind for c in fan_node.children] == ["item", "item"]
        for slot in fan_node.children:
            records = [c for c in slot.children if c.kind == "llm_call"]
            assert len(records) == 1
            assert records[0].meta["node"] == slot.title

    @pytest.mark.asyncio
    async def test_stream_chunks_land_on_the_record(self):
        _mock_http(lambda r: httpx.Response(200, content=SSE.encode()))
        collected = []
        async for chunk in llm.stream(user("q"), model=EP):
            collected.append((chunk, graph.root_nodes()[0].output))

        node = graph.root_nodes()[0]
        assert node.output == "hello"
        assert node.state == "complete"
        # output grew as chunks arrived, not only at the end
        assert collected[0][1] == "hel"

    @pytest.mark.asyncio
    async def test_abandoned_stream_is_an_error_node(self):
        _mock_http(lambda r: httpx.Response(200, content=SSE.encode()))
        gen = llm.stream(user("q"), model=EP)
        await gen.__anext__()
        await gen.aclose()
        node = graph.root_nodes()[0]
        assert node.state == "error"
        assert "abandoned" in node.error


# --- The projection as a pure fold ---

class TestProjectionFold:
    def test_events_for_unknown_ids_are_noops(self):
        from motif.record import project
        project(CallChunk("nope", "x"))
        project(CallCompleted("nope", "r", {}))
        project(CallFailed("nope", "e"))
        assert graph.root_nodes() == []

    def test_dict_results_render_fully(self):
        from motif.record import project
        project(CallStarted("c1", "extract", user("q"), "m",
                            Endpoint("m"), {}, {}))
        project(CallCompleted("c1", {"key": "value"}, {"input_tokens": 1}))
        node = graph.root_nodes()[0]
        assert json.loads(node.output) == {"key": "value"}
        assert node.meta["usage"] == {"input_tokens": 1}

    def test_failed_usage_reaches_meta(self):
        from motif.record import project
        project(CallStarted("c2", "extract", user("q"), "m",
                            Endpoint("m"), {}, {}))
        project(CallFailed("c2", "truncated", usage={"output_tokens": 7}))
        node = graph.root_nodes()[0]
        assert node.state == "error"
        assert node.meta["usage"] == {"output_tokens": 7}


# --- narrate policy for call records ---

def _node(kind, title, output="", meta=None, children=None, error=None):
    return Node(id=title, kind=kind, title=title, output=output,
                meta=meta or {}, children=children or [],
                state="complete", error=error)


class TestNarratePolicy:
    def test_root_record_is_shown(self):
        doc = narrate([_node("llm_call", "complete", output="The answer.")])
        assert "The answer." in doc

    def test_record_under_narrating_parent_is_hidden(self):
        """A pattern node that displays its own output narrates for its
        children — the record is redundant there."""
        record = _node("llm_call", "complete", output="Raw result.")
        parent = _node("call", "synthesis", output="Raw result.",
                       children=[record])
        doc = narrate([parent])
        assert doc.count("Raw result.") == 1

    def test_record_under_silent_parent_is_shown(self):
        """An output-less container (a group, a bare loop's phase) has
        no narrative of its own — the records are the content."""
        record = _node("llm_call", "complete", output="Loop result.")
        parent = _node("group", "phase 1", children=[record])
        doc = narrate([parent])
        assert "Loop result." in doc

    def test_structure_role_record_collapses(self):
        doc = narrate([_node("llm_call", "extract", output="one-liner",
                             meta={"model": "role:structure"})])
        assert "- **extract**" in doc

    def test_failed_record_surfaces_through_narrating_parent(self):
        record = _node("llm_call", "complete", error="rate limited")
        parent = _node("call", "synthesis", output="partial",
                       children=[record])
        doc = narrate([parent])
        assert "rate limited" in doc

    def test_tree_decomposition_lists_subtrees_not_records(self):
        subtree = _node("tree", "part one", output="sub result")
        record = _node("llm_call", "extract", output="{}")
        parent = _node("tree", "decompose", output="merged",
                       children=[record, subtree])
        doc = narrate([parent])
        assert "part one" in doc
        assert "extract" not in doc
