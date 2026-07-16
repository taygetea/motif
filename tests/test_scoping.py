"""Observer session-scoping.

Registration is scope-local: observers attached inside a
graph.session() live in that session and detach with it. Emission is
additive: process-global observers (attached outside any session) see
every run — a startup-time CostTracker must not go blind because runs
adopted sessions. Nested sessions scope to the innermost.

Note: conftest wraps every test in a session, so "inside a session"
tests open a nested one, and "global" tests touch the module lists
directly (monkeypatched fresh, so nothing leaks).
"""

import asyncio

import httpx
import pytest

import motif.llm as llm
import motif.flow as flow
import motif.show as show
from motif import graph, user, Endpoint
from motif.show import Section


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # Fresh process-global lists per test; session scopes are isolated
    # by conftest's autouse session already.
    monkeypatch.setattr(llm, "_call_observers", [])
    monkeypatch.setattr(flow, "_observers", [])
    monkeypatch.setattr(show, "_show_observers", [])
    yield
    llm._http = None


EP = Endpoint("test-model", base_url="https://example.test/v1")


def _mock_ok():
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })))


class TestLlmScoping:
    @pytest.mark.asyncio
    async def test_session_observers_detach_with_the_session(self):
        _mock_ok()
        events = []
        with graph.session():
            llm.observe_calls(events.append)
        await llm.complete(user("after"), model=EP)
        assert events == []

    @pytest.mark.asyncio
    async def test_global_observers_see_session_events(self):
        _mock_ok()
        events = []
        llm._call_observers.append(events.append)
        with graph.session():
            await llm.complete(user("q"), model=EP)
        assert len(events) == 2  # started + completed

    @pytest.mark.asyncio
    async def test_concurrent_sessions_are_isolated(self):
        _mock_ok()
        results = {}

        async def run(tag):
            with graph.session():
                seen = []
                llm.observe_calls(seen.append)
                await llm.complete(user(tag), model=EP)
                results[tag] = seen

        await asyncio.gather(run("alpha"), run("beta"))
        for tag, seen in results.items():
            assert len(seen) == 2
            assert seen[0].msg.segments[0].text == tag

    def test_clear_observers_clears_only_the_active_scope(self):
        keeper = lambda e: None
        llm._call_observers.append(keeper)
        with graph.session():
            llm.observe_calls(lambda e: None)
            llm.clear_observers()
            assert llm._active_observers() == []
        assert llm._call_observers == [keeper]


class TestSessionBoundary:
    @pytest.mark.asyncio
    async def test_inner_session_does_not_inherit_outer_current_node(self):
        """A session is a fresh run scope: work inside must root in the
        session, not attach to whatever node was current outside."""
        _mock_ok()
        with flow.group("outer") as outer_node:
            with graph.session() as inner:
                await llm.complete(user("q"), model=Endpoint(
                    "test-model", base_url="https://example.test/v1"))
        assert outer_node.children == []
        assert [n.kind for n in inner.roots] == ["llm_call"]


class TestFlowScoping:
    def test_session_observer_sees_events_then_detaches(self):
        seen = []
        with graph.session():
            flow.observe(seen.append)
            flow._emit(flow.FlowEvent("start", "inside"))
        flow._emit(flow.FlowEvent("start", "outside"))
        assert [e.label for e in seen] == ["inside"]

    def test_global_flow_observer_sees_session_events(self):
        seen = []
        flow._observers.append(seen.append)
        with graph.session():
            flow._emit(flow.FlowEvent("start", "x"))
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_observing_removes_from_the_list_it_added_to(self):
        with graph.session():
            async with flow.observing(lambda e: None):
                assert len(flow._session_observers.get()) == 1
            assert flow._session_observers.get() == []


class TestShowScoping:
    def test_session_renderer_detaches_with_the_session(self):
        seen = []
        with graph.session():
            show.show_to(seen.append)
            show.show(Section(title="inside"))
        show.show(Section(title="outside"))
        assert [c.title for c in seen] == ["inside"]
