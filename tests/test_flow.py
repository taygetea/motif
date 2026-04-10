"""Tests for flow patterns with mocked LLM — verify topology, not content."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from motif import system, user, Msg
from motif import flow, graph
from motif.flow import FlowEvent
from motif.display import Trace
from motif.llm import ActResult, ToolRequest


# --- Helpers ---

def mock_complete(response: str = "mock response"):
    """Create a mock llm.complete that returns a fixed string."""
    return AsyncMock(return_value=response)


def mock_extract(response: dict):
    """Create a mock llm.extract that returns a fixed dict."""
    return AsyncMock(return_value=response)


def mock_act_done(text: str = "final answer"):
    """Mock act() that returns a done result."""
    return AsyncMock(return_value=ActResult(text=text, stop_reason="end_turn"))


def mock_act_tool_then_done(tool_name="search", tool_input=None):
    """Mock act() that calls a tool on first call, finishes on second."""
    call_count = 0

    async def _act(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ActResult(
                text="Let me search",
                tool_calls=[ToolRequest(id="c1", name=tool_name, input=tool_input or {})],
                stop_reason="tool_use",
            )
        return ActResult(text="Found it", stop_reason="end_turn")

    return _act


# --- fan ---

class TestFan:
    @pytest.mark.asyncio
    async def test_fires_n_calls(self):
        graph.reset()
        items = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        with patch("motif.flow.llm.complete", new=mock_complete("result")):
            results = await flow.fan(
                items, lambda m: user(m["name"]), title="t", model="test"
            )
        assert len(results) == 3
        assert all(r == "result" for r in results)

    @pytest.mark.asyncio
    async def test_emits_events(self):
        graph.reset()
        trace = Trace()
        flow.observe(trace)
        try:
            with patch("motif.flow.llm.complete", new=mock_complete("r")):
                await flow.fan([{"name": "x"}], lambda m: user("q"), title="t", model="t")
        finally:
            flow.clear_observers()

        kinds = [e.kind for e in trace.events]
        assert "start" in kinds
        assert "complete" in kinds

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """With max_concurrency=1, calls are sequential."""
        graph.reset()
        order = []

        async def _slow_complete(msg, **kw):
            order.append("start")
            await asyncio.sleep(0.01)
            order.append("end")
            return "done"

        with patch("motif.flow.llm.complete", new=_slow_complete):
            results = await flow.fan(
                [{"name": "a"}, {"name": "b"}],
                lambda m: user("q"),
                title="t",
                model="t",
                max_concurrency=1,
            )
        # With concurrency=1, second starts after first ends
        assert order == ["start", "end", "start", "end"]


# --- branch ---

class TestBranch:
    @pytest.mark.asyncio
    async def test_extracts_first_list(self):
        graph.reset()
        response = {"items": [{"name": "a"}, {"name": "b"}]}
        with patch("motif.flow.llm.extract", new=mock_extract(response)):
            items = await flow.branch(user("q"), schema={}, title="t", model="t")
        assert len(items) == 2
        assert items[0]["name"] == "a"

    @pytest.mark.asyncio
    async def test_single_result_wrapped(self):
        """If no list in response, wraps the whole dict as [result]."""
        graph.reset()
        response = {"answer": "42"}
        with patch("motif.flow.llm.extract", new=mock_extract(response)):
            items = await flow.branch(user("q"), schema={}, title="t", model="t")
        assert len(items) == 1
        assert items[0]["answer"] == "42"


# --- reduce ---

class TestReduce:
    @pytest.mark.asyncio
    async def test_passes_combined_text(self):
        graph.reset()
        captured = {}

        async def _capture(msg, **kw):
            # Extract the user text from the Msg
            for seg in msg.segments:
                if hasattr(seg, 'text') and hasattr(seg, 'role') and seg.role == "user":
                    captured["text"] = seg.text
            return "synthesis"

        with patch("motif.flow.llm.complete", new=_capture):
            result = await flow.reduce(
                ["result A", "result B"],
                lambda combined: user(combined),
                title="t",
                model="t",
            )
        assert "result A" in captured["text"]
        assert "result B" in captured["text"]
        assert result == "synthesis"

    @pytest.mark.asyncio
    async def test_labels(self):
        graph.reset()
        captured = {}

        async def _capture(msg, **kw):
            for seg in msg.segments:
                if hasattr(seg, 'text') and hasattr(seg, 'role') and seg.role == "user":
                    captured["text"] = seg.text
            return "done"

        with patch("motif.flow.llm.complete", new=_capture):
            await flow.reduce(
                ["text a", "text b"],
                lambda c: user(c),
                labels=["alpha", "beta"],
                title="t",
                model="t",
            )
        assert "[alpha]:" in captured["text"]
        assert "[beta]:" in captured["text"]


# --- agent ---

class TestAgent:
    @pytest.mark.asyncio
    async def test_direct_answer(self):
        """Model answers immediately without tools."""
        graph.reset()
        with patch("motif.flow.llm.act", new=mock_act_done("the answer")):
            result = await flow.agent(
                user("question"),
                tools={},
                tool_schemas=[],
                model="t",
                max_tokens=0,
            )
        assert result.output == "the answer"
        assert result.signal is None
        assert result.steps == 1

    @pytest.mark.asyncio
    async def test_tool_then_answer(self):
        """Model calls a tool, then answers."""
        graph.reset()
        async def fake_search(inp):
            return "search result"

        with patch("motif.flow.llm.act", new=mock_act_tool_then_done("search", {"q": "x"})):
            result = await flow.agent(
                user("find x"),
                tools={"search": fake_search},
                tool_schemas=[],
                model="t",
                max_tokens=0,
            )
        assert result.output == "Found it"
        assert result.steps == 2
        # Msg should contain tool_use and tool_result segments
        seg_types = [type(s).__name__ for s in result.msg.segments]
        assert "ToolCall" in seg_types
        assert "ToolResult" in seg_types

    @pytest.mark.asyncio
    async def test_preserves_assistant_narration(self):
        """Text before tool calls is preserved in the Msg."""
        graph.reset()
        async def fake_tool(inp):
            return "ok"

        with patch("motif.flow.llm.act", new=mock_act_tool_then_done("fn")):
            result = await flow.agent(
                user("go"),
                tools={"fn": fake_tool},
                tool_schemas=[],
                model="t",
                max_tokens=0,
            )
        # Should have assistant("Let me search") before the tool_use
        from motif.prompt import TextSegment
        assistant_segs = [s for s in result.msg.segments
                         if isinstance(s, TextSegment) and s.role == "assistant"]
        assert any("Let me search" in s.text for s in assistant_segs)

    @pytest.mark.asyncio
    async def test_max_steps(self):
        """Agent stops after max_steps."""
        graph.reset()
        # act() always returns tool calls — never finishes
        always_tool = AsyncMock(return_value=ActResult(
            tool_calls=[ToolRequest(id="c1", name="fn", input={})],
            stop_reason="tool_use",
        ))

        async def fake_tool(inp):
            return "ok"

        with patch("motif.flow.llm.act", new=always_tool):
            result = await flow.agent(
                user("go"),
                tools={"fn": fake_tool},
                tool_schemas=[],
                model="t",
                max_steps=3,
                max_tokens=0,
            )
        assert result.signal == "max_steps"
        assert result.steps == 3

    @pytest.mark.asyncio
    async def test_signal_tool_breaks_loop(self):
        """A signal tool terminates the loop."""
        graph.reset()
        async def finish_handler(inp):
            return inp.get("answer", "done")

        act_calls = AsyncMock(return_value=ActResult(
            tool_calls=[ToolRequest(id="c1", name="finish", input={"answer": "42"})],
            stop_reason="tool_use",
        ))

        with patch("motif.flow.llm.act", new=act_calls):
            result = await flow.agent(
                user("go"),
                tools={"finish": finish_handler},
                tool_schemas=[],
                signal_tools={"finish": flow.FINISH},
                model="t",
                max_tokens=0,
            )
        assert result.signal == flow.FINISH
        assert result.output == "42"

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Unknown tool name produces error tool_result, loop continues."""
        graph.reset()
        call_count = 0

        async def _act(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ActResult(
                    tool_calls=[ToolRequest(id="c1", name="nonexistent", input={})],
                    stop_reason="tool_use",
                )
            return ActResult(text="recovered", stop_reason="end_turn")

        with patch("motif.flow.llm.act", new=_act):
            result = await flow.agent(
                user("go"),
                tools={},
                tool_schemas=[],
                model="t",
                max_tokens=0,
            )
        assert result.output == "recovered"
        # The error tool_result should be in the Msg
        from motif.prompt import ToolResult as TR
        error_results = [s for s in result.msg.segments
                        if isinstance(s, TR) and s.is_error]
        assert len(error_results) == 1
