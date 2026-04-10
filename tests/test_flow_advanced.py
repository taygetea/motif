"""Tests for flow patterns not covered by test_flow.py:
agent truncation, agent timeout, cascade, blackboard, compact, tree."""

import asyncio
from unittest.mock import AsyncMock, patch, call

import pytest

from motif import system, user, assistant, tool_use, tool_result, Msg, Block
from motif.prompt import TextSegment, ToolCall, ToolResult as TR
from motif import flow, graph
from motif.flow import FlowEvent
from motif.llm import ActResult, ToolRequest


# ---------------------------------------------------------------------------
# Agent: truncation recovery
# ---------------------------------------------------------------------------

class TestAgentTruncation:
    @pytest.mark.asyncio
    async def test_max_tokens_continues(self):
        """stop_reason='max_tokens' appends partial text and retries."""
        graph.reset()
        call_count = 0

        async def _act(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ActResult(text="partial respo", stop_reason="max_tokens")
            return ActResult(text="full response", stop_reason="end_turn")

        with patch("motif.flow.llm.act", new=_act):
            result = await flow.agent(
                user("go"), tools={}, tool_schemas=[],
                model="t", max_tokens=0,
            )

        assert result.output == "full response"
        assert result.steps == 2
        # Partial text should be in the Msg as an assistant segment
        assistant_segs = [s for s in result.msg.segments
                         if isinstance(s, TextSegment) and s.role == "assistant"]
        assert any("partial respo" in s.text for s in assistant_segs)

    @pytest.mark.asyncio
    async def test_max_tokens_no_text(self):
        """Truncation with no text still continues."""
        graph.reset()
        call_count = 0

        async def _act(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ActResult(text=None, stop_reason="max_tokens")
            return ActResult(text="recovered", stop_reason="end_turn")

        with patch("motif.flow.llm.act", new=_act):
            result = await flow.agent(
                user("go"), tools={}, tool_schemas=[],
                model="t", max_tokens=0,
            )
        assert result.output == "recovered"


# ---------------------------------------------------------------------------
# Agent: timeout
# ---------------------------------------------------------------------------

class TestAgentTimeout:
    @pytest.mark.asyncio
    async def test_timeout_fires(self):
        """Agent returns signal='timeout' when wall clock exceeds limit."""
        graph.reset()
        async def _slow_act(*args, **kwargs):
            await asyncio.sleep(0.1)
            return ActResult(
                tool_calls=[ToolRequest(id="c1", name="fn", input={})],
                stop_reason="tool_use",
            )

        async def fake_tool(inp):
            return "ok"

        with patch("motif.flow.llm.act", new=_slow_act):
            result = await flow.agent(
                user("go"),
                tools={"fn": fake_tool},
                tool_schemas=[],
                model="t",
                max_steps=100,
                max_tokens=0,
                timeout=0.25,
            )

        assert result.signal == "timeout"
        assert result.steps > 0


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

class TestCascade:
    @pytest.mark.asyncio
    async def test_first_model_sufficient(self):
        """If the first model passes, no escalation."""
        graph.reset()
        with patch("motif.flow.llm.complete", new=AsyncMock(return_value="answer")), \
             patch("motif.flow.llm.extract", new=AsyncMock(return_value={"sufficient": True})):
            result, model_used = await flow.cascade(
                user("q"),
                test_fn=lambda ans: user(ans),
                test_schema={},
                models=["cheap", "expensive"],
                title="t",
            )
        assert result == "answer"
        assert model_used == "cheap"

    @pytest.mark.asyncio
    async def test_escalates_on_insufficient(self):
        """Insufficient first model escalates to second."""
        graph.reset()
        complete_call_count = 0

        async def _complete(msg, *, model, **kw):
            nonlocal complete_call_count
            complete_call_count += 1
            return f"answer from {model}"

        extract_call_count = 0

        async def _extract(msg, schema, *, model, **kw):
            nonlocal extract_call_count
            extract_call_count += 1
            # First model insufficient, never called for second (last model)
            return {"sufficient": False}

        with patch("motif.flow.llm.complete", new=_complete), \
             patch("motif.flow.llm.extract", new=_extract):
            result, model_used = await flow.cascade(
                user("q"),
                test_fn=lambda ans: user(ans),
                test_schema={},
                models=["cheap", "mid", "expensive"],
                title="t",
            )

        assert model_used == "expensive"
        assert "expensive" in result

    @pytest.mark.asyncio
    async def test_last_model_accepts_regardless(self):
        """Last model is accepted without testing."""
        graph.reset()
        extract_mock = AsyncMock(return_value={"sufficient": False})

        with patch("motif.flow.llm.complete", new=AsyncMock(return_value="ans")), \
             patch("motif.flow.llm.extract", new=extract_mock):
            result, model_used = await flow.cascade(
                user("q"),
                test_fn=lambda ans: user(ans),
                test_schema={},
                models=["only"],
                title="t",
            )

        assert model_used == "only"
        # extract should NOT have been called — last model skips test
        extract_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------

class TestBlackboard:
    @pytest.mark.asyncio
    async def test_rounds_and_agents(self):
        """N agents × M rounds produces correct history structure."""
        graph.reset()
        agent_responses = iter(["a1r1", "b1r1", "a1r2", "b1r2"])

        async def _complete(msg, **kw):
            return next(agent_responses)

        with patch("motif.flow.llm.complete", new=_complete):
            board, history = await flow.blackboard(
                agents=[
                    ("alice", lambda b: user(b)),
                    ("bob", lambda b: user(b)),
                ],
                seed="topic",
                rounds=2,
                title="t",
            )

        assert len(history) == 2
        assert set(history[0].keys()) == {"alice", "bob"}
        assert history[0]["alice"] == "a1r1"
        assert history[1]["bob"] == "b1r2"

    @pytest.mark.asyncio
    async def test_filter_fn_called(self):
        """filter_fn receives (board, history, name, round)."""
        graph.reset()
        filter_calls = []

        def _filter(board, history, name, rnd):
            filter_calls.append((name, rnd, len(history)))
            return board  # pass through

        with patch("motif.flow.llm.complete", new=AsyncMock(return_value="resp")):
            await flow.blackboard(
                agents=[("a", lambda b: user(b)), ("b", lambda b: user(b))],
                seed="topic",
                rounds=2,
                filter_fn=_filter,
                title="t",
            )

        # 2 agents × 2 rounds = 4 filter calls
        assert len(filter_calls) == 4
        names = [c[0] for c in filter_calls]
        assert names.count("a") == 2
        assert names.count("b") == 2
        # Round 1: history has 0 entries; round 2: history has 1 entry
        r1_calls = [c for c in filter_calls if c[1] == 1]
        r2_calls = [c for c in filter_calls if c[1] == 2]
        assert all(c[2] == 0 for c in r1_calls)
        assert all(c[2] == 1 for c in r2_calls)

    @pytest.mark.asyncio
    async def test_filter_fn_controls_visibility(self):
        """filter_fn can restrict what an agent sees."""
        graph.reset()
        seen_by_bob = []

        async def _complete(msg, **kw):
            # Capture what was sent to the model
            for seg in msg.segments:
                if isinstance(seg, TextSegment) and seg.role == "user":
                    seen_by_bob.append(seg.text)
            return "response"

        def _filter(board, history, name, rnd):
            if name == "bob":
                return "FILTERED"
            return board

        with patch("motif.flow.llm.complete", new=_complete):
            await flow.blackboard(
                agents=[
                    ("alice", lambda b: user(b)),
                    ("bob", lambda b: user(b)),
                ],
                seed="topic",
                rounds=1,
                filter_fn=_filter,
                title="t",
            )

        # Bob should have seen "FILTERED", not the full board
        assert any("FILTERED" in s for s in seen_by_bob)


# ---------------------------------------------------------------------------
# Compact (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------

class TestCompactEndToEnd:
    @pytest.mark.asyncio
    async def test_under_threshold_noop(self):
        """compact() returns the Msg unchanged when under threshold."""
        graph.reset()
        msg = system("sys") | user("short")
        result = await flow.compact(msg, max_tokens=100_000)
        assert result is msg  # exact same object

    @pytest.mark.asyncio
    async def test_over_threshold_summarizes(self):
        """compact() calls llm.complete and produces a shorter Msg."""
        graph.reset()
        # Build a long Msg
        msg = system("sys", cache=True)
        for i in range(30):
            msg = msg | user(f"question {i} " + "x" * 200)
            msg = msg | assistant(f"answer {i} " + "y" * 200)

        with patch("motif.flow.llm.complete", new=AsyncMock(return_value="Summary of conversation.")):
            result = await flow.compact(msg, max_tokens=500)

        # Should be shorter
        assert len(result.segments) < len(msg.segments)
        # System segment preserved
        sys_segs = [s for s in result.segments if isinstance(s, TextSegment) and s.role == "system"]
        assert len(sys_segs) == 1
        assert sys_segs[0].cache is True
        # Summary segment present
        assert any("Summary" in s.text for s in result.segments
                   if isinstance(s, TextSegment))

    @pytest.mark.asyncio
    async def test_tool_pairs_preserved_after_compact(self):
        """compact() preserves tool_use/tool_result referential integrity."""
        graph.reset()
        msg = system("sys")
        for i in range(20):
            msg = msg | tool_use(f"c{i}", "fn", {}) | tool_result(f"c{i}", f"r{i}")
        msg = msg | user("final")

        with patch("motif.flow.llm.complete", new=AsyncMock(return_value="Summary.")):
            result = await flow.compact(msg, max_tokens=100)

        # Check integrity of kept segments
        tool_use_ids = {s.id for s in result.segments if isinstance(s, ToolCall)}
        for seg in result.segments:
            if isinstance(seg, TR):
                assert seg.tool_use_id in tool_use_ids


# ---------------------------------------------------------------------------
# Tree (recursive split → analyze → merge)
# ---------------------------------------------------------------------------

class TestTree:
    @pytest.mark.asyncio
    async def test_leaf_no_split(self):
        """If splitter says is_leaf, analyze directly."""
        graph.reset()
        with patch("motif.flow.llm.extract", new=AsyncMock(return_value={"is_leaf": True})), \
             patch("motif.flow.llm.complete", new=AsyncMock(return_value="leaf analysis")):
            result = await flow.tree(
                task="short text",
                split_fn=lambda t: user(t),
                split_schema={},
                leaf_fn=lambda t: user(t),
                merge_fn=lambda rs, ls: user(Block.join(rs, labels=ls)),
                title="t",
            )
        assert result == "leaf analysis"

    @pytest.mark.asyncio
    async def test_split_then_merge(self):
        """Splitter returns subtasks → recurse → merge."""
        graph.reset()
        extract_call_count = 0

        async def _extract(msg, schema, **kw):
            nonlocal extract_call_count
            extract_call_count += 1
            if extract_call_count == 1:
                # First call: split into two
                return {
                    "is_leaf": False,
                    "subtasks": [
                        {"label": "part A", "start_paragraph": 0, "end_paragraph": 1},
                        {"label": "part B", "start_paragraph": 1, "end_paragraph": 2},
                    ],
                }
            # Subsequent calls: leaf
            return {"is_leaf": True}

        complete_calls = []

        async def _complete(msg, **kw):
            complete_calls.append("called")
            if len(complete_calls) <= 2:
                return f"analysis {len(complete_calls)}"
            return "merged result"

        with patch("motif.flow.llm.extract", new=_extract), \
             patch("motif.flow.llm.complete", new=_complete):
            result = await flow.tree(
                task="paragraph one\n\nparagraph two",
                split_fn=lambda t: user(t),
                split_schema={},
                leaf_fn=lambda t: user(t),
                merge_fn=lambda rs, ls: user(Block.join(rs, labels=ls)),
                title="t",
            )

        assert result == "merged result"
        # 2 leaf analyses + 1 merge = 3 complete calls
        assert len(complete_calls) == 3

    @pytest.mark.asyncio
    async def test_max_depth_forces_leaf(self):
        """At max_depth, analyze directly even if content is large."""
        graph.reset()
        with patch("motif.flow.llm.extract") as mock_extract, \
             patch("motif.flow.llm.complete", new=AsyncMock(return_value="forced leaf")):
            result = await flow.tree(
                task="text",
                split_fn=lambda t: user(t),
                split_schema={},
                leaf_fn=lambda t: user(t),
                merge_fn=lambda rs, ls: user(Block.join(rs, labels=ls)),
                max_depth=0,  # force immediate leaf
                title="t",
            )

        assert result == "forced leaf"
        mock_extract.assert_not_called()  # never asked to split
