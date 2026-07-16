"""Coverage for previously untested flow-pattern behavior."""

from unittest.mock import AsyncMock, patch

import pytest

from motif import flow, graph, user


def _user_text(msg):
    return "\n".join(
        segment.text
        for segment in msg.segments
        if getattr(segment, "role", None) == "user"
    )


class TestBestOf:
    @pytest.mark.asyncio
    async def test_selects_winner_and_preserves_judgment_order(self):
        graph.reset()
        candidates = ["draft zero", "draft one", "draft two"]
        judgments = [
            {"score": 3, "note": "first"},
            {"score": 9, "note": "second"},
            {"score": 5, "note": "third"},
        ]
        by_candidate = dict(zip(candidates, judgments))

        async def _extract(msg, **kw):
            return by_candidate[_user_text(msg)]

        with patch("motif.flow.llm.extract", new=_extract):
            winner, winner_idx, returned_judgments = await flow.best_of(
                candidates,
                lambda candidate: user(candidate),
                {"type": "object"},
                title="choose draft",
                model="test",
            )

        assert winner == "draft one"
        assert winner_idx == 1
        assert returned_judgments == judgments
        assert all(
            returned is original
            for returned, original in zip(returned_judgments, judgments)
        )

        roots = graph.root_nodes()
        assert len(roots) == 1
        node = roots[0]
        assert node.kind == "best_of"
        assert node.title == "choose draft"
        assert node.state == "complete"
        assert node.output == "winner: #1 (score 9)"
        assert node.output.splitlines() == [node.output]
        assert node.children == []

    @pytest.mark.asyncio
    async def test_custom_score_key_selects_winner(self):
        graph.reset()
        extract = AsyncMock(side_effect=[
            {"quality": 0.8, "score": 100},
            {"quality": 0.9, "score": 0},
        ])

        with patch("motif.flow.llm.extract", new=extract):
            winner, winner_idx, judgments = await flow.best_of(
                ["a", "b"],
                lambda candidate: user(candidate),
                {"type": "object"},
                title="custom scoring",
                model="test",
                score_key="quality",
            )

        assert (winner, winner_idx) == ("b", 1)
        assert judgments == [
            {"quality": 0.8, "score": 100},
            {"quality": 0.9, "score": 0},
        ]
        assert graph.root_nodes()[0].output == "winner: #1 (score 0.9)"

    @pytest.mark.asyncio
    async def test_error_marks_node_and_is_reraised(self):
        graph.reset()
        failure = RuntimeError("judge unavailable")
        extract = AsyncMock(side_effect=failure)

        # TaskGroup semantics (2026-07-16): a failing judgment cancels
        # its siblings and surfaces as an ExceptionGroup, same as fan.
        with patch("motif.flow.llm.extract", new=extract):
            with pytest.raises(ExceptionGroup) as exc_info:
                await flow.best_of(
                    ["a", "b"],
                    lambda candidate: user(candidate),
                    {"type": "object"},
                    title="failed judging",
                    model="test",
                )

        assert failure in exc_info.value.exceptions
        roots = graph.root_nodes()
        assert len(roots) == 1
        assert roots[0].kind == "best_of"
        assert roots[0].state == "error"
        assert "judge unavailable" in roots[0].error


class TestTournament:
    @pytest.mark.asyncio
    async def test_odd_byes_bracket_log_and_original_winner_index(self):
        graph.reset()
        candidates = ["A", "B", "C", "D", "E"]
        judgments = {
            "A vs B": {"winner": "b", "reason": "B wins"},
            "C vs D": {"winner": "a", "reason": "C wins"},
            "E vs B": {"winner": "a", "reason": "E wins"},
            "C vs E": {"winner": "b", "reason": "E wins again"},
        }
        compared = []

        def _judge(a, b):
            compared.append((a, b))
            return user(f"{a} vs {b}")

        async def _extract(msg, **kw):
            return judgments[_user_text(msg)]

        with patch("motif.flow.llm.extract", new=_extract):
            winner, winner_idx, rounds_log = await flow.tournament(
                candidates,
                _judge,
                {"type": "object"},
                title="bracket",
                model="test",
            )

        assert winner == "E"
        assert winner_idx == 4
        assert compared == [
            ("A", "B"),
            ("C", "D"),
            ("E", "B"),
            ("C", "E"),
        ]
        assert rounds_log == [
            [
                {
                    "a_idx": 0,
                    "b_idx": 1,
                    "winner_idx": 1,
                    "judgment": judgments["A vs B"],
                },
                {
                    "a_idx": 2,
                    "b_idx": 3,
                    "winner_idx": 2,
                    "judgment": judgments["C vs D"],
                },
            ],
            [
                {
                    "a_idx": 4,
                    "b_idx": 1,
                    "winner_idx": 4,
                    "judgment": judgments["E vs B"],
                },
            ],
            [
                {
                    "a_idx": 2,
                    "b_idx": 4,
                    "winner_idx": 4,
                    "judgment": judgments["C vs E"],
                },
            ],
        ]

        node = graph.root_nodes()[0]
        assert node.kind == "tournament"
        assert node.output == "winner: #4"
        assert node.meta["rounds"] == 3
        assert [child.title for child in node.children] == [
            "round 1",
            "round 2",
            "round 3",
        ]

    @pytest.mark.asyncio
    async def test_single_candidate_returns_without_node_or_judgment(self):
        graph.reset()
        extract = AsyncMock()

        with patch("motif.flow.llm.extract", new=extract):
            result = await flow.tournament(
                ["only"],
                lambda a, b: user("unused"),
                {"type": "object"},
                title="no bracket needed",
                model="test",
            )

        assert result == ("only", 0, [])
        extract.assert_not_awaited()
        assert graph.root_nodes() == []


class TestCall:
    @pytest.mark.asyncio
    async def test_text_mode_routes_to_complete(self):
        graph.reset()
        msg = user("question")
        complete = AsyncMock(return_value="answer")
        extract = AsyncMock()

        with patch("motif.flow.llm.complete", new=complete), \
             patch("motif.flow.llm.extract", new=extract):
            result = await flow.call(msg, title="text call", model="test")

        assert result == "answer"
        complete.assert_awaited_once_with(msg, model="test")
        extract.assert_not_awaited()
        node = graph.root_nodes()[0]
        assert node.kind == "call"
        assert node.output == "answer"

    @pytest.mark.asyncio
    async def test_schema_mode_routes_to_extract_and_previews_dict(self):
        graph.reset()
        msg = user("question")
        schema = {"type": "object"}
        extracted = {"answer": "forty-two", "confidence": 0.9}
        complete = AsyncMock()
        extract = AsyncMock(return_value=extracted)

        with patch("motif.flow.llm.complete", new=complete), \
             patch("motif.flow.llm.extract", new=extract):
            result = await flow.call(
                msg,
                schema=schema,
                title="structured call",
                model="test",
            )

        assert result is extracted
        extract.assert_awaited_once_with(msg, schema=schema, model="test")
        complete.assert_not_awaited()
        node = graph.root_nodes()[0]
        assert node.kind == "call"
        # The annotation node stays silent in schema mode (2026-07-16):
        # the llm_call record beneath it carries the full JSON, and a
        # lossy preview here would replace it in the document.
        assert node.output == ""

    @pytest.mark.asyncio
    async def test_label_kwarg_is_rejected_before_calling_llm(self):
        graph.reset()
        complete = AsyncMock()

        with patch("motif.flow.llm.complete", new=complete):
            with pytest.raises(TypeError, match="Use title= instead of label="):
                await flow.call(
                    user("question"),
                    title="new title",
                    label="old label",
                    model="test",
                )

        complete.assert_not_awaited()
        assert graph.root_nodes() == []

    @pytest.mark.asyncio
    async def test_error_marks_node_and_is_reraised(self):
        graph.reset()
        failure = RuntimeError("completion unavailable")
        complete = AsyncMock(side_effect=failure)

        with patch("motif.flow.llm.complete", new=complete):
            with pytest.raises(RuntimeError, match="completion unavailable") as exc_info:
                await flow.call(
                    user("question"),
                    title="failed call",
                    model="test",
                )

        assert exc_info.value is failure
        roots = graph.root_nodes()
        assert len(roots) == 1
        assert roots[0].kind == "call"
        assert roots[0].state == "error"
        assert roots[0].error == "completion unavailable"


class TestGuards:
    """Boundary failures fail loudly, and intermediate nodes never
    stay 'running' after an error (found during the 2026-07-16 gap
    coverage; fixed the same day)."""

    @pytest.mark.asyncio
    async def test_best_of_empty_candidates_raises_at_boundary(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            await flow.best_of(
                [], lambda c: user(c), {"type": "object"},
                title="empty", model="test")
        assert graph.root_nodes() == []

    @pytest.mark.asyncio
    async def test_tournament_rejects_invalid_winner_value(self):
        extract = AsyncMock(return_value={"winner": "banana"})
        with patch("motif.flow.llm.extract", new=extract):
            with pytest.raises(ValueError, match="must be 'a' or 'b'"):
                await flow.tournament(
                    ["x", "y"], lambda a, b: user("m"), {"type": "object"},
                    title="strict", model="test")
        node = graph.root_nodes()[0]
        assert node.state == "error"
        rounds = [c for c in node.children if c.kind == "round"]
        assert rounds and all(r.state == "error" for r in rounds)

    @pytest.mark.asyncio
    async def test_tournament_judge_failure_errors_the_round(self):
        extract = AsyncMock(side_effect=RuntimeError("judge down"))
        with patch("motif.flow.llm.extract", new=extract):
            with pytest.raises(ExceptionGroup):
                await flow.tournament(
                    ["x", "y"], lambda a, b: user("m"), {"type": "object"},
                    title="bracket", model="test")
        node = graph.root_nodes()[0]
        rounds = [c for c in node.children if c.kind == "round"]
        assert rounds and all(r.state == "error" for r in rounds)

    @pytest.mark.asyncio
    async def test_blackboard_agent_failure_errors_the_round(self):
        complete = AsyncMock(side_effect=RuntimeError("expert down"))
        with patch("motif.flow.llm.complete", new=complete):
            with pytest.raises(ExceptionGroup):
                await flow.blackboard(
                    [("expert", lambda b: user(b))], "seed",
                    title="panel", rounds=1, model="test")
        node = graph.root_nodes()[0]
        rounds = [c for c in node.children if c.kind == "round"]
        assert rounds and all(r.state == "error" for r in rounds)

    @pytest.mark.asyncio
    async def test_cascade_model_failure_errors_the_attempt(self):
        complete = AsyncMock(side_effect=RuntimeError("model down"))
        with patch("motif.flow.llm.complete", new=complete):
            with pytest.raises(RuntimeError):
                await flow.cascade(
                    user("q"), lambda r: user(r), {"type": "object"},
                    models=["cheap", "expensive"], title="ladder")
        node = graph.root_nodes()[0]
        attempts = [c for c in node.children if c.kind == "call"]
        assert attempts and all(a.state == "error" for a in attempts)


class TestBranchLabels:
    @pytest.mark.asyncio
    async def test_label_key_and_fallback_chain(self):
        graph.reset()
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
            },
        }
        items = [
            {
                "display": "declared field",
                "name": "ignored name",
                "label": "ignored label",
                "title": "ignored title",
            },
            {"name": "name fallback", "label": "ignored", "title": "ignored"},
            {"label": "label fallback", "title": "ignored"},
            {"title": "title fallback"},
            {},
        ]
        extract = AsyncMock(return_value={"items": items})

        with patch("motif.flow.llm.extract", new=extract):
            result = await flow.branch(
                user("discover"),
                schema,
                title="labeled branch",
                label_key="display",
                model="test",
            )

        assert result is items
        node = graph.root_nodes()[0]
        assert node.kind == "branch"
        assert node.output == (
            "declared field, name fallback, label fallback, "
            "title fallback, item_4"
        )
