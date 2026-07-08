"""Tests for narrate() — the graph→markdown fold — and graph.session().

All pure: fake Node trees in, markdown out. No LLM calls.
"""

import asyncio

import pytest

from motif import graph
from motif.graph import Node, enter_node, exit_node
from motif.show import narrate, _node_policy


def _node(kind, title, output="", children=None, meta=None, error=None):
    return Node(id=title, kind=kind, title=title, output=output,
                children=children or [], meta=meta or {}, error=error,
                state="error" if error else "complete")


# --- graph.session ---

class TestSession:
    def test_roots_isolated_per_session(self):
        with graph.session() as s1:
            n1, p1 = enter_node("call", "one")
            exit_node(n1, p1)
        with graph.session() as s2:
            n2, p2 = enter_node("call", "two")
            exit_node(n2, p2)
        assert [n.title for n in s1.roots] == ["one"]
        assert [n.title for n in s2.roots] == ["two"]

    def test_root_nodes_reads_active_session(self):
        with graph.session() as s:
            n, p = enter_node("call", "inside")
            exit_node(n, p)
            assert [x.title for x in graph.root_nodes()] == ["inside"]

    @pytest.mark.asyncio
    async def test_concurrent_sessions_do_not_interleave(self):
        """Two pipelines in sibling tasks, each with its own session."""
        async def run(tag, sess_holder):
            with graph.session() as s:
                sess_holder.append(s)
                for i in range(3):
                    n, p = enter_node("call", f"{tag}-{i}")
                    await asyncio.sleep(0)  # force interleaving
                    exit_node(n, p)

        holders: list = []
        await asyncio.gather(run("a", holders), run("b", holders))
        titles_a = {n.title for n in holders[0].roots}
        titles_b = {n.title for n in holders[1].roots}
        assert all(t.startswith("a-") for t in titles_a)
        assert all(t.startswith("b-") for t in titles_b)
        assert len(titles_a) == len(titles_b) == 3

    def test_reset_clears_active_scope(self):
        with graph.session() as s:
            n, p = enter_node("call", "x")
            exit_node(n, p)
            graph.reset()
            assert s.roots == []


# --- salience policy ---

class TestPolicy:
    def test_compact_hidden(self):
        assert _node_policy(_node("compact", "compact")) == "hidden"

    def test_structure_kinds_collapsed(self):
        for kind in ("branch", "best_of", "cascade", "tournament"):
            assert _node_policy(_node(kind, kind)) == "collapsed"

    def test_content_call_shown(self):
        n = _node("call", "synthesis", meta={"model": "role:content"})
        assert _node_policy(n) == "shown"

    def test_structure_call_collapsed(self):
        n = _node("call", "reframe", meta={"model": "role:structure"})
        assert _node_policy(n) == "collapsed"

    def test_author_override_wins(self):
        n = _node("branch", "important", meta={"show": "shown"})
        assert _node_policy(n) == "shown"
        n2 = _node("call", "noise", meta={"show": "hidden"})
        assert _node_policy(n2) == "hidden"

    def test_error_promotes(self):
        n = _node("branch", "broken", error="boom")
        assert _node_policy(n) == "shown"


# --- the fold ---

class TestNarrate:
    def test_call_becomes_section(self):
        doc = narrate([_node("call", "synthesis", output="The report body.")])
        assert "## synthesis" in doc
        assert "The report body." in doc

    def test_branch_collapses_to_line(self):
        doc = narrate([_node("branch", "decompose", output="a, b, c")])
        assert "## decompose" not in doc
        assert "- **decompose** — a, b, c" in doc

    def test_compact_invisible(self):
        doc = narrate([_node("compact", "compact", output="123 → 45 tokens")])
        assert doc == ""

    def test_hidden_error_still_surfaces(self):
        doc = narrate([_node("compact", "compact", error="api died")])
        assert "⚠ error in compact" in doc

    def test_fan_children_are_sections(self):
        fan = _node("fan", "analyses", children=[
            _node("call", "rhetoric", output="Rhetoric analysis."),
            _node("call", "logic", output="Logic analysis."),
        ])
        doc = narrate([fan])
        assert "## analyses" in doc
        assert "### rhetoric" in doc
        assert "Logic analysis." in doc

    def test_wide_fan_collapses(self):
        fan = _node("fan", "swarm", children=[
            _node("call", f"c{i}", output=f"out {i}") for i in range(20)
        ])
        doc = narrate([fan], fan_limit=12)
        assert "20 calls (collapsed):" in doc
        assert "### c0" not in doc
        assert "- **c0** — out 0" in doc

    def test_agent_shows_final_and_activity(self):
        agent = _node("agent", "researcher", output="Final brief.", children=[
            _node("step", "step 1", children=[
                _node("tool_call", "web_search"),
            ]),
            _node("step", "step 2", output="2 tool calls"),
        ])
        doc = narrate([agent])
        assert "## researcher" in doc
        assert "Final brief." in doc
        assert "- step 1: web_search" in doc

    def test_blackboard_rounds(self):
        bb = _node("blackboard", "panel", output="board summary…", children=[
            _node("round", "round 1", children=[
                _node("call", "economist", output="Take one."),
                _node("call", "historian", output="Take two."),
            ]),
        ])
        doc = narrate([bb])
        assert "## panel" in doc
        assert "### round 1" in doc
        assert "**economist**: Take one." in doc
        # the truncated board summary is NOT the content
        assert "board summary…" not in doc

    def test_show_override_in_document(self):
        shown_branch = _node("branch", "the angles", output="a, b",
                             meta={"show": "shown"})
        hidden_call = _node("call", "scratch", output="junk",
                            meta={"show": "hidden"})
        doc = narrate([shown_branch, hidden_call])
        assert "## the angles" in doc
        assert "junk" not in doc

    def test_error_call_annotated(self):
        doc = narrate([_node("call", "flaky", output="partial", error="timeout")])
        assert "⚠ error in flaky: timeout" in doc
        assert "partial" in doc

    def test_document_order_follows_node_order(self):
        doc = narrate([
            _node("call", "first", output="1"),
            _node("call", "second", output="2"),
        ])
        assert doc.index("first") < doc.index("second")


class TestHeadingDemotion:
    def test_embedded_h1_nests_under_section(self):
        n = _node("call", "brief", output="# Research Brief\n\n## Core\n\nBody.")
        doc = narrate([n])  # section at level 2
        assert "## brief" in doc
        assert "### Research Brief" in doc
        assert "#### Core" in doc
        assert "\n# Research Brief" not in doc

    def test_code_fences_untouched(self):
        n = _node("call", "code", output="```python\n# not a heading\n```")
        doc = narrate([n])
        assert "# not a heading" in doc
        assert "### not a heading" not in doc

    def test_already_nested_output_unchanged(self):
        n = _node("call", "ok", output="#### Deep heading\n\ntext")
        doc = narrate([n])
        assert "#### Deep heading" in doc
