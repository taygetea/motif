"""Tests for compaction referential integrity — no API calls needed.

The boundary-walking algorithm that preserves tool_use/tool_result
pairs is pure logic. We test it by building Msgs with known structure
and verifying the split never breaks a pair.
"""

from motif import system, user, assistant, tool_use, tool_result, Msg
from motif.prompt import TextSegment, ToolCall, ToolResult as TR


def _split_segments(msg: Msg, keep_recent: int = 6):
    """Reproduce the compaction split logic from flow.py for testing.

    Returns (system_segs, to_compact, to_keep).
    """
    segments = list(msg.segments)

    system_segs = []
    rest = []
    for seg in segments:
        if isinstance(seg, TextSegment) and seg.role == "system":
            system_segs.append(seg)
        else:
            rest.append(seg)

    if len(rest) <= keep_recent:
        return system_segs, [], rest

    split_at = len(rest) - keep_recent

    # Collect tool IDs in tail
    tail_tool_result_ids = set()
    tail_tool_use_ids = set()
    for seg in rest[split_at:]:
        if isinstance(seg, TR):
            tail_tool_result_ids.add(seg.tool_use_id)
        elif isinstance(seg, ToolCall):
            tail_tool_use_ids.add(seg.id)

    # Walk backward to preserve pairs
    while split_at > 0:
        seg = rest[split_at - 1]
        pull = False
        if isinstance(seg, ToolCall) and seg.id in tail_tool_result_ids:
            pull = True
            tail_tool_use_ids.add(seg.id)
        elif isinstance(seg, TR) and seg.tool_use_id in tail_tool_use_ids:
            pull = True
            tail_tool_result_ids.add(seg.tool_use_id)
        if pull:
            split_at -= 1
        else:
            break

    return system_segs, rest[:split_at], rest[split_at:]


def _check_integrity(to_keep):
    """Every tool_result in to_keep references a tool_use also in to_keep."""
    tool_use_ids = {s.id for s in to_keep if isinstance(s, ToolCall)}
    for seg in to_keep:
        if isinstance(seg, TR):
            assert seg.tool_use_id in tool_use_ids, (
                f"Orphaned tool_result referencing {seg.tool_use_id}, "
                f"available tool_use ids: {tool_use_ids}"
            )


class TestCompactionIntegrity:
    def test_simple_split_no_tools(self):
        """Text-only conversation splits cleanly."""
        msg = system("sys") | user("q1") | assistant("a1") | user("q2") | assistant("a2") | user("q3") | assistant("a3") | user("q4")
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=4)
        assert len(sys_segs) == 1
        assert len(kept) == 4
        assert len(compacted) == 3

    def test_tool_pair_not_split(self):
        """A tool_use/tool_result pair straddling the boundary is pulled into kept."""
        msg = (
            system("sys")
            | user("start")
            | tool_use("c1", "search", {"q": "a"})    # this would be compacted...
            | tool_result("c1", "found a")              # ...but this is in kept, so c1 is pulled
            | user("next")
            | assistant("done")
        )
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=3)
        _check_integrity(kept)
        # The tool_use for c1 must be in kept since tool_result references it
        kept_tool_ids = {s.id for s in kept if isinstance(s, ToolCall)}
        assert "c1" in kept_tool_ids

    def test_multiple_tool_pairs(self):
        """Multiple tool pairs — older ones can be compacted, recent ones kept."""
        msg = (
            system("sys")
            | user("q")
            | tool_use("old1", "fn", {})
            | tool_result("old1", "r1")
            | tool_use("old2", "fn", {})
            | tool_result("old2", "r2")
            | user("more")
            | tool_use("new1", "fn", {})
            | tool_result("new1", "r3")
            | assistant("done")
        )
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=4)
        _check_integrity(kept)
        # old pairs can be in compacted — check their integrity too
        _check_integrity(compacted)

    def test_chain_of_tool_calls(self):
        """Three sequential tool calls — splitting in the middle pulls the chain."""
        msg = (
            system("sys")
            | user("go")
            | tool_use("c1", "step1", {})
            | tool_result("c1", "r1")
            | tool_use("c2", "step2", {})
            | tool_result("c2", "r2")
            | tool_use("c3", "step3", {})
            | tool_result("c3", "r3")
            | assistant("final")
        )
        # keep_recent=3 puts c3 result + assistant in tail
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=3)
        _check_integrity(kept)
        _check_integrity(compacted)

    def test_tool_result_at_boundary(self):
        """tool_result is first segment in kept tail — its tool_use gets pulled."""
        msg = (
            system("sys")
            | user("q")
            | assistant("searching")
            | tool_use("c1", "search", {"q": "x"})
            | tool_result("c1", "found")   # would be first in tail
            | user("thanks")
        )
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=2)
        _check_integrity(kept)

    def test_nothing_to_compact(self):
        """Short conversation — nothing compacted."""
        msg = system("sys") | user("q") | assistant("a")
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=6)
        assert compacted == []
        assert len(kept) == 2  # user + assistant

    def test_all_tool_pairs_in_tail(self):
        """If all segments are tool pairs and recent, nothing gets compacted."""
        msg = (
            system("sys")
            | tool_use("c1", "fn", {})
            | tool_result("c1", "r")
            | tool_use("c2", "fn", {})
            | tool_result("c2", "r")
        )
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=4)
        assert compacted == []
        _check_integrity(kept)

    def test_system_segments_always_preserved(self):
        """System segments are never compacted."""
        msg = (
            system("persona", cache=True)
            | system("scene")
            | user("q1") | assistant("a1")
            | user("q2") | assistant("a2")
            | user("q3") | assistant("a3")
            | user("q4") | assistant("a4")
        )
        sys_segs, compacted, kept = _split_segments(msg, keep_recent=4)
        assert len(sys_segs) == 2
        assert all(isinstance(s, TextSegment) and s.role == "system" for s in sys_segs)
