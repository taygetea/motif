"""Tests for Trace — serialization, roundtrip, summary."""

import json
import tempfile
from pathlib import Path

from motif.flow import FlowEvent
from motif.display import Trace


class TestTraceBasics:
    def test_callable_as_observer(self):
        trace = Trace()
        trace(FlowEvent("start", "test", 0))
        assert len(trace) == 1

    def test_events_collected_in_order(self):
        trace = Trace()
        trace(FlowEvent("start", "a", 0))
        trace(FlowEvent("start", "b", 1))
        trace(FlowEvent("complete", "b", 1, result="done"))
        assert [e.label for e in trace.events] == ["a", "b", "b"]
        assert [e.kind for e in trace.events] == ["start", "start", "complete"]

    def test_repr(self):
        trace = Trace()
        assert "0 events" in repr(trace)
        trace(FlowEvent("start", "x", 0))
        assert "1 events" in repr(trace)


class TestTraceSerialization:
    def test_to_json_structure(self):
        trace = Trace()
        trace(FlowEvent("start", "root", 0, meta={"model": "test"}))
        trace(FlowEvent("complete", "root", 0, result="done", elapsed=1.5))
        data = trace.to_json()
        assert len(data) == 2
        assert data[0]["kind"] == "start"
        assert data[0]["meta"]["model"] == "test"
        assert data[1]["result"] == "done"
        assert data[1]["elapsed"] == 1.5
        # Timestamps are relative
        assert "time" in data[0]
        assert data[0]["time"] == 0.0

    def test_roundtrip_save_load(self):
        trace = Trace()
        trace(FlowEvent("start", "a", 0))
        trace(FlowEvent("split", "a", 0, children=["b", "c"], elapsed=1.0))
        trace(FlowEvent("complete", "b", 1, result="result b", elapsed=0.5))
        trace(FlowEvent("complete", "c", 1, result="result c", elapsed=0.7))
        trace(FlowEvent("merge", "a", 0, result="merged", elapsed=2.0))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        trace.save(path)
        loaded = Trace.load(path)

        assert len(loaded) == len(trace)
        assert [e.kind for e in loaded.events] == [e.kind for e in trace.events]
        assert [e.label for e in loaded.events] == [e.label for e in trace.events]
        assert loaded.events[2].result == "result b"

        Path(path).unlink()

    def test_json_is_valid_json(self):
        trace = Trace()
        trace(FlowEvent("start", "x", 0, meta={"key": "value"}))
        raw = json.dumps(trace.to_json())
        parsed = json.loads(raw)
        assert len(parsed) == 1


class TestTraceSummary:
    def test_summary_includes_all_event_types(self):
        trace = Trace()
        trace(FlowEvent("start", "root", 0))
        trace(FlowEvent("split", "root", 0, children=["a", "b"], elapsed=1.0))
        trace(FlowEvent("complete", "a", 1, result="done a", elapsed=0.5))
        trace(FlowEvent("merge", "root", 0, result="merged", elapsed=2.0))
        trace(FlowEvent("error", "fail", 1, result="bad thing"))

        s = trace.summary()
        assert "●" in s      # start
        assert "◆" in s      # split
        assert "✓" in s      # complete
        assert "⇐" in s      # merge
        assert "✗" in s      # error
        assert "root" in s
        assert "done a" in s

    def test_summary_respects_depth(self):
        trace = Trace()
        trace(FlowEvent("start", "outer", 0))
        trace(FlowEvent("start", "inner", 1))
        s = trace.summary()
        lines = s.split("\n")
        # inner should be indented more than outer
        assert lines[1].startswith("  ")
        assert not lines[0].startswith("  ")

    def test_summary_truncates_long_results(self):
        trace = Trace()
        long_text = "x" * 200
        trace(FlowEvent("complete", "node", 0, result=long_text))
        s = trace.summary(preview_width=50)
        assert len(s.split("\n")[1]) < 60  # truncated
