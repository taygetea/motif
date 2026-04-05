"""Observability for flow executions.

Three things:
    Trace            — observer that collects events into a serializable record
    LiveFlowDisplay  — observer that renders a live updating tree (requires rich)
    print_observer   — observer that prints events as they happen (no deps)

The pattern from the outer project: Trace IS an observer AND the record.
Attach it, run the pipeline, inspect or save afterward.

    trace = Trace()
    flow.observe(trace)

    result = await flow.fan(items, fn)

    trace.save("run.json")        # serializable
    print(trace.summary())        # text summary
    trace.events                  # the raw list
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .flow import FlowEvent


# ---------------------------------------------------------------------------
# Trace — the serializable execution record
# ---------------------------------------------------------------------------

class Trace:
    """Collects flow events. Callable — use directly as an observer.

        trace = Trace()
        flow.observe(trace)
        # ... run pipeline ...
        trace.save("output.json")
    """

    def __init__(self):
        self.events: list[FlowEvent] = []
        self._t0: float | None = None

    def __call__(self, event: FlowEvent):
        if self._t0 is None:
            self._t0 = event.timestamp
        self.events.append(event)

    def to_json(self) -> list[dict]:
        """Serialize events for storage or rendering."""
        result = []
        for e in self.events:
            d = asdict(e)
            # Convert timestamp to relative seconds from start
            if self._t0 is not None:
                d["time"] = round(e.timestamp - self._t0, 2)
            del d["timestamp"]
            result.append(d)
        return result

    @classmethod
    def load(cls, path: str) -> Trace:
        """Load a trace from JSON. Events have relative timestamps (seconds
        from start), not monotonic clock values. Loaded traces can be
        summarized and inspected but not meaningfully compared with live events."""
        trace = cls()
        data = json.loads(Path(path).read_text())
        for d in data:
            t = d.pop("time", 0)
            d["timestamp"] = t  # relative time replaces monotonic
            trace.events.append(FlowEvent(**d))
        trace._t0 = 0
        return trace

    def summary(self, *, preview_width: int = 80) -> str:
        """Render a text summary of the execution. No rich dependency."""
        lines = []
        for e in self.events:
            prefix = "  " * e.depth
            elapsed = f" ({e.elapsed:.1f}s)" if e.elapsed else ""
            preview = (e.result or "")[:preview_width]
            if len(e.result or "") > preview_width:
                preview += "..."

            if e.kind == "start":
                lines.append(f"{prefix}● {e.label}")
            elif e.kind == "split":
                children = ", ".join(e.children or [])
                lines.append(f"{prefix}◆ {e.label} → [{children}]{elapsed}")
            elif e.kind == "complete":
                lines.append(f"{prefix}✓ {e.label}{elapsed}")
                if preview:
                    lines.append(f"{prefix}  {preview}")
            elif e.kind == "merge":
                lines.append(f"{prefix}⇐ {e.label}{elapsed}")
                if preview:
                    lines.append(f"{prefix}  {preview}")
            elif e.kind == "error":
                lines.append(f"{prefix}✗ {e.label}: {preview}")

        return "\n".join(lines)

    @property
    def total_elapsed(self) -> float:
        """Total wall time from first to last event."""
        if not self.events:
            return 0
        return self.events[-1].timestamp - self.events[0].timestamp

    def to_markdown(self) -> str:
        """Render the trace as readable markdown.

        Complete results are shown in full (not truncated like summary()).
        The tree structure is preserved with headings and indentation.
        """
        lines = []
        for e in self.events:
            depth = e.depth
            elapsed = f" ({e.elapsed:.1f}s)" if e.elapsed else ""

            match e.kind:
                case "start":
                    pass  # starts are implicit in the structure
                case "split":
                    children = ", ".join(e.children or [])
                    heading = "#" * min(depth + 2, 6)
                    lines.append(f"\n{heading} {e.label} → {children}{elapsed}\n")
                case "complete":
                    heading = "#" * min(depth + 2, 6)
                    lines.append(f"\n{heading} {e.label}{elapsed}\n")
                    if e.result:
                        lines.append(e.result)
                        lines.append("")
                case "merge":
                    heading = "#" * min(depth + 2, 6)
                    lines.append(f"\n{heading} {e.label} (synthesis){elapsed}\n")
                    if e.result:
                        lines.append(e.result)
                        lines.append("")
                case "error":
                    lines.append(f"\n**ERROR: {e.label}**: {e.result or 'unknown'}\n")

        return "\n".join(lines)

    def save(self, path: str, *, format: str = "auto"):
        """Save the trace. Format is inferred from extension or specified.

            trace.save("run.json")       # JSON (machine-readable)
            trace.save("run.md")         # Markdown (human-readable)
            trace.save("run.txt")        # Text summary
            trace.save("run", format="json")  # explicit format
        """
        if format == "auto":
            if path.endswith(".md"):
                format = "markdown"
            elif path.endswith(".txt"):
                format = "text"
            else:
                format = "json"

        match format:
            case "json":
                Path(path).write_text(
                    json.dumps(self.to_json(), indent=2, ensure_ascii=False))
            case "markdown":
                Path(path).write_text(self.to_markdown())
            case "text":
                Path(path).write_text(self.summary())
            case _:
                raise ValueError(f"Unknown format: {format!r}")

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        return f"Trace({len(self.events)} events, {self.total_elapsed:.1f}s)"


# ---------------------------------------------------------------------------
# Live display — rich tree that updates during execution
# ---------------------------------------------------------------------------

# Deferred import: rich is optional (install with prompt-primitives[display])

SYMBOLS = {"running": "⠋", "complete": "✓", "split": "◆", "merge": "⇐", "error": "✗"}
STYLES = {
    "running": "bold yellow", "complete": "green", "split": "bold cyan",
    "merge": "bold magenta", "error": "bold red",
    "label": "bold white", "preview": "dim", "meta": "dim cyan",
}


@dataclass
class _Node:
    label: str
    depth: int
    status: str = "running"
    result_preview: str = ""
    children: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    meta: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)


class LiveFlowDisplay:
    """Observer that renders flow events as a live rich Tree.

    **Label uniqueness assumption:** Nodes are keyed by label string.
    If two flow patterns produce children with the same label (e.g.,
    nested fans both producing "item_0"), the display will merge them.
    Use distinct labels when composing multiple flow patterns.

        display = LiveFlowDisplay()
        flow.observe(display)
        async with display:
            result = await flow.fan(items, fn)

    Optionally attach a Trace too — the display is visual, the trace is the record:

        trace = Trace()
        display = LiveFlowDisplay()
        flow.observe(trace, display)
    """

    def __init__(self, *, preview_width: int = 80):
        from rich.console import Console
        self.console = Console()
        self.preview_width = preview_width
        self._nodes: dict[str, _Node] = {}
        self._roots: list[str] = []
        self._live = None

    def __call__(self, event: FlowEvent):
        label = event.label

        if event.kind == "start":
            if label not in self._nodes:
                self._nodes[label] = _Node(label=label, depth=event.depth,
                                           meta=event.meta)
                if event.depth == 0:
                    self._roots.append(label)

        elif event.kind == "split":
            node = self._nodes.get(label)
            if node:
                node.status = "split"
                node.children = event.children or []
                node.elapsed = event.elapsed
                node.meta.update(event.meta)
                for child in node.children:
                    if child not in self._nodes:
                        self._nodes[child] = _Node(label=child, depth=event.depth + 1)

        elif event.kind in ("complete", "leaf"):
            node = self._nodes.get(label)
            if node:
                node.status = "complete"
                node.result_preview = event.result or ""
                node.elapsed = event.elapsed
                node.meta.update(event.meta)

        elif event.kind == "merge":
            node = self._nodes.get(label)
            if node:
                node.status = "merge"
                node.result_preview = event.result or ""
                node.elapsed = event.elapsed

        elif event.kind == "error":
            node = self._nodes.get(label)
            if node:
                node.status = "error"
                node.result_preview = event.result or ""

        if self._live:
            self._live.update(self._build_tree())

    def _build_tree(self):
        from rich.tree import Tree
        root = Tree("[bold]Flow[/bold]", guide_style="dim")
        for label in self._roots:
            self._add(root, label)
        return root

    def _add(self, parent, label: str):
        from rich.text import Text
        node = self._nodes.get(label)
        if not node:
            parent.add(f"[dim]? {label}[/dim]")
            return
        branch = parent.add(self._format(node))
        for child in node.children:
            self._add(branch, child)

    def _format(self, node: _Node):
        from rich.text import Text
        t = Text()
        sym = SYMBOLS.get(node.status, "?")
        style = STYLES.get(node.status, "")

        t.append(f"{sym} ", style=style)
        t.append(node.label, style=STYLES["label"])

        if node.status == "running":
            elapsed = time.monotonic() - node.start_time
            t.append(f" ({elapsed:.0f}s)", style=STYLES["meta"])
        elif node.status == "split":
            count = node.meta.get("count", len(node.children))
            t.append(f" → {count} branches", style=STYLES["meta"])
            if node.elapsed:
                t.append(f" ({node.elapsed:.1f}s)", style=STYLES["meta"])
        else:
            if node.elapsed:
                t.append(f" ({node.elapsed:.1f}s)", style=STYLES["meta"])
            if node.result_preview:
                preview = node.result_preview[:self.preview_width]
                t.append("\n  ", style="")
                t.append(preview, style=STYLES["preview"])

        return t

    async def __aenter__(self):
        from rich.live import Live
        self._live = Live(
            self._build_tree(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()
        return self

    async def __aexit__(self, *args):
        if self._live:
            self._live.update(self._build_tree())
            self._live.__exit__(*args)
            self._live = None


# ---------------------------------------------------------------------------
# Print observer — no dependencies
# ---------------------------------------------------------------------------

def print_observer(event: FlowEvent):
    """Simple print-based observer. No rich dependency."""
    prefix = "  " * event.depth
    elapsed = f" ({event.elapsed:.1f}s)" if event.elapsed else ""

    if event.kind == "start":
        print(f"{prefix}● {event.label}...")
    elif event.kind == "split":
        children = ", ".join(event.children or [])
        print(f"{prefix}◆ {event.label} → [{children}]{elapsed}")
    elif event.kind == "complete":
        preview = (event.result or "")[:80]
        if len(event.result or "") > 80:
            preview += "..."
        print(f"{prefix}✓ {event.label}{elapsed}: {preview}")
    elif event.kind == "merge":
        preview = (event.result or "")[:80]
        if len(event.result or "") > 80:
            preview += "..."
        print(f"{prefix}⇐ {event.label}{elapsed}: {preview}")
