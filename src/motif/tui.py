"""Textual TUI viewer for streaming flow executions.

Reads the computation graph directly — no observer wiring needed.
Panels poll graph nodes for output changes via dirty-checking (_version).

    from motif.tui import FlowApp

    app = FlowApp(title="My Pipeline")
    app.run_pipeline(my_pipeline)
    app.run()

The graph builds automatically as flow patterns execute.
The TUI discovers new nodes by polling graph.root_nodes().

Requires: pip install motif-llm[tui]
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll, HorizontalScroll
from textual.widgets import Header, Footer, Static, Markdown
from textual.widget import Widget
from textual import work

from . import graph


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class NodePanel(Widget):
    """A panel that displays a graph node's streaming output.

    Polls node._version every 100ms. When it changes, updates the
    markdown display and status line. No callbacks, no subscriptions.
    """

    DEFAULT_CSS = """
    NodePanel {
        width: 1fr;
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    NodePanel .panel-label {
        dock: top;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 2;
        height: 1;
    }
    NodePanel .panel-status {
        dock: top;
        color: $text-muted;
        padding: 0 2;
        height: 1;
        text-style: italic;
    }
    NodePanel VerticalScroll {
        height: 1fr;
    }
    """

    def __init__(self, node: graph.Node, **kwargs):
        super().__init__(**kwargs)
        self._node = node
        self._rendered_version = -1
        self._md = None

    def compose(self) -> ComposeResult:
        yield Static(self._node.title, classes="panel-label")
        yield Static(self._state_text(), classes="panel-status", id="status")
        yield VerticalScroll(Markdown(""))

    def on_mount(self):
        self._md = self.query_one(Markdown)
        self.set_interval(0.1, self._check)

    def _check(self):
        if self._node._version != self._rendered_version:
            self._rendered_version = self._node._version
            if self._md and self._node.output:
                self._md.update(self._node.output)
            status = self.query_one("#status", Static)
            status.update(self._state_text())

    def _state_text(self) -> str:
        match self._node.state:
            case "pending":
                return "waiting..."
            case "running":
                return "running..."
            case "streaming":
                return "streaming..."
            case "complete":
                return f"done ({self._node.elapsed:.1f}s)"
            case "error":
                return f"error: {self._node.error or 'unknown'}"
            case _:
                return self._node.state


class PanelRow(Widget):
    """A horizontal scrollable row of NodePanels (for fan children)."""

    DEFAULT_CSS = """
    PanelRow {
        height: 1fr;
        max-height: 50%;
    }
    PanelRow HorizontalScroll {
        height: 1fr;
    }
    PanelRow .row-label {
        dock: top;
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    PanelRow NodePanel {
        width: 1fr;
        min-width: 50;
        height: 100%;
    }
    """

    def __init__(self, title: str, nodes: list[graph.Node], **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._graph_nodes = nodes

    def compose(self) -> ComposeResult:
        yield Static(f"\u25c6 {self._title}", classes="row-label")
        with HorizontalScroll():
            for node in self._graph_nodes:
                yield NodePanel(node, id=f"panel-{node.id}")


class SinglePanel(Widget):
    """A full-width panel for a single sequential step."""

    DEFAULT_CSS = """
    SinglePanel {
        height: 1fr;
        max-height: 50%;
        max-width: 100;
    }
    """

    def __init__(self, node: graph.Node, **kwargs):
        super().__init__(**kwargs)
        self._node = node

    def compose(self) -> ComposeResult:
        yield NodePanel(self._node, id=f"panel-{self._node.id}")


class StatusBar(Static):
    """Top bar showing current pipeline status."""

    DEFAULT_CSS = """
    StatusBar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """


def _safe_id(text: str) -> str:
    """Make a string safe for use as a Textual widget ID."""
    return text.lower().replace(" ", "-").replace("(", "").replace(")", "")[:30]


# ---------------------------------------------------------------------------
# Symbols for trace sidebar
# ---------------------------------------------------------------------------

_STATE_SYM = {
    "pending": "\u25cb",     # ○
    "running": "\u25cf",     # ●
    "streaming": "\u25c6",   # ◆
    "complete": "\u2713",    # ✓
    "error": "\u2717",       # ✗
}


# ---------------------------------------------------------------------------
# FlowApp — the main TUI application
# ---------------------------------------------------------------------------

class FlowApp(App):
    """Textual app that builds its layout from the computation graph.

    No observer wiring needed — the app reads graph nodes directly.

        async def pipeline():
            items = await flow.branch(msg, title="discover", schema=S)
            return await flow.fan(items, fn, title="analyze")

        app = FlowApp(title="My Analysis")
        app.run_pipeline(pipeline)
        app.run()
    """

    CSS = """
    #body {
        height: 1fr;
    }
    #main {
        height: 1fr;
        overflow-y: auto;
    }
    #trace-sidebar {
        width: 30;
        dock: left;
        border-right: solid $accent;
        padding: 1;
    }
    .trace-label {
        background: $surface;
        color: $text-muted;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    #trace-content {
        width: 100%;
    }
    """

    def __init__(self, title: str = "motif", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self._main: VerticalScroll | None = None
        self._status: StatusBar | None = None
        self._trace_content: Static | None = None
        self._pipeline_fn = None
        self._pipeline_result = None
        self._seen_nodes: set[str] = set()  # node IDs we've created widgets for

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBar("ready", id="status")
        with Horizontal(id="body"):
            yield VerticalScroll(
                Static("Flow", classes="trace-label"),
                Static("", id="trace-content"),
                id="trace-sidebar",
            )
            yield VerticalScroll(id="main")
        yield Footer()

    def on_mount(self):
        self._main = self.query_one("#main", VerticalScroll)
        self._status = self.query_one("#status", StatusBar)
        self._trace_content = self.query_one("#trace-content", Static)
        # Poll graph for new nodes and trace updates
        self.set_interval(0.2, self._poll_graph)
        self.set_interval(0.5, self._refresh_trace)
        if self._pipeline_fn:
            self._start_pipeline()

    def run_pipeline(self, fn):
        """Set the pipeline to run on mount. fn is an async callable."""
        self._pipeline_fn = fn

    @work(thread=False)
    async def _start_pipeline(self):
        """Run the pipeline in the main async loop.

        In-loop so LLM calls share the event loop with Textual's
        rendering. Graph node updates are immediately visible to polls.
        """
        try:
            if self._status:
                self._status.update("running...")
            graph.reset()  # clean slate for this pipeline
            self._seen_nodes.clear()
            self._pipeline_result = await self._pipeline_fn()
            if self._status:
                self._status.update("done")
        except Exception as e:
            if self._status:
                self._status.update(f"error: {e}")

    # --- Graph polling ---

    def _poll_graph(self):
        """Discover new graph nodes and create widgets for them."""
        for root in graph.root_nodes():
            self._visit(root, is_fan_child=False)

    def _visit(self, node: graph.Node, *, is_fan_child: bool):
        """Walk the graph, creating widgets for new content nodes."""
        if node.id in self._seen_nodes:
            # Already have a widget — but check for new children
            for child in node.children:
                child_is_fan = (node.kind == "fan")
                self._visit(child, is_fan_child=child_is_fan)
            return

        if node.kind == "fan":
            # Fan: create a horizontal row once children exist
            if node.children:
                self._seen_nodes.add(node.id)
                for child in node.children:
                    self._seen_nodes.add(child.id)
                row = PanelRow(node.title, node.children,
                               id=f"row-{node.id}")
                if self._main:
                    self._main.mount(row)
                # Update status
                if self._status:
                    self._status.update(f"\u25c6 {node.title}")
            # else: wait for children to appear (next poll)

        elif node.kind == "reduce":
            self._seen_nodes.add(node.id)
            panel = SinglePanel(node, id=f"single-{node.id}")
            if self._main:
                self._main.mount(panel)
            if self._status:
                self._status.update(f"\u25cf {node.title}")

        elif node.kind == "call" and not is_fan_child:
            # Standalone call (not part of a fan — fan children are
            # handled by PanelRow above)
            self._seen_nodes.add(node.id)
            panel = SinglePanel(node, id=f"single-{node.id}")
            if self._main:
                self._main.mount(panel)

        elif node.kind in ("branch", "compact", "step", "tool_call",
                           "round", "best_of", "cascade", "tournament"):
            # Structural nodes — no panel, just mark as seen
            self._seen_nodes.add(node.id)

        elif node.kind in ("agent", "blackboard", "tree"):
            # Container nodes — no panel for the container itself,
            # but recurse into children
            self._seen_nodes.add(node.id)
            if self._status:
                self._status.update(f"\u25cf {node.title}")

        # Recurse into children we haven't visited
        for child in node.children:
            child_is_fan = (node.kind == "fan")
            self._visit(child, is_fan_child=child_is_fan)

    # --- Trace sidebar ---

    def _refresh_trace(self):
        """Rebuild the trace sidebar from the graph tree."""
        if not self._trace_content:
            return
        lines = []
        for root in graph.root_nodes():
            self._trace_walk(root, lines, indent=0)
        self._trace_content.update("\n".join(lines))

    def _trace_walk(self, node: graph.Node, lines: list[str], indent: int):
        prefix = "  " * indent
        sym = _STATE_SYM.get(node.state, "?")
        elapsed = f" ({node.elapsed:.1f}s)" if node.elapsed else ""
        title = node.title
        if len(title) > 25:
            title = title[:22] + "..."
        lines.append(f"{prefix}{sym} {title}{elapsed}")
        for child in node.children:
            self._trace_walk(child, lines, indent + 1)
