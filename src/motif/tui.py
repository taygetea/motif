"""Textual TUI viewer for streaming flow executions.

Observes both llm (for streaming chunks) and flow (for topology).
Builds the layout dynamically from flow events:
  - split → horizontal scrollable row of panels
  - sequential steps → vertical stacking
  - chunks routed to panels by node label

    from motif.tui import FlowApp

    app = FlowApp()
    flow.observe(app.flow_observer)
    llm.observe(app.llm_observer)
    app.run_async(your_pipeline())

Requires: pip install motif-llm[tui]
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, HorizontalScroll
from textual.message import Message
from textual.widgets import Header, Footer, Static, Markdown, Rule
from textual.widget import Widget
from textual import work

from .flow import FlowEvent


class StreamPanel(Widget):
    """A labeled panel with streaming markdown content."""

    DEFAULT_CSS = """
    StreamPanel {
        width: 40;
        min-width: 30;
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    StreamPanel .panel-label {
        dock: top;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    StreamPanel .panel-status {
        dock: top;
        color: $text-muted;
        padding: 0 1;
        height: 1;
    }
    StreamPanel VerticalScroll {
        height: 1fr;
    }
    """

    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._stream = None
        self._md = None
        self._status_widget = None
        self._text = ""

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="panel-label")
        yield Static("waiting...", classes="panel-status", id="status")
        yield VerticalScroll(Markdown(""))

    def on_mount(self):
        self._md = self.query_one(Markdown)
        self._status_widget = self.query_one("#status", Static)
        self._stream = Markdown.get_stream(self._md)

    async def write_chunk(self, text: str):
        """Write a streaming chunk."""
        self._text += text
        if self._stream:
            await self._stream.write(text)

    def set_status(self, text: str):
        if self._status_widget:
            self._status_widget.update(text)

    async def mark_complete(self, elapsed: float = 0):
        if self._stream:
            await self._stream.stop()
        self.set_status(f"done ({elapsed:.1f}s)")

    async def reset(self):
        """Clear for a new turn."""
        self._text = ""
        if self._stream:
            await self._stream.stop()
        if self._md:
            await self._md.update("")
            self._stream = Markdown.get_stream(self._md)
        self.set_status("streaming...")


class PanelRow(Widget):
    """A horizontal scrollable row of panels (for parallel branches)."""

    DEFAULT_CSS = """
    PanelRow {
        height: 1fr;
        min-height: 15;
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
    """

    def __init__(self, label: str, panel_names: list[str], **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._panel_names = panel_names
        self.panels: dict[str, StreamPanel] = {}

    def compose(self) -> ComposeResult:
        yield Static(f"◆ {self._label}", classes="row-label")
        with HorizontalScroll():
            for name in self._panel_names:
                panel = StreamPanel(name, id=f"panel-{_safe_id(name)}")
                self.panels[name] = panel
                yield panel


class SinglePanel(Widget):
    """A full-width panel for a single sequential step."""

    DEFAULT_CSS = """
    SinglePanel {
        height: auto;
        min-height: 10;
        max-height: 30;
    }
    """

    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self.panel: StreamPanel | None = None

    def compose(self) -> ComposeResult:
        self.panel = StreamPanel(self._label, id=f"panel-{_safe_id(self._label)}")
        yield self.panel


class StatusBar(Static):
    """Top bar showing current status."""

    DEFAULT_CSS = """
    StatusBar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """


def _safe_id(label: str) -> str:
    """Convert a label to a valid Textual widget ID."""
    return label.lower().replace(" ", "-").replace("(", "").replace(")", "")[:30]


class FlowApp(App):
    """Textual app that builds its layout from flow events.

    Attach as observer to both flow and llm:

        app = FlowApp()
        flow.observe(app.flow_observer)
        llm.observe(app.llm_observer)

    Then run your pipeline inside the app:

        async def main():
            app = FlowApp(title="My Analysis")
            flow.observe(app.flow_observer)
            llm.observe(app.llm_observer)

            async def pipeline():
                return await flow.fan(items, fn, streaming=True)

            app.run_pipeline(pipeline)
            app.run()
    """

    CSS = """
    #main {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, title: str = "motif", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self._panels: dict[str, StreamPanel] = {}  # label → panel
        self._main: VerticalScroll | None = None
        self._status: StatusBar | None = None
        self._pipeline_fn = None
        self._pipeline_result = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBar("ready", id="status")
        yield VerticalScroll(id="main")
        yield Footer()

    def on_mount(self):
        self._main = self.query_one("#main", VerticalScroll)
        self._status = self.query_one("#status", StatusBar)
        if self._pipeline_fn:
            self._start_pipeline()

    def run_pipeline(self, fn):
        """Set the pipeline to run on mount. fn is an async callable."""
        self._pipeline_fn = fn

    @work(thread=False)
    async def _start_pipeline(self):
        """Run the pipeline in the main async loop (not a thread).

        This is critical: LLM calls are async and need the event loop.
        Running in-loop means the pipeline shares the event loop with
        Textual's rendering, so observers fire in the right context.
        """
        try:
            if self._status:
                self._status.update("running...")
            self._pipeline_result = await self._pipeline_fn()
            if self._status:
                self._status.update("done")
        except Exception as e:
            if self._status:
                self._status.update(f"error: {e}")

    def flow_observer(self, event: FlowEvent):
        """Attach to flow.observe(). Builds layout from topology.

        Called from within the same async loop as Textual (because
        the pipeline runs with thread=False), so direct widget
        manipulation is safe.
        """
        match event.kind:
            case "split":
                children = event.children or []
                if children and self._main:
                    row = PanelRow(
                        event.label, children,
                        id=f"row-{_safe_id(event.label)}")
                    self._main.mount(row)
                    # Panels aren't available until after mount completes,
                    # so register them with a callback
                    def _register(r=row):
                        for name, panel in r.panels.items():
                            self._panels[name] = panel
                    self.call_later(_register)
                if self._status:
                    self._status.update(f"◆ {event.label} → {len(children)} branches")

            case "start":
                if event.label not in self._panels and self._main:
                    single = SinglePanel(
                        event.label,
                        id=f"single-{_safe_id(event.label)}")
                    self._main.mount(single)
                    def _register(s=single, lbl=event.label):
                        if s.panel:
                            self._panels[lbl] = s.panel
                    self.call_later(_register)

                panel = self._panels.get(event.label)
                if panel:
                    panel.set_status("streaming...")

                if self._status:
                    self._status.update(f"● {event.label}")

            case "complete":
                panel = self._panels.get(event.label)
                if panel:
                    panel.set_status(f"done ({event.elapsed:.1f}s)")

            case "merge":
                if self._status:
                    self._status.update(f"⇐ {event.label} ({event.elapsed:.1f}s)")
                if event.result and self._main:
                    single = SinglePanel(
                        f"⇐ {event.label}",
                        id=f"merge-{_safe_id(event.label)}")
                    self._main.mount(single)
                    def _register(s=single, lbl=f"⇐ {event.label}", r=event.result, el=event.elapsed):
                        if s.panel:
                            self._panels[lbl] = s.panel
                            s.panel.set_status(f"done ({el:.1f}s)")
                    self.call_later(_register)

    def llm_observer(self, verb: str, msg: Any, result: Any, model: str, meta: dict):
        """Attach to llm.observe(). Routes chunks to panels."""
        if verb == "chunk":
            node = meta.get("node")
            if node:
                panel = self._panels.get(node)
                if panel and panel._stream:
                    panel._stream.write(result)
