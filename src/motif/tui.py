"""Textual TUI for multi-stream LLM display.

A viewer that shows multiple LLM streams simultaneously with
streaming markdown rendering. Each stream gets a labeled panel
that updates token-by-token.

    from motif.tui import StreamPanel, MultiStreamApp

Requires the 'tui' extra: pip install motif-llm[tui]
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Markdown, Rule
from textual.widget import Widget
from textual import work


class StreamPanel(Widget):
    """A labeled panel that streams markdown content.

        panel = StreamPanel("Literary", id="literary")
        await panel.stream_from(llm.stream(prompt))
        # or
        await panel.write("chunk of text")
        await panel.finish()
    """

    DEFAULT_CSS = """
    StreamPanel {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    StreamPanel .panel-label {
        dock: top;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    StreamPanel Markdown {
        height: 1fr;
    }
    """

    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._stream = None
        self._md_widget = None

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="panel-label")
        yield VerticalScroll(Markdown(""))

    def on_mount(self):
        self._md_widget = self.query_one(Markdown)
        self._stream = Markdown.get_stream(self._md_widget)

    async def write(self, text: str):
        """Write a chunk of text to the streaming display."""
        if self._stream:
            await self._stream.write(text)

    async def clear(self):
        """Clear and restart the stream."""
        if self._stream:
            await self._stream.stop()
        if self._md_widget:
            await self._md_widget.update("")
            self._stream = Markdown.get_stream(self._md_widget)

    async def finish(self):
        """Signal that the current stream is done."""
        if self._stream:
            await self._stream.stop()

    async def stream_from(self, aiter):
        """Stream from an async iterator (e.g., llm.stream()).

        Returns the full accumulated text.
        """
        full = []
        async for chunk in aiter:
            full.append(chunk)
            await self.write(chunk)
        await self.finish()
        return "".join(full)


class LyricBar(Static):
    """Displays the current lyric line."""

    DEFAULT_CSS = """
    LyricBar {
        dock: top;
        height: 3;
        background: $surface;
        color: $text;
        text-align: center;
        padding: 1;
        text-style: italic;
    }
    """


class MultiStreamApp(App):
    """App with multiple streaming panels and a lyric bar.

    Subclass and override run_analysis() to drive the streams.

        class MyApp(MultiStreamApp):
            async def run_analysis(self):
                for line in lyrics:
                    self.set_lyric(line)
                    await self.panels["literary"].stream_from(
                        llm.stream(prompt)
                    )
    """

    CSS = """
    #panels {
        height: 1fr;
    }
    """

    def __init__(self, panel_names: list[str], title: str = "motif"):
        super().__init__()
        self.title = title
        self._panel_names = panel_names
        self.panels: dict[str, StreamPanel] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield LyricBar("♪ waiting...", id="lyric")
        with Horizontal(id="panels"):
            for name in self._panel_names:
                panel_id = name.lower().replace(" ", "_")
                panel = StreamPanel(name, id=panel_id)
                self.panels[name] = panel
                yield panel
        yield Footer()

    def set_lyric(self, text: str):
        """Update the lyric bar."""
        self.query_one("#lyric", LyricBar).update(f"♪ {text}")

    def on_mount(self):
        self.run_analysis_worker()

    @work
    async def run_analysis_worker(self):
        """Override run_analysis() in subclass."""
        await self.run_analysis()

    async def run_analysis(self):
        """Override this. Drive the panels from here."""
        pass
