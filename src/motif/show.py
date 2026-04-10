"""Display components and renderers.

Pipeline authors emit display components via flow.show().
Renderers consume them to produce output in various formats.

    from motif import flow
    from motif.show import Section, Panels, MarkdownRenderer

    renderer = MarkdownRenderer()
    async with flow.showing(renderer):
        flow.show(Section(title="Analysis"))
        results = await flow.fan(items, fn, title="parallel work")
        flow.show(Panels(items=results, titles=names))

    print(renderer.output())

Components are immutable dataclasses. They're React-shaped because
Claude writes pipelines and Claude knows JSX patterns.

Renderers are homomorphisms: Component -> format.
The renderer is dumb — presentation logic lives in render methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Component base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    """Base for display components. Immutable description of what to render.

    Built-in components implement render_markdown(). Custom components
    are just new dataclasses that inherit Component and implement
    the render methods they need.
    """

    def render_markdown(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement render_markdown()")

    def render_html(self) -> str:
        """Default: wrap markdown. Override for real HTML."""
        return f"<div>{self.render_markdown()}</div>"


# ---------------------------------------------------------------------------
# Built-in components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Section(Component):
    """A titled section. The basic document structure element."""
    title: str
    content: str | None = None
    level: int = 1

    def render_markdown(self) -> str:
        heading = "#" * min(self.level, 6)
        parts = [f"{heading} {self.title}"]
        if self.content:
            parts.append(self.content)
        return "\n\n".join(parts)


@dataclass(frozen=True)
class ContentBlock(Component):
    """A titled content block. For single outputs that deserve a frame.

    Named ContentBlock (not Block) to avoid collision with prompt.Block.
    """
    title: str
    content: str

    def render_markdown(self) -> str:
        return f"### {self.title}\n\n{self.content}"


@dataclass(frozen=True)
class Panels(Component):
    """Side-by-side content blocks. For parallel outputs."""
    items: list[str]
    titles: list[str]

    def __post_init__(self):
        if len(self.items) != len(self.titles):
            raise ValueError(
                f"items ({len(self.items)}) must match titles ({len(self.titles)})")

    def render_markdown(self) -> str:
        parts = []
        for title, content in zip(self.titles, self.items):
            parts.append(f"### {title}\n\n{content}")
        return "\n\n---\n\n".join(parts)


@dataclass(frozen=True)
class Chat(Component):
    """Multi-party conversation display."""
    participants: list[str]
    messages: list[dict]  # [{name: str, text: str}, ...]

    def render_markdown(self) -> str:
        lines = []
        for msg in self.messages:
            name = msg.get("name", "unknown")
            text = msg.get("text", "")
            lines.append(f"**{name}**: {text}")
        return "\n\n".join(lines)


@dataclass(frozen=True)
class Code(Component):
    """Code block with language annotation."""
    content: str
    language: str = ""

    def render_markdown(self) -> str:
        return f"```{self.language}\n{self.content}\n```"


@dataclass(frozen=True)
class Table(Component):
    """Structured data as table."""
    headers: list[str]
    rows: list[list[str]]

    def render_markdown(self) -> str:
        if not self.headers:
            return ""
        lines = ["| " + " | ".join(self.headers) + " |"]
        lines.append("| " + " | ".join("---" for _ in self.headers) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


@dataclass(frozen=True)
class Progress(Component):
    """Status update for long-running operations.

    Emissions with the same group replace the previous one.
    """
    group: str
    status: str
    detail: str = ""

    def render_markdown(self) -> str:
        if self.detail:
            return f"*{self.status}*: {self.detail}"
        return f"*{self.status}*"


# ---------------------------------------------------------------------------
# Show machinery — the display observer list
# ---------------------------------------------------------------------------

_show_observers: list[Callable[[Component], None]] = []


def show(component: Component):
    """Emit a display component to all attached renderers."""
    for obs in _show_observers:
        try:
            obs(component)
        except Exception:
            pass  # renderers should not break the pipeline


def show_to(*observers: Callable[[Component], None]):
    """Attach display observers (renderers)."""
    _show_observers.extend(observers)


def clear_show_observers():
    """Remove all display observers."""
    _show_observers.clear()


class showing:
    """Context manager that attaches display observers and removes them on exit.

        renderer = MarkdownRenderer()
        async with showing(renderer):
            flow.show(Section(title="Results"))
        print(renderer.output())

    Async context manager to match flow.observing().
    """

    def __init__(self, *observers: Callable[[Component], None]):
        self._observers = list(observers)

    async def __aenter__(self):
        _show_observers.extend(self._observers)
        return self

    async def __aexit__(self, *args):
        for obs in self._observers:
            try:
                _show_observers.remove(obs)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class MarkdownRenderer:
    """Collects show() emissions, produces a markdown document.

        renderer = MarkdownRenderer()
        flow.show_to(renderer)
        # ... run pipeline with flow.show() calls ...
        print(renderer.output())

    Progress components with the same group replace previous emissions.
    """

    def __init__(self):
        self._parts: list[str] = []
        self._progress: dict[str, int] = {}  # group -> index in _parts

    def __call__(self, component: Component):
        if isinstance(component, Progress):
            if component.group in self._progress:
                idx = self._progress[component.group]
                self._parts[idx] = component.render_markdown()
            else:
                self._progress[component.group] = len(self._parts)
                self._parts.append(component.render_markdown())
        else:
            self._parts.append(component.render_markdown())

    def output(self) -> str:
        """Return the complete markdown document."""
        return "\n\n".join(p for p in self._parts if p)

    def reset(self):
        """Clear for reuse."""
        self._parts.clear()
        self._progress.clear()
