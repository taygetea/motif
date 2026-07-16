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
# narrate — fold the computation graph into a markdown narrative
# ---------------------------------------------------------------------------
#
# The default display: a homomorphism from the graph to a document, the
# same move render() makes from Msg to API payloads. Salience is decided
# by what the node IS — its pattern kind, its declared role, its
# cardinality, whether it errored — with meta["show"] (the show= kwarg
# on flow patterns) as the author override. Pipelines get a readable
# document for free; flow.show() components remain the escape hatch for
# bespoke curation.
#
# Default policy:
#   hidden     compact (invisible by its own contract); llm_call records
#              under a parent that narrates its own output (the record
#              is redundant with the narrative — drill-in views read it)
#   collapsed  branch / best_of / cascade / tournament (topology, not
#              content — their outputs are already one-line summaries);
#              calls declared role:structure
#   shown      fan children, reduce, call, agent finals, blackboard
#              rounds, tree results; llm_call records at the root or
#              under output-less parents (bare verb loops — the record
#              is the only narrative there)
#   always     errors surface regardless of policy; a fan wider than
#              fan_limit collapses to a preview list (salience moves
#              from the nodes to the aggregate)


_COLLAPSED_KINDS = {"branch", "best_of", "cascade", "tournament"}


def _first_line(text: str, length: int = 160) -> str:
    if not text:
        return ""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            return line[: length - 1] + "…" if len(line) > length else line
    return text[: length - 1] + "…" if len(text) > length else text


def _node_policy(node, *, parent_narrates: bool = False) -> str:
    """shown | collapsed | hidden — meta['show'] wins, then errors,
    then kind/role defaults. parent_narrates means the parent node
    displays its own output, making an llm_call record beneath it
    redundant for the narrative (it stays in the graph for drill-in)."""
    override = node.meta.get("show")
    if override:
        return override
    if node.error:
        return "shown"
    if node.kind == "compact":
        return "hidden"
    if node.kind == "llm_call" and parent_narrates:
        return "hidden"
    if node.kind in _COLLAPSED_KINDS:
        return "collapsed"
    if (node.kind in ("call", "llm_call")
            and str(node.meta.get("model", "")) == "role:structure"):
        return "collapsed"
    return "shown"


def _collapsed_line(node) -> str:
    preview = _first_line(node.output)
    line = f"- **{node.title}**"
    if preview:
        line += f" — {preview}"
    if node.error:
        line += f" — ⚠ error: {_first_line(node.error)}"
    return line


def _agent_steps_log(node) -> str:
    """One line per agent step: which tools fired."""
    lines = []
    for step in node.children:
        if step.kind != "step":
            continue
        tools = [c.title for c in step.children if c.kind == "tool_call"]
        detail = ", ".join(tools) if tools else _first_line(step.output, 80)
        lines.append(f"- {step.title}: {detail}" if detail
                     else f"- {step.title}")
    return "\n".join(lines)


def narrate(nodes, *, level: int = 2, fan_limit: int = 12) -> str:
    """Fold graph nodes into a markdown narrative.

        with graph.session() as s:
            await pipeline()
        doc = narrate(s.roots)

    level is the heading depth for top-level nodes. fan_limit is the
    cardinality above which a fan's children collapse to a preview list.
    """
    parts = [p for n in nodes for p in _narrate_node(n, level, fan_limit)]
    return "\n\n".join(p for p in parts if p)


def _heading(level: int, title: str) -> str:
    return f"{'#' * min(level, 6)} {title}"


def _demote_headings(text: str, below: int) -> str:
    """Shift markdown headings in model-generated output so they nest
    under a section at `below` — an embedded '# Title' inside an h2
    section becomes '### Title'. Fenced code blocks are left alone."""
    lines = text.split("\n")
    in_fence = False
    top = None
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            if line[depth : depth + 1] == " ":
                top = depth if top is None else min(top, depth)
    if top is None:
        return text
    shift = (below + 1) - top
    if shift <= 0:
        return text

    out = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif not in_fence and line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            if line[depth : depth + 1] == " ":
                out.append("#" * min(depth + shift, 6) + line[depth:])
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def _narrate_node(node, level: int, fan_limit: int,
                  parent_narrates: bool = False) -> list[str]:
    policy = _node_policy(node, parent_narrates=parent_narrates)

    if policy == "hidden":
        if node.error:  # errors surface even from hidden nodes
            return [f"> ⚠ error in {node.title}: {_first_line(node.error)}"]
        return []

    if policy == "collapsed":
        return [_collapsed_line(node)]

    parts: list[str] = []
    if node.error:
        parts.append(f"> ⚠ error in {node.title}: {_first_line(node.error)}")

    match node.kind:
        case "fan":
            parts.append(_heading(level, node.title))
            if len(node.children) > fan_limit:
                parts.append(f"{len(node.children)} calls (collapsed):")
                parts.append("\n".join(_collapsed_line(c) for c in node.children))
            else:
                for child in node.children:
                    parts.extend(_narrate_node(child, level + 1, fan_limit))

        case "agent":
            parts.append(_heading(level, node.title))
            if node.output:
                parts.append(_demote_headings(node.output, level))
            steps = _agent_steps_log(node)
            if steps:
                parts.append(f"*Activity:*\n{steps}")

        case "blackboard":
            parts.append(_heading(level, node.title))
            for rnd in node.children:
                if rnd.kind == "round":
                    parts.append(_heading(level + 1, rnd.title))
                    for turn in rnd.children:
                        if turn.output:
                            parts.append(f"**{turn.title}**: {turn.output}")
                elif rnd.output:
                    parts.append(f"**{rnd.title}**: {rnd.output}")

        case "tree":
            parts.append(_heading(level, node.title))
            if node.output:
                parts.append(_demote_headings(node.output, level))
            # Decomposition lists the subtree structure, not the call
            # records tree's own split/leaf/merge calls leave behind.
            sub = [_collapsed_line(c) for c in node.children
                   if c.kind == "tree"]
            if sub:
                parts.append("*Decomposition:*\n" + "\n".join(sub))

        case _:
            # call, llm_call, item, reduce, round, step, and anything
            # future: title as a section, own output as content, visible
            # children recursed. A node that displays its own output
            # narrates for its children — their call records go quiet.
            parts.append(_heading(level, node.title))
            if node.output:
                parts.append(_demote_headings(node.output, level))
            for child in node.children:
                parts.extend(_narrate_node(child, level + 1, fan_limit,
                                           parent_narrates=bool(node.output)))

    return parts


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
