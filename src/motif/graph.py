"""Computation graph — the structural record of a pipeline execution.

Flow patterns build this automatically via contextvar. Each Node
represents a flow operation (branch, fan, reduce, etc.) or an LLM
call. The graph assembles from Python's execution nesting:

    async def pipeline():
        items = await flow.branch(msg, title="discover", schema=S)
        results = await flow.fan(items, fn, title="analyze")
        # branch and fan are sibling nodes under root

When fan() creates parallel tasks, each task inherits the context
with the fan node as current — child nodes attach to the fan
automatically. No explicit wiring.

The graph persists after execution. Renderers read it:
  - TUI polls node.output growth for live streaming display
  - Trace.graph gives the complete structural record
  - MarkdownRenderer uses flow.show() components (separate layer)

Two display layers:
  - Graph nodes: the complete record (every chunk, every state change)
  - flow.show(): the curated narrative (what the author chose to emit)
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Node:
    """A node in the computation graph.

    State machine:
        pending → running → streaming → complete
           │         │          │
           └─────────┴──────────┴──→ error

    output grows during streaming (llm.stream() appends chunks).
    _version bumps on every mutation — renderers dirty-check this
    instead of subscribing to events.
    """
    id: str
    kind: str          # "branch", "fan", "reduce", "call", "agent", "step", etc.
    title: str
    children: list[Node] = field(default_factory=list)
    state: str = "pending"
    output: str = ""
    elapsed: float = 0.0
    meta: dict = field(default_factory=dict)
    error: str | None = None
    _version: int = 0
    _start_time: float = 0.0

    def _bump(self):
        self._version += 1

    def append_output(self, text: str):
        """Append streaming text. Sets state to 'streaming' on first chunk."""
        if self.state == "running":
            self.state = "streaming"
        self.output += text
        self._bump()

    def to_dict(self) -> dict:
        """Serialize for JSON. Recursive."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "state": self.state,
            "output": self.output[:200] + "..." if len(self.output) > 200 else self.output,
            "elapsed": round(self.elapsed, 2),
            "meta": self.meta,
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }


# The contextvar — each asyncio task gets its own copy.
# When fan() creates tasks via TaskGroup, each task inherits the
# fan node as _current_node. Child enter_node() calls attach to it.
_current_node: ContextVar[Node | None] = ContextVar('_current_node', default=None)

# Top-level nodes (those with no parent in the graph).
_root_nodes: list[Node] = []


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def enter_node(kind: str, title: str, **meta) -> tuple[Node, Node | None]:
    """Create a node, attach to parent, set as current.

    Returns (new_node, previous_node). Caller must pass previous_node
    to exit_node() to restore context — even on error.

    Usage in flow patterns:
        node, parent = enter_node("fan", title, model=model)
        try:
            # ... do work ...
            exit_node(node, parent)
        except Exception as e:
            exit_node(node, parent, error=str(e))
            raise
    """
    parent = _current_node.get(None)
    node = Node(
        id=_new_id(),
        kind=kind,
        title=title,
        state="running",
        meta=meta,
        _start_time=time.monotonic(),
    )
    if parent:
        parent.children.append(node)
    else:
        _root_nodes.append(node)
    _current_node.set(node)
    return node, parent


def exit_node(node: Node, parent: Node | None, *, error: str | None = None):
    """Mark node complete or error, restore parent as current."""
    node.elapsed = time.monotonic() - node._start_time
    if error:
        node.state = "error"
        node.error = error
    elif node.state != "complete":
        # Don't override if already marked complete (e.g., by streaming finish)
        node.state = "complete"
    node._bump()
    _current_node.set(parent)


def current_node() -> Node | None:
    """Get the current graph node. Used by llm.stream() to write chunks."""
    return _current_node.get(None)


def root_nodes() -> list[Node]:
    """Snapshot of top-level nodes."""
    return list(_root_nodes)


def reset():
    """Clear the graph. For testing and between pipeline runs."""
    _root_nodes.clear()
    _current_node.set(None)
