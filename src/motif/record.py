"""Graph projection of the call-lifecycle seam.

llm.py emits pure facts (CallStarted / CallChunk / CallCompleted /
CallFailed) and knows nothing about the computation graph. This module
folds those facts into graph nodes: every verb invocation becomes an
"llm_call" node under whatever node is current in the caller's context —
a flow pattern's node, a group, or the session root for bare verbs. Raw
loops over the verbs trace as well as the blessed patterns (roadmap
principle 6), and multi-call patterns get per-call identity: best_of's
five judgments are five nodes, not one.

The record node's id IS the call_id — one identity, no linking table.
It retains the input Msg (Node.msg) and stamps usage into meta; the
loom, lineage matching, and cost views read from here.

The projection is fixed infrastructure, installed at import (see
__init__.py) via the llm._projection slot — deliberately not an entry
in the user observer registry, so llm.clear_observers() cannot silence
the graph. Exactly one projection exists; a second would double-record
every call.
"""

from __future__ import annotations

import json
import time

from . import llm
from .graph import Node, attach, _register_scoped

# Calls in flight: call_id -> Node. Single-threaded asyncio access;
# entries are removed when the call settles, so this cannot grow.
_open: dict[str, Node] = {}


def _label(declared) -> str:
    """Display label for the declared model: RoleRefs become
    "role:<name>" (the salience policy keys on this), Endpoints their
    model id, strings themselves."""
    if isinstance(declared, llm.RoleRef):
        return f"role:{declared.name}"
    if isinstance(declared, llm.Endpoint):
        return declared.model
    return str(declared)


def _render_result(result) -> str:
    """A call node's output is the full result in textual form — the
    graph is the complete record, not a preview."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    # ActResult and future shapes: text plus what was invoked
    text = getattr(result, "text", None)
    calls = getattr(result, "tool_calls", None)
    if calls:
        names = ", ".join(c.name for c in calls)
        return f"{text or ''}\n[tool calls: {names}]".strip()
    return str(text if text is not None else result)


def project(event) -> None:
    """Fold one call-lifecycle event into the graph."""
    if isinstance(event, llm.CallStarted):
        node = Node(
            id=event.call_id,
            kind="llm_call",
            title=event.verb,
            state="running",
            msg=event.msg,
            meta={"model": _label(event.declared),
                  "endpoint": event.endpoint.model,
                  **({"node": event.meta["node"]}
                     if "node" in event.meta else {})},
            _start_time=time.monotonic(),
        )
        attach(node)
        _open[event.call_id] = node

    elif isinstance(event, llm.CallChunk):
        node = _open.get(event.call_id)
        if node is not None:
            node.append_output(event.text)

    elif isinstance(event, llm.CallCompleted):
        node = _open.pop(event.call_id, None)
        if node is None:
            return
        if not node.output:  # streamed calls already carry their chunks
            node.output = _render_result(event.result)
        node.meta["usage"] = event.usage
        if event.stop_reason is not None:
            node.meta["stop_reason"] = event.stop_reason
        node.elapsed = time.monotonic() - node._start_time
        node.state = "complete"
        node._bump()

    elif isinstance(event, llm.CallFailed):
        node = _open.pop(event.call_id, None)
        if node is None:
            return
        if event.usage:
            node.meta["usage"] = event.usage
        node.elapsed = time.monotonic() - node._start_time
        node.state = "error"
        node.error = event.error
        node._bump()


# Install: llm.py exposes a single projection slot instead of importing
# the graph — the dependency points upward, injected here. The same
# inversion wires llm's observer scope into graph.session (flow and
# show register their own; llm cannot without importing graph).
llm._projection = project
_register_scoped(llm._session_call_observers)
