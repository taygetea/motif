"""Flow combinators for prompt primitives.

prompt.py handles what you say to one LLM call.
flow.py handles how multiple calls relate to each other.

Nine patterns — eight with predetermined topology, one that generates
topology at runtime:

    branch     — one call discovers structure → list of items
    fan        — items → parallel calls → results
    reduce     — results → synthesis call → one result
    best_of    — results → parallel judging → best one
    cascade    — try cheap model, escalate if needed
    tree       — recursive decomposition and reassembly
    tournament — bracket-style elimination
    blackboard — expert panel with shared state across rounds
    agent      — tool-use loop: Msg grows until a signal tool fires

All functions build a computation graph via contextvar (see graph.py)
and emit FlowEvents to observers for backward compatibility.
"""

from __future__ import annotations

import asyncio
import copy
import time
import warnings
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Any

from .prompt import (
    Msg, Block, TextSegment, ToolCall, ToolResult,
    system, user, assistant, tool_use, tool_result,
)
from . import llm
from .graph import (
    enter_node, exit_node, current_node, Node, _new_id, _register_scoped,
)

# Re-export show machinery so users can do flow.show(), flow.showing(), etc.
from .show import show, show_to, showing, clear_show_observers, narrate

# Structural decisions (branching, judging, routing, splitting) default to
# llm.role("structure"); content generation to llm.role("content"). RoleRefs
# are lazy — the verbs resolve them at call time against the current profile
# (llm.use_profile), so which model fills each role is a deployment concern.
# The role split itself is intentional: topology decisions and content
# generation are different kinds of work, whatever they happen to cost.
_STRUCTURE = llm.role("structure")
_CONTENT = llm.role("content")


# ---------------------------------------------------------------------------
# Events — backward-compatible topology notifications
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FlowEvent:
    """What happened in the computation graph."""
    kind: str        # "start", "complete", "split", "merge", "error"
    label: str       # human-readable node name
    depth: int = 0
    result: str | None = None      # truncated output preview
    children: list[str] | None = None
    elapsed: float = 0.0
    meta: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


_observers: list[Callable[[FlowEvent], None]] = []
_session_observers: ContextVar[list | None] = ContextVar(
    "motif_session_flow_observers", default=None)
_register_scoped(_session_observers)


def _active_observers() -> list:
    scope = _session_observers.get()
    return scope if scope is not None else _observers


def observe(*observers: Callable[[FlowEvent], None]):
    """Attach observers that receive every flow event. Inside a
    graph.session(), scoped to the session."""
    _active_observers().extend(observers)


class observing:
    """Context manager that attaches observers and removes them on exit.

        async with flow.observing(trace, display):
            result = await flow.fan(items, fn, title="analyze")
        # observers automatically removed — no manual clear_observers()
    """

    def __init__(self, *observers: Callable[[FlowEvent], None]):
        self._observers = list(observers)
        self._target: list | None = None

    async def __aenter__(self):
        self._target = _active_observers()
        self._target.extend(self._observers)
        return self

    async def __aexit__(self, *args):
        for obs in self._observers:
            try:
                self._target.remove(obs)
            except ValueError:
                pass


def clear_observers():
    """Remove the active scope's flow observers."""
    _active_observers().clear()


def _emit(event: FlowEvent):
    scope = _session_observers.get()
    for observers in (_observers, scope or ()):
        for obs in observers:
            try:
                obs(event)
            except Exception:
                pass


def _truncate(text: str, length: int = 120) -> str:
    """Truncate to first meaningful line for display previews."""
    if not text:
        return ""
    for line in text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            return line[:length - 3] + "..." if len(line) > length else line
    return text[:length - 3] + "..." if len(text) > length else text


def _item_label(item: Any, idx: int, key: str | None = None) -> str:
    """Extract a display label from a list item.

    If `key` is given, prefer that field. Otherwise fall back to common
    label-shaped keys ("name", "label", "title") before defaulting to
    a positional placeholder.
    """
    if isinstance(item, dict):
        if key is not None and key in item:
            return str(item[key])[:80]
        for fallback in ("name", "label", "title"):
            if fallback in item:
                return str(item[fallback])[:80]
        return f"item_{idx}"
    return f"item_{idx}"


def _join(texts: list[str], labels: list[str] | None = None) -> str:
    """Join results for convergence steps. Delegates to Block.join."""
    return Block.join(texts, labels=labels)


def _estimate_tokens(msg: Msg) -> int:
    """Rough token estimate. chars/4 is the standard heuristic."""
    total = 0
    for seg in msg.segments:
        if isinstance(seg, TextSegment):
            total += len(seg.text)
        elif isinstance(seg, ToolCall):
            total += len(seg.name) + len(str(seg.input))
        elif isinstance(seg, ToolResult):
            total += len(seg.content)
    return total // 4


def _model_label(model) -> str:
    """Human/JSON-safe label for a model param: RoleRefs become
    "role:<name>" (the salience policy reads this), Endpoints their
    model id, strings themselves."""
    if isinstance(model, llm.RoleRef):
        return f"role:{model.name}"
    if isinstance(model, llm.Endpoint):
        return model.model
    return str(model)


_SHOW_VALUES = ("shown", "collapsed", "hidden")


def _show_meta(show: str | None) -> dict:
    """Validate an author display override for node meta."""
    if show is None:
        return {}
    if show not in _SHOW_VALUES:
        raise ValueError(f"show= must be one of {_SHOW_VALUES}, got {show!r}")
    return {"show": show}


def _check_label_kwarg(kw: dict):
    """Catch old label= usage in **kw before it silently passes through."""
    if "label" in kw:
        raise TypeError(
            "Use title= instead of label= (renamed in motif 0.2). "
            "title is now a required keyword argument.")


def _branch_items_key(schema: dict, items_key: str | None) -> str:
    """Resolve the top-level array property branch() should fan over."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(
            "branch() schema must define top-level properties containing an "
            "array. Add schema['properties'] = "
            "{'items': {'type': 'array', 'items': {...}}}.")

    array_keys = [
        key for key, property_schema in properties.items()
        if isinstance(property_schema, dict)
        and property_schema.get("type") == "array"
    ]

    if items_key is not None:
        if items_key not in array_keys:
            raise ValueError(
                f"branch() items_key={items_key!r} must name a top-level "
                "array property in the schema. Declare "
                f"schema['properties'][{items_key!r}] with "
                "{'type': 'array', 'items': {...}}.")
        return items_key

    if not array_keys:
        raise ValueError(
            "branch() schema has no top-level array property. Add one under "
            "schema['properties'], for example "
            "'items': {'type': 'array', 'items': {...}}.")
    if len(array_keys) > 1:
        choices = " or ".join(f"items_key={key!r}" for key in array_keys)
        names = ", ".join(repr(key) for key in array_keys)
        raise ValueError(
            f"branch() schema has multiple top-level array properties: "
            f"{names}. Pass {choices} to choose which one to fan over.")
    return array_keys[0]


def _check_paragraph_partition(subtasks: list[dict], paragraph_count: int):
    """Require ordered half-open ranges covering every paragraph once."""
    expected_start = 0
    contract = (
        f"Return contiguous, non-overlapping ranges covering all "
        f"{paragraph_count} paragraphs: the first start_paragraph must be 0, "
        "each later start_paragraph must equal the previous end_paragraph, "
        f"and the final end_paragraph must be {paragraph_count}. "
        "end_paragraph is exclusive."
    )

    for i, subtask in enumerate(subtasks):
        if not isinstance(subtask, dict):
            raise ValueError(
                f"Invalid tree partition: subtask {i} must be an object. "
                f"{contract}")
        if "start_paragraph" not in subtask or "end_paragraph" not in subtask:
            raise ValueError(
                f"Invalid tree partition: subtask {i} must include both "
                f"start_paragraph and end_paragraph. {contract}")

        start = subtask["start_paragraph"]
        end = subtask["end_paragraph"]
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)):
            raise ValueError(
                f"Invalid tree partition: subtask {i} has range "
                f"[{start!r}, {end!r}); both bounds must be integers. "
                f"{contract}")
        if (start < 0 or start > paragraph_count
                or end < 0 or end > paragraph_count):
            raise ValueError(
                f"Invalid tree partition: subtask {i} range [{start}, {end}) "
                f"is outside [0, {paragraph_count}). {contract}")
        if start >= end:
            raise ValueError(
                f"Invalid tree partition: subtask {i} range [{start}, {end}) "
                f"must be non-empty with start_paragraph < end_paragraph. "
                f"{contract}")
        if start != expected_start:
            problem = "gap" if start > expected_start else "overlap or reordering"
            raise ValueError(
                f"Invalid tree partition: subtask {i} starts at {start}, "
                f"expected {expected_start} ({problem}). {contract}")
        expected_start = end

    if expected_start != paragraph_count:
        raise ValueError(
            f"Invalid tree partition: final end_paragraph is {expected_start}, "
            f"expected {paragraph_count} (trailing gap). {contract}")


# ---------------------------------------------------------------------------
# Compaction — keep Msgs within token limits transparently
# ---------------------------------------------------------------------------

COMPACT_PROMPT = system("""Summarize this conversation history concisely. Preserve:
- Key decisions and their reasoning
- Tool call results and what was learned
- The current state of the task
- Any commitments or plans mentioned

Write as a neutral record, not as a participant. This summary will
replace the conversation history in an ongoing exchange.""", cache=True)


def _compact_split(rest: list, keep_recent: int) -> int:
    """Index splitting rest into (compactable prefix, kept tail) such that
    no tool_use/tool_result pair straddles the boundary.

    A pair can straddle with unrelated segments in between, so each round
    scans the entire compactable prefix — not just the segment adjacent to
    the split — and moves the boundary back to the earliest straddler.
    Pulling segments into the tail can bring new tool ids with them, so
    repeat until stable. Terminates: split_at strictly decreases.
    """
    split_at = len(rest) - keep_recent

    while split_at > 0:
        tail_tool_result_ids = set()
        tail_tool_use_ids = set()
        for seg in rest[split_at:]:
            if isinstance(seg, ToolResult):
                tail_tool_result_ids.add(seg.tool_use_id)
            elif isinstance(seg, ToolCall):
                tail_tool_use_ids.add(seg.id)

        straddler = None
        for i in range(split_at):
            seg = rest[i]
            if isinstance(seg, ToolCall) and seg.id in tail_tool_result_ids:
                straddler = i
                break
            if isinstance(seg, ToolResult) and seg.tool_use_id in tail_tool_use_ids:
                straddler = i
                break

        if straddler is None:
            break
        split_at = straddler

    return max(split_at, 0)


async def compact(
    msg: Msg,
    *,
    max_tokens: int = 100_000,
    keep_recent: int = 6,
    model: str | llm.Endpoint | llm.RoleRef = _CONTENT,
) -> Msg:
    """Compact a Msg if it exceeds the token threshold.

    Preserves system segments (persona, instructions) and the most
    recent turns. Summarizes everything in between into a single
    user segment. Returns the original Msg unchanged if under threshold.

    Referential integrity: tool_use/tool_result pairs are never split.
    The boundary walks backward to keep all pairs intact.

    Called automatically by agent() — users don't need to call this directly.
    """
    est = _estimate_tokens(msg)
    if est <= max_tokens:
        return msg

    segments = list(msg.segments)

    # Split into: system prefix, middle (compactable), recent tail
    system_segs = []
    rest = []
    for seg in segments:
        if isinstance(seg, TextSegment) and seg.role == "system":
            system_segs.append(seg)
        else:
            rest.append(seg)

    if len(rest) <= keep_recent:
        return msg  # not enough to compact

    split_at = _compact_split(rest, keep_recent)

    to_compact = rest[:split_at]
    to_keep = rest[split_at:]

    if not to_compact:
        return msg  # nothing safe to compact

    lines = []
    for seg in to_compact:
        match seg:
            case TextSegment(role=role, text=text):
                lines.append(f"[{role}]: {text}")
            case ToolCall(name=name, input=inp):
                lines.append(f"[tool_use: {name}]: {str(inp)[:500]}")
            case ToolResult(content=content):
                lines.append(f"[tool_result]: {content[:500]}")

    history_text = "\n\n".join(lines)

    node, parent = enter_node("compact", "compact",
                              tokens_before=est, segments=len(to_compact))
    _emit(FlowEvent("start", "compact", 0,
                     meta={"tokens_before": est, "segments_compacted": len(to_compact)}))

    try:
        summary = await llm.complete(
            COMPACT_PROMPT | user(history_text),
            model=model,
        )

        summary_seg = TextSegment("user", f"[Prior conversation summary]\n{summary}")
        new_msg = Msg(segments=tuple(system_segs + [summary_seg] + to_keep))
        new_est = _estimate_tokens(new_msg)

        node.output = f"{est} → {new_est} tokens (est)"
        exit_node(node, parent)
        _emit(FlowEvent("complete", "compact", 0, elapsed=node.elapsed,
                         result=node.output,
                         meta={"tokens_after": new_est}))
        return new_msg
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


# ---------------------------------------------------------------------------
# Single call — wrap one llm.complete/extract as a flow node
# ---------------------------------------------------------------------------

async def call(
    msg: Msg,
    *,
    title: str,
    show: str | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    schema: dict | None = None,
    depth: int = 0,
    **kw,
) -> Any:
    """An author annotation around one LLM call — a title and display
    override for a call that is recorded automatically anyway.

    Every bare llm.complete()/llm.extract() already appears in the graph
    (the call-lifecycle projection records it), so call() is not needed
    for visibility. Use it to name the call's place in the pipeline and
    control its salience. Returns text if no schema, dict if a schema is
    given.

        report = await flow.call(SYNTHESIZER | user(material), title="synthesis")

        plan = await flow.call(PLANNER | user(topic),
                               schema=PLAN_SCHEMA, title="planning")

    The one remaining display gap for bare calls is legacy FlowEvents
    (the live Trace/LiveFlowDisplay layer) — call() still emits those.
    """
    _check_label_kwarg(kw)
    node, parent = enter_node("call", title, model=_model_label(model), **_show_meta(show))
    _emit(FlowEvent("start", title, depth, meta={"model": model}))

    try:
        if schema is not None:
            result = await llm.extract(msg, schema=schema, model=model, **kw)
            # The annotation node stays silent: the llm_call record
            # beneath it holds the full JSON and narrates it. Setting a
            # lossy preview here would replace the real data in the
            # document (the preview still feeds the live FlowEvent).
            preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in result.items())
        else:
            result = await llm.complete(msg, model=model, **kw)
            node.output = result
            preview = result

        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         result=_truncate(preview)))
        return result
    except BaseException as e:
        exit_node(node, parent, error=llm._error_text(e))
        _emit(FlowEvent("error", title, depth, result=llm._error_text(e)))
        raise


# ---------------------------------------------------------------------------
# Grouping — a titled node with no LLM call of its own
# ---------------------------------------------------------------------------

class group:
    """Group work under a titled node — turns, phases, authoring
    sections. The group makes no LLM call; bare verbs inside it record
    beneath it, and narrate renders those records as the group's
    content (an output-less parent doesn't narrate for its children).

        with flow.group("turn 1"):
            thought = await llm.complete(persona | user(state))
            speech = await llm.complete(persona | user(thought))

    Works as `with` or `async with` (entering a group is context
    bookkeeping, not I/O — both forms do the same thing). An exception
    inside marks the group node errored and propagates.
    """

    def __init__(self, title: str, *, show: str | None = None):
        self._title = title
        self._show = _show_meta(show)
        self.node: Node | None = None
        self._parent: Node | None = None

    def __enter__(self) -> Node:
        self.node, self._parent = enter_node("group", self._title,
                                             **self._show)
        _emit(FlowEvent("start", self._title, 0))
        return self.node

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            exit_node(self.node, self._parent, error=llm._error_text(exc))
            _emit(FlowEvent("error", self._title, 0,
                             result=llm._error_text(exc)))
        else:
            exit_node(self.node, self._parent)
            _emit(FlowEvent("complete", self._title, 0,
                             elapsed=self.node.elapsed))
        return False

    async def __aenter__(self) -> Node:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)


# ---------------------------------------------------------------------------
# Branching — one becomes many
# ---------------------------------------------------------------------------

async def branch(
    msg: Msg,
    schema: dict,
    *,
    title: str,
    show: str | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _STRUCTURE,
    items_key: str | None = None,
    label_key: str | None = None,
    depth: int = 0,
    **kw,
) -> list[dict]:
    """One call discovers structure. Returns a list of items.

    The schema must produce an object with one top-level array field. If it
    has multiple top-level arrays, `items_key` explicitly selects the one to
    return.

    `label_key` picks which field of each item to use as its display
    label in the live tree. If omitted, falls back to "name"/"label"/"title"
    or a positional placeholder.

        methods = await branch(
            system("List methodologies...") | user(doc),
            title="discover angles",
            schema=METHODS_SCHEMA,
            label_key="approach",   # use item["approach"] as the label
        )
    """
    _check_label_kwarg(kw)
    resolved_items_key = _branch_items_key(schema, items_key)
    node, parent = enter_node("branch", title, model=_model_label(model), **_show_meta(show))
    _emit(FlowEvent("start", title, depth, meta={"model": model}))

    try:
        result = await llm.extract(msg, schema=schema, model=model, **kw)

        if (not isinstance(result, dict)
                or not isinstance(result.get(resolved_items_key), list)):
            raise ValueError(
                f"branch() expected extraction result field "
                f"{resolved_items_key!r} to be a list because the schema "
                "declares it as an array. Ensure the structured-output "
                "response conforms to the schema.")
        items = result[resolved_items_key]

        child_labels = [_item_label(item, i, key=label_key) for i, item in enumerate(items)]
        node.output = ", ".join(child_labels)
        exit_node(node, parent)
        # leaf_children=True tells display observers these "children" are
        # output values, not subtasks — they shouldn't render as pending.
        _emit(FlowEvent("split", title, depth, children=child_labels,
                         elapsed=node.elapsed,
                         meta={"count": len(items), "model": model,
                               "leaf_children": True}))
        return items
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


async def fan(
    items: list,
    fn: Callable[[Any], Msg],
    *,
    title: str,
    show: str | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    max_concurrency: int | None = None,
    streaming: bool = False,
    depth: int = 0,
    **kw,
) -> list[str]:
    """Parallel complete() over items. fn maps each item to a Msg.

    max_concurrency limits how many calls run simultaneously.
    streaming=True emits per-chunk notifications for live display.

        analyses = await fan(
            methods,
            lambda m: analyst | user(f"Use {m['name']}:\\n{doc}"),
            title="parallel analysis",
            max_concurrency=5,
            streaming=True,
        )
    """
    _check_label_kwarg(kw)
    child_labels = [_item_label(item, i) for i, item in enumerate(items)]
    node, parent = enter_node("fan", title, model=_model_label(model), count=len(items),
                              **_show_meta(show))
    _emit(FlowEvent("start", title, depth, meta={"count": len(items), "model": model}))
    _emit(FlowEvent("split", title, depth, children=child_labels,
                     meta={"count": len(items), "model": model}))
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _one(item, idx):
        # enter_node BEFORE semaphore so all children appear in the graph
        # immediately — TUI can build layout before work starts. The
        # child is an item slot (the unit of work), not a claim that a
        # call occurred — the call record attaches beneath it when the
        # verb actually runs.
        name = _item_label(item, idx)
        child, child_parent = enter_node("item", name, model=_model_label(model))
        _emit(FlowEvent("start", name, depth + 1, meta={"model": model}))

        # BaseException throughout: when a sibling fails, the TaskGroup
        # CANCELS this task — possibly while parked on the semaphore or
        # mid-call — and a cancelled item must settle its node instead
        # of staying "running" forever.
        try:
            if sem:
                await sem.acquire()
        except BaseException as e:
            exit_node(child, child_parent, error=llm._error_text(e))
            _emit(FlowEvent("error", name, depth + 1,
                             result=llm._error_text(e)))
            raise
        try:
            result = await llm.complete(
                fn(item), model=model, streaming=streaming,
                meta={"node": name}, **kw)
            child.output = result
            exit_node(child, child_parent)
            _emit(FlowEvent("complete", name, depth + 1,
                             result=_truncate(result), elapsed=child.elapsed))
            return result
        except BaseException as e:
            exit_node(child, child_parent, error=llm._error_text(e))
            _emit(FlowEvent("error", name, depth + 1,
                             result=llm._error_text(e)))
            raise
        finally:
            if sem:
                sem.release()

    try:
        # TaskGroup cancels remaining tasks if one fails (better than gather
        # for rate-limited APIs — don't fire 49 more into a 429)
        results: list = [None] * len(items)
        async with asyncio.TaskGroup() as tg:
            async def _run(i, item):
                results[i] = await _one(item, i)
            for i, item in enumerate(items):
                tg.create_task(_run(i, item))

        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         meta={"count": len(results)}))
        return results
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


# ---------------------------------------------------------------------------
# Converging — many become one
# ---------------------------------------------------------------------------

async def reduce(
    results: list[str],
    msg_fn: Callable[[str], Msg],
    *,
    title: str,
    show: str | None = None,
    labels: list[str] | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    depth: int = 0,
    **kw,
) -> str:
    """Many results converge into one. msg_fn receives the combined text.

    labels, if provided, wraps each result as [label]:\\n...

        synthesis = await reduce(
            analyses,
            lambda combined: synthesizer | user(combined),
            title="synthesis",
            labels=[m["name"] for m in methods],
        )
    """
    _check_label_kwarg(kw)
    node, parent = enter_node("reduce", title, model=_model_label(model),
                              inputs=len(results), **_show_meta(show))
    _emit(FlowEvent("start", title, depth, meta={"inputs": len(results), "model": model}))

    try:
        combined = _join(results, labels=labels)
        result = await llm.complete(msg_fn(combined), model=model,
                                     meta={"node": title}, **kw)

        node.output = result
        exit_node(node, parent)
        _emit(FlowEvent("merge", title, depth, result=_truncate(result),
                         elapsed=node.elapsed))
        return result
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


async def best_of(
    candidates: list[str],
    judge_fn: Callable[[str], Msg],
    judge_schema: dict,
    *,
    title: str,
    show: str | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _STRUCTURE,
    score_key: str = "score",
    depth: int = 0,
) -> tuple[str, int, list[dict]]:
    """Judge picks the best from N candidates.

    Returns (best, index, all_judgments). Judging happens in parallel.

        best, idx, scores = await best_of(
            drafts,
            lambda d: judge | user(f"Rate 1-10:\\n{d}"),
            title="select best",
            schema=SCORE_SCHEMA,
        )
    """
    if not candidates:
        raise ValueError("best_of() needs at least one candidate")
    node, parent = enter_node("best_of", title, model=_model_label(model),
                              candidates=len(candidates), **_show_meta(show))
    _emit(FlowEvent("start", title, depth,
                     meta={"candidates": len(candidates), "model": model}))

    try:
        # TaskGroup, not gather: when one judgment fails, the others are
        # cancelled instead of running (and billing) toward a verdict
        # that can no longer be delivered. Callers see ExceptionGroup.
        judgments: list = [None] * len(candidates)
        async with asyncio.TaskGroup() as tg:
            async def _judge(i, c):
                judgments[i] = await llm.extract(
                    judge_fn(c), schema=judge_schema, model=model)
            for i, c in enumerate(candidates):
                tg.create_task(_judge(i, c))

        best_idx = max(range(len(judgments)),
                       key=lambda i: judgments[i].get(score_key, 0))

        node.output = f"winner: #{best_idx} (score {judgments[best_idx].get(score_key)})"
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         result=node.output))
        return candidates[best_idx], best_idx, judgments
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


# ---------------------------------------------------------------------------
# Linear — cost optimization
# ---------------------------------------------------------------------------

async def cascade(
    msg: Msg,
    test_fn: Callable[[str], Msg],
    test_schema: dict,
    models: list[str],
    *,
    title: str,
    show: str | None = None,
    model_test: str | llm.Endpoint | llm.RoleRef = _STRUCTURE,
    depth: int = 0,
) -> tuple[str, str]:
    """Try cheap models first, escalate until quality passes.

    Returns (result, model_used). test_schema needs a "sufficient" boolean.

        answer, model_used = await cascade(
            system("Answer precisely.") | user(question),
            test_fn=lambda ans: checker | user(f"Is this correct?\\n{ans}"),
            test_schema=QUALITY_SCHEMA,
            title="cost cascade",
            models=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
        )
    """
    node, parent = enter_node("cascade", title, models=models, **_show_meta(show))
    _emit(FlowEvent("start", title, depth, meta={"models": models}))
    used_model = models[-1]  # fallback
    result = ""

    try:
        for used_model in models:
            child, child_parent = enter_node("call", used_model)
            _emit(FlowEvent("start", used_model, depth + 1))

            try:
                result = await llm.complete(msg, model=used_model)

                if used_model == models[-1]:  # last model — accept regardless
                    child.output = result
                    exit_node(child, child_parent)
                    _emit(FlowEvent("complete", used_model, depth + 1,
                                     result=_truncate(result),
                                     elapsed=child.elapsed))
                    break

                judgment = await llm.extract(test_fn(result), schema=test_schema,
                                             model=model_test)
            except BaseException as e:
                exit_node(child, child_parent, error=llm._error_text(e))
                raise

            if judgment.get("sufficient", False):
                child.output = result
                exit_node(child, child_parent)
                _emit(FlowEvent("complete", used_model, depth + 1,
                                 result=_truncate(result), elapsed=child.elapsed))
                break
            else:
                child.output = "insufficient"
                exit_node(child, child_parent)
                _emit(FlowEvent("complete", used_model, depth + 1,
                                 result="insufficient — escalating",
                                 elapsed=child.elapsed))

        node.output = f"settled on {used_model}"
        node.meta["model_used"] = used_model
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         result=node.output, meta={"model_used": used_model}))
        return result, used_model
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


# ---------------------------------------------------------------------------
# Composite — structured recursion and interaction
# ---------------------------------------------------------------------------

async def tree(
    task: str,
    split_fn: Callable[[str], Msg],
    split_schema: dict,
    leaf_fn: Callable[[str], Msg],
    merge_fn: Callable[[list[str], list[str]], Msg],
    *,
    title: str,
    show: str | None = None,
    paragraph_fn: Callable[[str], list[str]] | None = None,
    max_depth: int = 3,
    model_split: str | llm.Endpoint | llm.RoleRef = _STRUCTURE,
    model_leaf: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    model_merge: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    _depth: int = 0,
) -> str:
    """Recursive decomposition. Split until leaves, work leaves, merge up.

    The splitter returns an ordered partition of paragraph ranges, not
    reproduced text. Ranges are zero-based and half-open: start_paragraph
    is inclusive and end_paragraph is exclusive. They must be non-empty,
    contiguous, non-overlapping, and cover every paragraph exactly once.
    The original text is sliced by the framework — no JSON reproduction
    of large documents.

    paragraph_fn splits text into indexable chunks. Defaults to
    \\n\\n splitting. Override for texts where \\n\\n doesn't align
    with logical boundaries (e.g., code blocks, quoted passages).

    split_fn(task) -> Msg asking the model to split or analyze as-is.
    split_schema must have:
        is_leaf: bool
        subtasks: [{label: str, start_paragraph: int, end_paragraph: int}]
    leaf_fn(task) -> Msg for leaf-level work.
    merge_fn(results, labels) -> Msg for combining child results.

        result = await tree(
            task=long_document,
            split_fn=lambda t: splitter | user(t),
            split_schema=SPLIT_SCHEMA,
            leaf_fn=lambda t: worker | user(t),
            merge_fn=lambda rs, ls: combiner | user(Block.join(rs, labels=ls)),
            title="decompose document",
        )
    """
    node, parent = enter_node("tree", title, chars=len(task), **_show_meta(show))
    _emit(FlowEvent("start", title, _depth, meta={"chars": len(task)}))

    try:
        # At max depth, force leaf
        if _depth >= max_depth:
            result = await llm.complete(leaf_fn(task), model=model_leaf)
            node.output = result
            exit_node(node, parent)
            _emit(FlowEvent("complete", title, _depth,
                             result=_truncate(result), elapsed=node.elapsed))
            return result

        # Give the range-producing model the exact partition contract.
        _split = paragraph_fn or (lambda t: t.split("\n\n"))
        paragraphs = _split(task)
        partition_instruction = system(
            f"This text has {len(paragraphs)} paragraphs. If is_leaf is false, "
            "subtasks must partition all of them in their original order using "
            "zero-based half-open ranges [start_paragraph, end_paragraph): "
            "start_paragraph is inclusive and end_paragraph is exclusive. "
            "The first range must start at 0, each range must be non-empty, "
            "each later range must start at the previous range's end, and the "
            f"final range must end at {len(paragraphs)}. Do not return copied text."
        )
        decision = await llm.extract(
            split_fn(task) | partition_instruction,
            schema=split_schema,
            model=model_split,
        )

        if decision.get("is_leaf", True):
            result = await llm.complete(leaf_fn(task), model=model_leaf)
            node.output = result
            exit_node(node, parent)
            _emit(FlowEvent("complete", title, _depth,
                             result=_truncate(result), elapsed=node.elapsed))
            return result

        subtasks = decision.get("subtasks", [])
        if not subtasks or len(subtasks) < 2:
            result = await llm.complete(leaf_fn(task), model=model_leaf)
            node.output = result
            exit_node(node, parent)
            _emit(FlowEvent("complete", title, _depth,
                             result=_truncate(result), elapsed=node.elapsed))
            return result

        # Validate before slicing: Python accepts negative, reversed, and
        # overlapping slices, so raw slicing cannot enforce the partition.
        _check_paragraph_partition(subtasks, len(paragraphs))
        child_labels = []
        child_texts = []

        for s in subtasks:
            clabel = s.get("label", s.get("name", f"part_{len(child_labels)}"))
            child_labels.append(clabel)
            child_texts.append("\n\n".join(
                paragraphs[s["start_paragraph"]:s["end_paragraph"]]))

        _emit(FlowEvent("split", title, _depth, children=child_labels,
                         elapsed=time.monotonic() - node._start_time))

        # Recurse in parallel — each recursive call creates its own graph
        # node. TaskGroup: a failing subtree cancels its siblings.
        child_results: list = [None] * len(child_labels)
        async with asyncio.TaskGroup() as tg:
            async def _subtree(i, clabel, text):
                child_results[i] = await tree(
                    text, split_fn, split_schema, leaf_fn, merge_fn,
                    paragraph_fn=paragraph_fn, max_depth=max_depth,
                    model_split=model_split, model_leaf=model_leaf,
                    model_merge=model_merge, title=clabel, _depth=_depth + 1,
                )
            for i, (clabel, text) in enumerate(zip(child_labels, child_texts)):
                tg.create_task(_subtree(i, clabel, text))

        # Merge — pass labeled results to merge_fn
        merged = await llm.complete(
            merge_fn(child_results, child_labels), model=model_merge)
        node.output = merged
        exit_node(node, parent)
        _emit(FlowEvent("merge", title, _depth,
                         result=_truncate(merged), elapsed=node.elapsed))
        return merged
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


async def tournament(
    candidates: list[str],
    judge_fn: Callable[[str, str], Msg],
    judge_schema: dict,
    *,
    title: str,
    show: str | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _STRUCTURE,
    winner_key: str = "winner",
    depth: int = 0,
) -> tuple[str, int, list]:
    """Bracket-style elimination. judge_fn(a, b) -> Msg.

    judge_schema must have a field (winner_key) valued "a" or "b".
    Returns (winner_text, original_index, rounds_log).

        winner, idx, log = await tournament(
            drafts,
            lambda a, b: judge | user(f"Which is better?\\nA: {a}\\nB: {b}"),
            title="bracket",
            schema=WINNER_SCHEMA,
        )
    """
    if not candidates:
        raise ValueError("Need at least one candidate")
    if len(candidates) == 1:
        return candidates[0], 0, []

    node, parent = enter_node("tournament", title, model=_model_label(model),
                              candidates=len(candidates), **_show_meta(show))
    _emit(FlowEvent("start", title, depth,
                     meta={"candidates": len(candidates), "model": model}))

    try:
        active = list(enumerate(candidates))
        rounds_log = []
        round_num = 0

        while len(active) > 1:
            round_num += 1
            next_round = []
            pairs = []

            for i in range(0, len(active) - 1, 2):
                pairs.append((active[i], active[i + 1]))
            if len(active) % 2 == 1:
                next_round.append(active[-1])

            round_label = f"round {round_num}"
            round_node, round_parent = enter_node("round", round_label,
                                                   matches=len(pairs))
            _emit(FlowEvent("start", round_label, depth + 1,
                             meta={"matches": len(pairs)}))

            try:
                # TaskGroup: a failing match cancels the round's others.
                judgments: list = [None] * len(pairs)
                async with asyncio.TaskGroup() as tg:
                    async def _match(i, a_text, b_text):
                        judgments[i] = await llm.extract(
                            judge_fn(a_text, b_text), schema=judge_schema,
                            model=model)
                    for i, ((_, a_text), (_, b_text)) in enumerate(pairs):
                        tg.create_task(_match(i, a_text, b_text))

                round_results = []
                for pair, judgment in zip(pairs, judgments):
                    (a_idx, a_text), (b_idx, b_text) = pair
                    verdict = judgment.get(winner_key)
                    if verdict not in ("a", "b"):
                        raise ValueError(
                            f"tournament() judgment field {winner_key!r} must "
                            f"be 'a' or 'b', got {verdict!r}. Declare it as an "
                            f"enum in judge_schema: {{{winner_key!r}: "
                            "{'type': 'string', 'enum': ['a', 'b']}}.")
                    winner = pair[0] if verdict == "a" else pair[1]
                    next_round.append(winner)
                    round_results.append({
                        "a_idx": a_idx, "b_idx": b_idx,
                        "winner_idx": winner[0], "judgment": judgment,
                    })
            except BaseException as e:
                exit_node(round_node, round_parent, error=llm._error_text(e))
                raise

            rounds_log.append(round_results)
            round_node.output = f"{len(next_round)} remaining"
            exit_node(round_node, round_parent)
            _emit(FlowEvent("complete", round_label, depth + 1,
                             result=f"{len(next_round)} remaining"))
            active = next_round

        winner_idx, winner_text = active[0]
        node.output = f"winner: #{winner_idx}"
        node.meta["rounds"] = round_num
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         result=node.output, meta={"rounds": round_num}))
        return winner_text, winner_idx, rounds_log
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


async def blackboard(
    agents: list[tuple[str, Callable[[str], Msg]]],
    seed: str,
    rounds: int = 3,
    *,
    title: str,
    show: str | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    filter_fn: Callable[[str, list[dict], str, int], str] | None = None,
    depth: int = 0,
) -> tuple[str, list[dict]]:
    """Shared-state expert panel. Each agent sees all prior contributions.

    agents is [(name, msg_fn), ...]. msg_fn(board_state) -> Msg.
    All agents contribute each round in parallel.
    Returns (final_board, history).

    filter_fn(board, history, agent_name, round) -> filtered_board
        Controls what each agent sees.

        board, history = await blackboard(
            agents=[
                ("historian", lambda b: historian | user(b)),
                ("economist", lambda b: economist | user(b)),
            ],
            seed="Question: Why did Rome fall?",
            title="expert panel",
            rounds=2,
        )
    """
    board = seed
    history = []
    agent_names = [name for name, _ in agents]

    node, parent = enter_node("blackboard", title, agents=agent_names, rounds=rounds,
                              model=_model_label(model), **_show_meta(show))
    _emit(FlowEvent("start", title, depth,
                     meta={"agents": agent_names, "rounds": rounds, "model": model}))

    try:
        for round_num in range(rounds):
            round_label = f"round {round_num + 1}"
            round_children = [f"{n} (r{round_num + 1})" for n in agent_names]
            round_node, round_parent = enter_node("round", round_label)
            _emit(FlowEvent("start", round_label, depth + 1,
                             children=round_children))

            async def _agent_call(name, fn, full_board, hist, rnd):
                visible = filter_fn(full_board, hist, name, rnd) if filter_fn else full_board
                node_label = f"{name} (r{rnd})"
                agent_node, agent_parent = enter_node("call", node_label, round=rnd)
                _emit(FlowEvent("start", node_label, depth + 2,
                                 meta={"round": rnd}))
                try:
                    result = await llm.complete(fn(visible), model=model)
                    agent_node.output = result
                    exit_node(agent_node, agent_parent)
                    _emit(FlowEvent("complete", node_label, depth + 2,
                                     result=_truncate(result),
                                     elapsed=agent_node.elapsed))
                    return result
                except BaseException as e:
                    exit_node(agent_node, agent_parent, error=llm._error_text(e))
                    raise

            try:
                # TaskGroup, not gather: a failing expert cancels the
                # others instead of letting them run (and bill) into a
                # round that already failed. Callers see ExceptionGroup.
                contributions: list = [None] * len(agents)
                async with asyncio.TaskGroup() as tg:
                    async def _contribute(i, name, fn):
                        contributions[i] = await _agent_call(
                            name, fn, board, history, round_num + 1)
                    for i, (name, fn) in enumerate(agents):
                        tg.create_task(_contribute(i, name, fn))
            except BaseException as e:
                exit_node(round_node, round_parent, error=llm._error_text(e))
                raise

            round_record = {}
            for (name, _), contribution in zip(agents, contributions):
                round_record[name] = contribution
            history.append(round_record)

            names = [name for name, _ in agents]
            board = Block.join(
                [board] + [f"[{n}, round {round_num + 1}]:\n{c}"
                           for n, c in zip(names, contributions)]
            )

            exit_node(round_node, round_parent)
            _emit(FlowEvent("complete", round_label, depth + 1))

        node.output = board[:500]
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         meta={"rounds": rounds}))
        return board, history
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise


# ---------------------------------------------------------------------------
# Agent — the Msg grows until a signal tool fires
# ---------------------------------------------------------------------------

# Sentinel for flow signal tools
FINISH = "__finish__"
DELEGATE = "__delegate__"
ESCALATE = "__escalate__"
ASK_USER = "__ask_user__"


@dataclass(slots=True)
class AgentResult:
    """What the agent loop returned."""
    output: str               # final answer or signal payload
    signal: str | None = None # None = natural finish, else FINISH/DELEGATE/etc.
    msg: Msg = field(default_factory=Msg)  # the full conversation
    steps: int = 0


async def agent(
    msg: Msg,
    tools: dict[str, Callable],
    tool_schemas: list[dict],
    *,
    title: str = "agent",
    show: str | None = None,
    signal_tools: dict[str, str] | None = None,
    model: str | llm.Endpoint | llm.RoleRef = _CONTENT,
    max_steps: int = 20,
    max_tokens: int = 100_000,
    timeout: float | None = None,
    finalize: bool = True,
) -> AgentResult:
    """Run an agent loop. The Msg grows until the model finishes or
    calls a flow signal tool.

    tools: {name: async handler(input_dict) -> str}
    tool_schemas: [{...}] — Anthropic format tool definitions
    signal_tools: {name: signal_type} — tools that break the loop
    max_tokens: threshold for automatic compaction (0 to disable)
    timeout: wall-clock seconds limit
    finalize: when max_steps runs out with the model still calling
        tools, make one closing call with no tools so the agent's last
        word is a written answer, not a half-finished search

        result = await agent(
            system("You can search and calculate.") | user("What's 2+2?"),
            tools={"calc": calc_handler},
            tool_schemas=SCHEMAS,
            title="calculator agent",
        )
    """
    signal_tools = signal_tools or {}
    node, parent = enter_node("agent", title, model=_model_label(model),
                              max_steps=max_steps, **_show_meta(show))
    _emit(FlowEvent("start", title, 0, meta={"model": _model_label(model), "max_steps": max_steps}))
    last_text = ""  # tracks the most recent output for timeout/max_steps

    try:
        for step in range(max_steps):
            # Wall-clock timeout
            if timeout and (time.monotonic() - node._start_time) > timeout:
                node.output = last_text
                node.meta["signal"] = "timeout"
                exit_node(node, parent)
                _emit(FlowEvent("complete", title, 0, elapsed=node.elapsed,
                                 result=f"timeout ({timeout}s)",
                                 meta={"steps": step, "signal": "timeout"}))
                return AgentResult(
                    output=last_text, signal="timeout", msg=msg, steps=step)

            # Silent compaction
            if max_tokens:
                msg = await compact(msg, max_tokens=max_tokens)

            step_label = f"step {step + 1}"
            step_node, step_parent = enter_node("step", step_label, step=step + 1)
            _emit(FlowEvent("start", step_label, 1, meta={"step": step + 1}))

            result = await llm.act(msg, tool_schemas, model=model)
            if result.text:
                last_text = result.text

            if result.stop_reason == "max_tokens":
                warnings.warn(f"agent step {step + 1}: response truncated (max_tokens)")
                step_node.output = "truncated"
                exit_node(step_node, step_parent)
                _emit(FlowEvent("error", step_label, 1,
                                 result="truncated (max_tokens)"))
                if result.text:
                    msg = msg | assistant(result.text)
                continue

            if result.done:
                if result.text:
                    msg = msg | assistant(result.text)
                step_node.output = result.text or ""
                exit_node(step_node, step_parent)
                _emit(FlowEvent("complete", step_label, 1,
                                 result=_truncate(result.text or ""),
                                 elapsed=step_node.elapsed))

                node.output = result.text or ""
                node.meta["signal"] = None
                exit_node(node, parent)
                _emit(FlowEvent("complete", title, 0, elapsed=node.elapsed,
                                 result=_truncate(result.text or ""),
                                 meta={"steps": step + 1, "signal": None}))
                return AgentResult(
                    output=result.text or "", signal=None,
                    msg=msg, steps=step + 1)

            # Preserve assistant narration before tool calls
            if result.text:
                msg = msg | assistant(result.text)

            # Process tool calls
            for call in result.tool_calls:
                call_label = f"{call.name} ({call.id[:8]})"
                tool_node, tool_parent = enter_node("tool_call", call_label,
                                                     tool_id=call.id)
                _emit(FlowEvent("start", call_label, 2,
                                 meta={"tool_id": call.id}))

                msg = msg | tool_use(call.id, call.name, call.input)

                # Check if it's a signal tool
                if call.name in signal_tools:
                    signal = signal_tools[call.name]
                    try:
                        if call.name in tools:
                            # Handler gets its own copy — a mutating handler
                            # (input.pop(...)) must not rewrite the recorded call.
                            output = await tools[call.name](copy.deepcopy(call.input))
                        else:
                            output = str(call.input)
                    except Exception as e:
                        # A failing signal handler must not crash the loop
                        # (parity with regular tools): surface an error
                        # result and let the model decide what's next.
                        output = f"Error: {e}"
                        msg = msg | tool_result(call.id, output, is_error=True)
                        tool_node.output = output
                        exit_node(tool_node, tool_parent, error=str(e))
                        _emit(FlowEvent("error", call_label, 2,
                                         result=str(e), elapsed=tool_node.elapsed))
                        continue

                    tool_node.output = f"SIGNAL:{signal} {output[:200]}"
                    exit_node(tool_node, tool_parent)
                    _emit(FlowEvent("complete", call_label, 2,
                                     result=f"SIGNAL:{signal} {_truncate(output)}",
                                     elapsed=tool_node.elapsed))

                    msg = msg | tool_result(call.id, output)
                    step_node.output = f"signal: {signal}"
                    exit_node(step_node, step_parent)
                    _emit(FlowEvent("complete", step_label, 1,
                                     elapsed=step_node.elapsed))

                    node.output = output
                    node.meta["signal"] = signal
                    exit_node(node, parent)
                    _emit(FlowEvent("complete", title, 0, elapsed=node.elapsed,
                                     result=f"signal: {signal}",
                                     meta={"steps": step + 1, "signal": signal}))
                    return AgentResult(
                        output=output, signal=signal,
                        msg=msg, steps=step + 1)

                # Regular tool — execute and append result
                if call.name not in tools:
                    output = f"Error: unknown tool '{call.name}'"
                    msg = msg | tool_result(call.id, output, is_error=True)
                    tool_node.output = output
                    exit_node(tool_node, tool_parent, error=output)
                    _emit(FlowEvent("complete", call_label, 2,
                                     result=f"error: {output}",
                                     elapsed=tool_node.elapsed))
                    continue

                try:
                    handler = tools[call.name]
                    # Handler gets its own copy — a mutating handler
                    # (input.pop(...)) must not rewrite the recorded call.
                    output = await handler(copy.deepcopy(call.input))
                except Exception as e:
                    output = f"Error: {e}"
                    msg = msg | tool_result(call.id, output, is_error=True)
                    tool_node.output = output
                    exit_node(tool_node, tool_parent, error=str(e))
                    _emit(FlowEvent("error", call_label, 2,
                                     result=str(e), elapsed=tool_node.elapsed))
                    continue

                msg = msg | tool_result(call.id, str(output))
                tool_node.output = str(output)[:500]
                exit_node(tool_node, tool_parent)
                _emit(FlowEvent("complete", call_label, 2,
                                 result=_truncate(str(output)),
                                 elapsed=tool_node.elapsed))

            step_node.output = f"{len(result.tool_calls)} tool calls"
            exit_node(step_node, step_parent)
            _emit(FlowEvent("complete", step_label, 1,
                             elapsed=step_node.elapsed,
                             meta={"tool_calls": len(result.tool_calls)}))

        # Max steps reached
        if finalize and tool_schemas:
            # The model was still reaching for tools when the budget ran
            # out. One closing call with no tools turns everything
            # gathered into an actual answer.
            fin_node, fin_parent = enter_node("step", "finalize", finalize=True)
            _emit(FlowEvent("start", "finalize", 1, meta={"finalize": True}))
            try:
                final_text = await llm.complete(
                    msg | user(
                        "Your tool budget is exhausted. Write your complete "
                        "final answer now, based on everything gathered above."),
                    model=model)
                if final_text:
                    last_text = final_text
                    msg = msg | assistant(final_text)
                fin_node.output = final_text or ""
                exit_node(fin_node, fin_parent)
                _emit(FlowEvent("complete", "finalize", 1,
                                 result=_truncate(final_text or ""),
                                 elapsed=fin_node.elapsed))
            except Exception as e:
                # Finalize is best-effort — fall back to last_text.
                exit_node(fin_node, fin_parent, error=str(e))
                _emit(FlowEvent("error", "finalize", 1, result=str(e)))

        node.output = last_text
        node.meta["signal"] = "max_steps"
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, 0, elapsed=node.elapsed,
                         result=f"max steps ({max_steps})",
                         meta={"steps": max_steps, "signal": "max_steps"}))
        return AgentResult(
            output=last_text, signal="max_steps",
            msg=msg, steps=max_steps)
    except BaseException as e:
        # BaseException: a cancelled pattern must not leave its node
        # "running" forever. _error_text: str(RuntimeError()) is "".
        exit_node(node, parent, error=llm._error_text(e))
        raise
