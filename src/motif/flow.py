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
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Any

from .prompt import (
    Msg, Block, TextSegment, ToolCall, ToolResult,
    system, user, assistant, tool_use, tool_result,
)
from . import llm
from .graph import enter_node, exit_node, current_node, Node, _new_id

# Re-export show machinery so users can do flow.show(), flow.showing(), etc.
from .show import show, show_to, showing, clear_show_observers

# Flow structural decisions (branch, best_of, etc.) use the cheap model.
# Content generation uses llm.DEFAULT_MODEL. This split is intentional:
# the topology should be cheap to discover; the work should be high quality.
_CHEAP = "claude-haiku-4-5"


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


def observe(*observers: Callable[[FlowEvent], None]):
    """Attach observers that receive every flow event."""
    _observers.extend(observers)


class observing:
    """Context manager that attaches observers and removes them on exit.

        async with flow.observing(trace, display):
            result = await flow.fan(items, fn, title="analyze")
        # observers automatically removed — no manual clear_observers()
    """

    def __init__(self, *observers: Callable[[FlowEvent], None]):
        self._observers = list(observers)

    async def __aenter__(self):
        _observers.extend(self._observers)
        return self

    async def __aexit__(self, *args):
        for obs in self._observers:
            try:
                _observers.remove(obs)
            except ValueError:
                pass


def clear_observers():
    """Remove all flow observers."""
    _observers.clear()


def _emit(event: FlowEvent):
    for obs in _observers:
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


def _item_label(item: Any, idx: int) -> str:
    """Extract a display label from a list item."""
    if isinstance(item, dict):
        return str(item.get("name", item.get("label", f"item_{idx}")))
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


def _check_label_kwarg(kw: dict):
    """Catch old label= usage in **kw before it silently passes through."""
    if "label" in kw:
        raise TypeError(
            "Use title= instead of label= (renamed in motif 0.2). "
            "title is now a required keyword argument.")


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


async def compact(
    msg: Msg,
    *,
    max_tokens: int = 100_000,
    keep_recent: int = 6,
    model: str = llm.DEFAULT_MODEL,
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

    # Find the split point, then expand to preserve tool_use/tool_result pairs.
    split_at = len(rest) - keep_recent

    tail_tool_result_ids = set()
    tail_tool_use_ids = set()
    for seg in rest[split_at:]:
        if isinstance(seg, ToolResult):
            tail_tool_result_ids.add(seg.tool_use_id)
        elif isinstance(seg, ToolCall):
            tail_tool_use_ids.add(seg.id)

    while split_at > 0:
        seg = rest[split_at - 1]
        pull = False
        if isinstance(seg, ToolCall) and seg.id in tail_tool_result_ids:
            pull = True
            tail_tool_use_ids.add(seg.id)
        elif isinstance(seg, ToolResult) and seg.tool_use_id in tail_tool_use_ids:
            pull = True
            tail_tool_result_ids.add(seg.tool_use_id)
        if pull:
            split_at -= 1
        else:
            break

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
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise


# ---------------------------------------------------------------------------
# Branching — one becomes many
# ---------------------------------------------------------------------------

async def branch(
    msg: Msg,
    schema: dict,
    *,
    title: str,
    model: str = _CHEAP,
    depth: int = 0,
    **kw,
) -> list[dict]:
    """One call discovers structure. Returns a list of items.

    The schema should produce an object with an array field.
    branch() finds the first list in the result and returns it.

        methods = await branch(
            system("List methodologies...") | user(doc),
            title="discover angles",
            schema=METHODS_SCHEMA,
        )
    """
    _check_label_kwarg(kw)
    node, parent = enter_node("branch", title, model=model)
    _emit(FlowEvent("start", title, depth, meta={"model": model}))

    try:
        result = await llm.extract(msg, schema=schema, model=model, **kw)

        items = [result]
        for v in result.values():
            if isinstance(v, list):
                items = v
                break

        child_labels = [_item_label(item, i) for i, item in enumerate(items)]
        node.output = ", ".join(child_labels)
        exit_node(node, parent)
        _emit(FlowEvent("split", title, depth, children=child_labels,
                         elapsed=node.elapsed,
                         meta={"count": len(items), "model": model}))
        return items
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise


async def fan(
    items: list,
    fn: Callable[[Any], Msg],
    *,
    title: str,
    model: str = llm.DEFAULT_MODEL,
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
    node, parent = enter_node("fan", title, model=model, count=len(items))
    _emit(FlowEvent("start", title, depth, meta={"count": len(items), "model": model}))
    _emit(FlowEvent("split", title, depth, children=child_labels,
                     meta={"count": len(items), "model": model}))
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _one(item, idx):
        # enter_node BEFORE semaphore so all children appear in the graph
        # immediately — TUI can build layout before work starts.
        name = _item_label(item, idx)
        child, child_parent = enter_node("call", name, model=model)
        _emit(FlowEvent("start", name, depth + 1, meta={"model": model}))

        if sem:
            await sem.acquire()
        try:
            result = await llm.complete(
                fn(item), model=model, streaming=streaming,
                meta={"node": name}, **kw)
            child.output = result
            exit_node(child, child_parent)
            _emit(FlowEvent("complete", name, depth + 1,
                             result=_truncate(result), elapsed=child.elapsed))
            return result
        except Exception as e:
            exit_node(child, child_parent, error=str(e))
            _emit(FlowEvent("error", name, depth + 1, result=str(e)))
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
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise


# ---------------------------------------------------------------------------
# Converging — many become one
# ---------------------------------------------------------------------------

async def reduce(
    results: list[str],
    msg_fn: Callable[[str], Msg],
    *,
    title: str,
    labels: list[str] | None = None,
    model: str = llm.DEFAULT_MODEL,
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
    node, parent = enter_node("reduce", title, model=model, inputs=len(results))
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
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise


async def best_of(
    candidates: list[str],
    judge_fn: Callable[[str], Msg],
    judge_schema: dict,
    *,
    title: str,
    model: str = _CHEAP,
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
    node, parent = enter_node("best_of", title,
                              model=model, candidates=len(candidates))
    _emit(FlowEvent("start", title, depth,
                     meta={"candidates": len(candidates), "model": model}))

    try:
        judgments = await asyncio.gather(*[
            llm.extract(judge_fn(c), schema=judge_schema, model=model)
            for c in candidates
        ])

        best_idx = max(range(len(judgments)),
                       key=lambda i: judgments[i].get(score_key, 0))

        node.output = f"winner: #{best_idx} (score {judgments[best_idx].get(score_key)})"
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, depth, elapsed=node.elapsed,
                         result=node.output))
        return candidates[best_idx], best_idx, judgments
    except Exception as e:
        exit_node(node, parent, error=str(e))
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
    model_test: str = _CHEAP,
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
    node, parent = enter_node("cascade", title, models=models)
    _emit(FlowEvent("start", title, depth, meta={"models": models}))
    used_model = models[-1]  # fallback
    result = ""

    try:
        for used_model in models:
            child, child_parent = enter_node("call", used_model)
            _emit(FlowEvent("start", used_model, depth + 1))

            result = await llm.complete(msg, model=used_model)

            if used_model == models[-1]:  # last model — accept regardless
                child.output = result
                exit_node(child, child_parent)
                _emit(FlowEvent("complete", used_model, depth + 1,
                                 result=_truncate(result), elapsed=child.elapsed))
                break

            judgment = await llm.extract(test_fn(result), schema=test_schema,
                                         model=model_test)

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
    except Exception as e:
        exit_node(node, parent, error=str(e))
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
    paragraph_fn: Callable[[str], list[str]] | None = None,
    max_depth: int = 3,
    model_split: str = _CHEAP,
    model_leaf: str = llm.DEFAULT_MODEL,
    model_merge: str = llm.DEFAULT_MODEL,
    _depth: int = 0,
) -> str:
    """Recursive decomposition. Split until leaves, work leaves, merge up.

    The splitter returns paragraph ranges, not reproduced text. The
    original text is sliced by the framework — no JSON reproduction of
    large documents.

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
    node, parent = enter_node("tree", title, chars=len(task))
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

        # Ask whether to split
        decision = await llm.extract(split_fn(task), schema=split_schema,
                                     model=model_split)

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

        # Slice the original text by paragraph ranges — no text in JSON
        _split = paragraph_fn or (lambda t: t.split("\n\n"))
        paragraphs = _split(task)
        child_labels = []
        child_texts = []

        for s in subtasks:
            clabel = s.get("label", s.get("name", f"part_{len(child_labels)}"))
            child_labels.append(clabel)

            if "text" in s:
                child_texts.append(s["text"])
            elif "start_paragraph" in s:
                start = s.get("start_paragraph", 0)
                end = s.get("end_paragraph", len(paragraphs))
                child_texts.append("\n\n".join(paragraphs[start:end]))
            else:
                n = len(subtasks)
                chunk = len(paragraphs) // n
                i = len(child_texts)
                child_texts.append("\n\n".join(
                    paragraphs[i * chunk : (i + 1) * chunk if i < n - 1 else len(paragraphs)]))

        _emit(FlowEvent("split", title, _depth, children=child_labels,
                         elapsed=time.monotonic() - node._start_time))

        # Recurse in parallel — each recursive call creates its own graph node
        child_results = await asyncio.gather(*[
            tree(
                text, split_fn, split_schema, leaf_fn, merge_fn,
                paragraph_fn=paragraph_fn, max_depth=max_depth,
                model_split=model_split, model_leaf=model_leaf,
                model_merge=model_merge, title=clabel, _depth=_depth + 1,
            )
            for clabel, text in zip(child_labels, child_texts)
        ])

        # Merge — pass labeled results to merge_fn
        merged = await llm.complete(
            merge_fn(child_results, child_labels), model=model_merge)
        node.output = merged
        exit_node(node, parent)
        _emit(FlowEvent("merge", title, _depth,
                         result=_truncate(merged), elapsed=node.elapsed))
        return merged
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise


async def tournament(
    candidates: list[str],
    judge_fn: Callable[[str, str], Msg],
    judge_schema: dict,
    *,
    title: str,
    model: str = _CHEAP,
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

    node, parent = enter_node("tournament", title,
                              model=model, candidates=len(candidates))
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

            judgments = await asyncio.gather(*[
                llm.extract(judge_fn(a_text, b_text), schema=judge_schema,
                            model=model)
                for (_, a_text), (_, b_text) in pairs
            ])

            round_results = []
            for pair, judgment in zip(pairs, judgments):
                (a_idx, a_text), (b_idx, b_text) = pair
                winner = pair[0] if judgment.get(winner_key) == "a" else pair[1]
                next_round.append(winner)
                round_results.append({
                    "a_idx": a_idx, "b_idx": b_idx,
                    "winner_idx": winner[0], "judgment": judgment,
                })

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
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise


async def blackboard(
    agents: list[tuple[str, Callable[[str], Msg]]],
    seed: str,
    rounds: int = 3,
    *,
    title: str,
    model: str = llm.DEFAULT_MODEL,
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

    node, parent = enter_node("blackboard", title,
                              agents=agent_names, rounds=rounds, model=model)
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
                except Exception as e:
                    exit_node(agent_node, agent_parent, error=str(e))
                    raise

            contributions = await asyncio.gather(*[
                _agent_call(name, fn, board, history, round_num + 1)
                for name, fn in agents
            ])

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
    except Exception as e:
        exit_node(node, parent, error=str(e))
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
    signal_tools: dict[str, str] | None = None,
    model: str = llm.DEFAULT_MODEL,
    max_steps: int = 20,
    max_tokens: int = 100_000,
    timeout: float | None = None,
) -> AgentResult:
    """Run an agent loop. The Msg grows until the model finishes or
    calls a flow signal tool.

    tools: {name: async handler(input_dict) -> str}
    tool_schemas: [{...}] — Anthropic format tool definitions
    signal_tools: {name: signal_type} — tools that break the loop
    max_tokens: threshold for automatic compaction (0 to disable)
    timeout: wall-clock seconds limit

        result = await agent(
            system("You can search and calculate.") | user("What's 2+2?"),
            tools={"calc": calc_handler},
            tool_schemas=SCHEMAS,
            title="calculator agent",
        )
    """
    signal_tools = signal_tools or {}
    node, parent = enter_node("agent", title, model=model, max_steps=max_steps)
    _emit(FlowEvent("start", title, 0, meta={"model": model, "max_steps": max_steps}))
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
                    if call.name in tools:
                        output = await tools[call.name](call.input)
                    else:
                        output = str(call.input)

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
                    output = await handler(call.input)
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
        node.output = last_text
        node.meta["signal"] = "max_steps"
        exit_node(node, parent)
        _emit(FlowEvent("complete", title, 0, elapsed=node.elapsed,
                         result=f"max steps ({max_steps})",
                         meta={"steps": max_steps, "signal": "max_steps"}))
        return AgentResult(
            output=last_text, signal="max_steps",
            msg=msg, steps=max_steps)
    except Exception as e:
        exit_node(node, parent, error=str(e))
        raise
