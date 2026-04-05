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

All functions emit FlowEvents to observers. Attach a display with
flow.observe(display) to watch the computation graph execute.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Any

from .prompt import (
    Msg, Block, TextSegment, ToolCall, ToolResult,
    system, user, assistant, tool_use, tool_result,
)
from . import llm

# Flow structural decisions (branch, best_of, etc.) use the cheap model.
# Content generation uses llm.DEFAULT_MODEL. This split is intentional:
# the topology should be cheap to discover; the work should be high quality.
_CHEAP = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Events — the topology makes itself visible
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
    # A tool_result must reference an existing tool_use in the same conversation,
    # so we never split a pair across the compaction boundary.
    split_at = len(rest) - keep_recent

    # Collect all tool_use IDs in the kept tail
    tail_tool_result_ids = set()
    tail_tool_use_ids = set()
    for seg in rest[split_at:]:
        if isinstance(seg, ToolResult):
            tail_tool_result_ids.add(seg.tool_use_id)
        elif isinstance(seg, ToolCall):
            tail_tool_use_ids.add(seg.id)

    # Pull any tool_use referenced by a kept tool_result into the tail
    # Pull any tool_result referencing a kept tool_use into the tail
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

    # Render the compactable segments as text for the summarizer
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

    _emit(FlowEvent("start", "compact", 0,
                     meta={"tokens_before": est, "segments_compacted": len(to_compact)}))
    t0 = time.monotonic()

    summary = await llm.complete(
        COMPACT_PROMPT | user(history_text),
        model=model,
    )

    elapsed = time.monotonic() - t0
    summary_seg = TextSegment("user", f"[Prior conversation summary]\n{summary}")
    new_msg = Msg(segments=tuple(system_segs + [summary_seg] + to_keep))
    new_est = _estimate_tokens(new_msg)

    _emit(FlowEvent("complete", "compact", 0, elapsed=elapsed,
                     result=f"{est} → {new_est} tokens (est)",
                     meta={"tokens_after": new_est}))
    return new_msg


# ---------------------------------------------------------------------------
# Branching — one becomes many
# ---------------------------------------------------------------------------

async def branch(
    msg: Msg,
    schema: dict,
    *,
    model: str = _CHEAP,
    label: str = "branch",
    depth: int = 0,
    **kw,
) -> list[dict]:
    """One call discovers structure. Returns a list of items.

    The schema should produce an object with an array field.
    branch() finds the first list in the result and returns it.

        methods = await branch(
            system("List methodologies...") | user(doc),
            schema=METHODS_SCHEMA,
        )
    """
    _emit(FlowEvent("start", label, depth, meta={"model": model}))
    t0 = time.monotonic()

    result = await llm.extract(msg, schema=schema, model=model, **kw)

    items = [result]
    for v in result.values():
        if isinstance(v, list):
            items = v
            break

    elapsed = time.monotonic() - t0
    child_labels = [_item_label(item, i) for i, item in enumerate(items)]
    _emit(FlowEvent("split", label, depth, children=child_labels, elapsed=elapsed,
                     meta={"count": len(items), "model": model}))
    return items


async def fan(
    items: list,
    fn: Callable[[Any], Msg],
    *,
    model: str = llm.DEFAULT_MODEL,
    max_concurrency: int | None = None,
    label: str = "fan",
    depth: int = 0,
    **kw,
) -> list[str]:
    """Parallel complete() over items. fn maps each item to a Msg.

    max_concurrency limits how many calls run simultaneously (for rate limits).

        analyses = await fan(
            methods,
            lambda m: analyst | user(f"Use {m['name']}:\\n{doc}"),
            max_concurrency=5,
        )
    """
    _emit(FlowEvent("start", label, depth, meta={"count": len(items), "model": model}))
    t0 = time.monotonic()
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _one(item, idx):
        if sem:
            await sem.acquire()
        try:
            name = _item_label(item, idx)
            _emit(FlowEvent("start", name, depth + 1, meta={"model": model}))
            t1 = time.monotonic()
            result = await llm.complete(fn(item), model=model, **kw)
            elapsed = time.monotonic() - t1
            _emit(FlowEvent("complete", name, depth + 1,
                             result=_truncate(result), elapsed=elapsed))
            return result
        finally:
            if sem:
                sem.release()

    # TaskGroup cancels remaining tasks if one fails (better than gather
    # for rate-limited APIs — don't fire 49 more into a 429)
    results: list = [None] * len(items)
    async with asyncio.TaskGroup() as tg:
        async def _run(i, item):
            results[i] = await _one(item, i)
        for i, item in enumerate(items):
            tg.create_task(_run(i, item))

    elapsed = time.monotonic() - t0
    _emit(FlowEvent("complete", label, depth, elapsed=elapsed,
                     meta={"count": len(results)}))
    return results


# ---------------------------------------------------------------------------
# Converging — many become one
# ---------------------------------------------------------------------------

async def reduce(
    results: list[str],
    msg_fn: Callable[[str], Msg],
    *,
    labels: list[str] | None = None,
    model: str = llm.DEFAULT_MODEL,
    label: str = "reduce",
    depth: int = 0,
    **kw,
) -> str:
    """Many results converge into one. msg_fn receives the combined text.

    labels, if provided, wraps each result as [label]:\\n...

        synthesis = await reduce(
            analyses,
            lambda combined: synthesizer | user(combined),
            labels=[m["name"] for m in methods],
        )
    """
    _emit(FlowEvent("start", label, depth, meta={"inputs": len(results), "model": model}))
    t0 = time.monotonic()

    combined = _join(results, labels=labels)
    result = await llm.complete(msg_fn(combined), model=model, **kw)

    elapsed = time.monotonic() - t0
    _emit(FlowEvent("merge", label, depth, result=_truncate(result), elapsed=elapsed))
    return result


async def best_of(
    candidates: list[str],
    judge_fn: Callable[[str], Msg],
    judge_schema: dict,
    *,
    model: str = _CHEAP,
    score_key: str = "score",
    label: str = "best_of",
    depth: int = 0,
) -> tuple[str, int, list[dict]]:
    """Judge picks the best from N candidates.

    Returns (best, index, all_judgments). Judging happens in parallel.

        best, idx, scores = await best_of(
            drafts,
            lambda d: judge | user(f"Rate 1-10:\\n{d}"),
            schema=SCORE_SCHEMA,
        )
    """
    _emit(FlowEvent("start", label, depth,
                     meta={"candidates": len(candidates), "model": model}))
    t0 = time.monotonic()

    judgments = await asyncio.gather(*[
        llm.extract(judge_fn(c), schema=judge_schema, model=model)
        for c in candidates
    ])

    best_idx = max(range(len(judgments)),
                   key=lambda i: judgments[i].get(score_key, 0))
    elapsed = time.monotonic() - t0

    _emit(FlowEvent("complete", label, depth, elapsed=elapsed,
                     result=f"winner: #{best_idx} (score {judgments[best_idx].get(score_key)})"))
    return candidates[best_idx], best_idx, judgments


# ---------------------------------------------------------------------------
# Linear — cost optimization
# ---------------------------------------------------------------------------

async def cascade(
    msg: Msg,
    test_fn: Callable[[str], Msg],
    test_schema: dict,
    models: list[str],
    *,
    model_test: str = _CHEAP,
    label: str = "cascade",
    depth: int = 0,
) -> tuple[str, str]:
    """Try cheap models first, escalate until quality passes.

    Returns (result, model_used). test_schema needs a "sufficient" boolean.

        answer, model_used = await cascade(
            system("Answer precisely.") | user(question),
            test_fn=lambda ans: checker | user(f"Is this correct?\\n{ans}"),
            test_schema=QUALITY_SCHEMA,
            models=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
        )
    """
    _emit(FlowEvent("start", label, depth, meta={"models": models}))
    t0 = time.monotonic()
    used_model = models[-1]  # fallback
    result = ""

    for used_model in models:
        _emit(FlowEvent("start", used_model, depth + 1))
        t1 = time.monotonic()

        result = await llm.complete(msg, model=used_model)

        if used_model == models[-1]:  # last model — accept regardless
            elapsed = time.monotonic() - t1
            _emit(FlowEvent("complete", used_model, depth + 1,
                             result=_truncate(result), elapsed=elapsed))
            break

        judgment = await llm.extract(test_fn(result), schema=test_schema,
                                     model=model_test)
        elapsed = time.monotonic() - t1

        if judgment.get("sufficient", False):
            _emit(FlowEvent("complete", used_model, depth + 1,
                             result=_truncate(result), elapsed=elapsed))
            break
        else:
            _emit(FlowEvent("complete", used_model, depth + 1,
                             result="insufficient — escalating", elapsed=elapsed))

    total = time.monotonic() - t0
    _emit(FlowEvent("complete", label, depth, elapsed=total,
                     result=f"settled on {used_model}", meta={"model_used": used_model}))
    return result, used_model


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
    max_depth: int = 3,
    model_split: str = _CHEAP,
    model_leaf: str = llm.DEFAULT_MODEL,
    model_merge: str = llm.DEFAULT_MODEL,
    label: str = "tree",
    _depth: int = 0,
) -> str:
    """Recursive decomposition. Split until leaves, work leaves, merge up.

    The splitter returns paragraph ranges, not reproduced text. The
    original text is sliced by the framework — no JSON reproduction of
    large documents. Paragraphs are \\n\\n-separated chunks of the input.

    split_fn(task) → Msg asking the model to split or analyze as-is.
    split_schema must have:
        is_leaf: bool
        subtasks: [{label: str, start_paragraph: int, end_paragraph: int}]
    Also accepts subtasks with "text" field for backward compat or
    custom splitting that doesn't use paragraph ranges.
    leaf_fn(task) → Msg for leaf-level work.
    merge_fn(results, labels) → Msg for combining child results.

        result = await tree(
            task=long_document,
            split_fn=lambda t: splitter | user(t),
            split_schema=SPLIT_SCHEMA,
            leaf_fn=lambda t: worker | user(t),
            merge_fn=lambda rs, ls: combiner | user(Block.join(rs, labels=ls)),
        )
    """
    _emit(FlowEvent("start", label, _depth, meta={"chars": len(task)}))
    t0 = time.monotonic()

    # At max depth, force leaf
    if _depth >= max_depth:
        result = await llm.complete(leaf_fn(task), model=model_leaf)
        elapsed = time.monotonic() - t0
        _emit(FlowEvent("complete", label, _depth,
                         result=_truncate(result), elapsed=elapsed))
        return result

    # Ask whether to split
    decision = await llm.extract(split_fn(task), schema=split_schema,
                                 model=model_split)

    if decision.get("is_leaf", True):
        result = await llm.complete(leaf_fn(task), model=model_leaf)
        elapsed = time.monotonic() - t0
        _emit(FlowEvent("complete", label, _depth,
                         result=_truncate(result), elapsed=elapsed))
        return result

    subtasks = decision.get("subtasks", [])
    if not subtasks or len(subtasks) < 2:
        result = await llm.complete(leaf_fn(task), model=model_leaf)
        elapsed = time.monotonic() - t0
        _emit(FlowEvent("complete", label, _depth,
                         result=_truncate(result), elapsed=elapsed))
        return result

    # Slice the original text by paragraph ranges — no text in JSON
    paragraphs = task.split("\n\n")
    child_labels = []
    child_texts = []

    for s in subtasks:
        clabel = s.get("label", s.get("name", f"part_{len(child_labels)}"))
        child_labels.append(clabel)

        if "text" in s:
            # Backwards compat: if the model returned full text, use it
            child_texts.append(s["text"])
        elif "start_paragraph" in s:
            start = s.get("start_paragraph", 0)
            end = s.get("end_paragraph", len(paragraphs))
            child_texts.append("\n\n".join(paragraphs[start:end]))
        else:
            # Fallback: even split
            n = len(subtasks)
            chunk = len(paragraphs) // n
            i = len(child_texts)
            child_texts.append("\n\n".join(
                paragraphs[i * chunk : (i + 1) * chunk if i < n - 1 else len(paragraphs)]))

    _emit(FlowEvent("split", label, _depth, children=child_labels,
                     elapsed=time.monotonic() - t0))

    # Recurse in parallel
    child_results = await asyncio.gather(*[
        tree(
            text, split_fn, split_schema, leaf_fn, merge_fn,
            max_depth=max_depth, model_split=model_split,
            model_leaf=model_leaf, model_merge=model_merge,
            label=clabel, _depth=_depth + 1,
        )
        for clabel, text in zip(child_labels, child_texts)
    ])

    # Merge — pass labeled results to merge_fn
    merged = await llm.complete(
        merge_fn(child_results, child_labels), model=model_merge)
    elapsed = time.monotonic() - t0
    _emit(FlowEvent("merge", label, _depth,
                     result=_truncate(merged), elapsed=elapsed))
    return merged


async def tournament(
    candidates: list[str],
    judge_fn: Callable[[str, str], Msg],
    judge_schema: dict,
    *,
    model: str = _CHEAP,
    winner_key: str = "winner",
    label: str = "tournament",
    depth: int = 0,
) -> tuple[str, int, list]:
    """Bracket-style elimination. judge_fn(a, b) → Msg.

    judge_schema must have a field (winner_key) valued "a" or "b".
    Returns (winner_text, original_index, rounds_log).

        winner, idx, log = await tournament(
            drafts,
            lambda a, b: judge | user(f"Which is better?\\nA: {a}\\nB: {b}"),
            schema=WINNER_SCHEMA,
        )
    """
    if not candidates:
        raise ValueError("Need at least one candidate")
    if len(candidates) == 1:
        return candidates[0], 0, []

    _emit(FlowEvent("start", label, depth,
                     meta={"candidates": len(candidates), "model": model}))
    t0 = time.monotonic()

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

        _emit(FlowEvent("start", f"round {round_num}", depth + 1,
                         meta={"matches": len(pairs)}))

        judgments = await asyncio.gather(*[
            llm.extract(judge_fn(a_text, b_text), schema=judge_schema, model=model)
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
        _emit(FlowEvent("complete", f"round {round_num}", depth + 1,
                         result=f"{len(next_round)} remaining"))
        active = next_round

    winner_idx, winner_text = active[0]
    elapsed = time.monotonic() - t0
    _emit(FlowEvent("complete", label, depth, elapsed=elapsed,
                     result=f"winner: #{winner_idx}",
                     meta={"rounds": round_num}))
    return winner_text, winner_idx, rounds_log


async def blackboard(
    agents: list[tuple[str, Callable[[str], Msg]]],
    seed: str,
    rounds: int = 3,
    *,
    model: str = llm.DEFAULT_MODEL,
    filter_fn: Callable[[str, list[dict], str, int], str] | None = None,
    depth: int = 0,
) -> tuple[str, list[dict]]:
    """Shared-state expert panel. Each agent sees all prior contributions.

    agents is [(name, msg_fn), ...]. msg_fn(board_state) → Msg.
    All agents contribute each round in parallel.
    Returns (final_board, history).

    filter_fn(board, history, agent_name, round) → filtered_board
        Controls what each agent sees. Without it, everyone sees everything.
        board is the full text. history is the structured record:
        [{agent_name: contribution, ...}, ...] per round. The filter can
        operate on either — use the string for simple truncation, the
        structured history for selective listening (e.g., agent A hears
        agent B but ignores agent C based on arousal state).

        board, history = await blackboard(
            agents=[
                ("historian", lambda b, h, n, r: historian | user(b)),
                ("economist", lambda b, h, n, r: economist | user(b)),
                ("philosopher", lambda b, h, n, r: philosopher | user(b)),
            ],
            seed="Question: Why did Rome fall?",
            rounds=2,
            filter_fn=lambda board, history, name, rnd: board[-2000:] if rnd > 1 else board,
        )
    """
    board = seed
    history = []
    agent_names = [name for name, _ in agents]

    _emit(FlowEvent("start", "blackboard", depth,
                     meta={"agents": agent_names, "rounds": rounds, "model": model}))
    t_total = time.monotonic()

    for round_num in range(rounds):
        round_label = f"round {round_num + 1}"
        round_children = [f"{n} (r{round_num + 1})" for n in agent_names]
        _emit(FlowEvent("start", round_label, depth + 1, children=round_children))

        async def _agent_call(name, fn, full_board, hist, rnd):
            visible = filter_fn(full_board, hist, name, rnd) if filter_fn else full_board
            node_label = f"{name} (r{rnd})"
            _emit(FlowEvent("start", node_label, depth + 2, meta={"round": rnd}))
            t0 = time.monotonic()
            result = await llm.complete(fn(visible), model=model)
            elapsed = time.monotonic() - t0
            _emit(FlowEvent("complete", node_label, depth + 2,
                             result=_truncate(result), elapsed=elapsed))
            return result

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

        _emit(FlowEvent("complete", round_label, depth + 1))

    elapsed = time.monotonic() - t_total
    _emit(FlowEvent("complete", "blackboard", depth, elapsed=elapsed,
                     meta={"rounds": rounds}))
    return board, history


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
    signal_tools: dict[str, str] | None = None,
    model: str = llm.DEFAULT_MODEL,
    max_steps: int = 20,
    max_tokens: int = 100_000,
    timeout: float | None = None,
    label: str = "agent",
) -> AgentResult:
    """Run an agent loop. The Msg grows until the model finishes or
    calls a flow signal tool.

    tools: {name: async handler(input_dict) → str}
        The actual tool implementations.

    tool_schemas: [{"name": ..., "description": ..., "input_schema": ...}, ...]
        Tool definitions sent to the model (Anthropic format).

    timeout: wall-clock seconds for the entire agent run. None = no limit.

    signal_tools: {name: signal_type}
        Tools whose invocation breaks the loop instead of appending
        results. signal_type is one of FINISH, DELEGATE, ESCALATE, ASK_USER.
        The handler is still called — its return value becomes the output.
        If None, no signal tools are registered.

    max_tokens: estimated token threshold for automatic compaction.
        When the Msg exceeds this, older turns are silently summarized
        while preserving system segments and recent turns. Set to 0
        to disable.

        result = await agent(
            system("You can search and calculate.") | user("What's 2+2?"),
            tools={"calc": calc_handler, "search": search_handler},
            tool_schemas=SCHEMAS,
        )
        print(result.output)  # the final answer
        print(result.steps)   # how many tool-use rounds
    """
    signal_tools = signal_tools or {}
    _emit(FlowEvent("start", label, 0, meta={"model": model, "max_steps": max_steps}))
    t_total = time.monotonic()

    for step in range(max_steps):
        # Wall-clock timeout
        if timeout and (time.monotonic() - t_total) > timeout:
            total = time.monotonic() - t_total
            _emit(FlowEvent("complete", label, 0, elapsed=total,
                             result=f"timeout ({timeout}s)",
                             meta={"steps": step, "signal": "timeout"}))
            return AgentResult(
                output=result.text if step > 0 and result else "",
                signal="timeout",
                msg=msg,
                steps=step,
            )

        # Silent compaction — user never thinks about context limits
        if max_tokens:
            msg = await compact(msg, max_tokens=max_tokens)

        step_label = f"step {step + 1}"
        _emit(FlowEvent("start", step_label, 1, meta={"step": step + 1}))
        t0 = time.monotonic()

        result = await llm.act(msg, tool_schemas, model=model)

        if result.stop_reason == "max_tokens":
            # Truncated — warn and continue so the model can finish
            import warnings
            warnings.warn(f"agent step {step + 1}: response truncated (max_tokens)")
            _emit(FlowEvent("error", step_label, 1, result="truncated (max_tokens)"))
            # Append what we got so the model can continue from it
            if result.text:
                msg = msg | assistant(result.text)
            continue

        if result.done:
            # Model returned final text — natural finish
            elapsed = time.monotonic() - t0
            _emit(FlowEvent("complete", step_label, 1,
                             result=_truncate(result.text or ""), elapsed=elapsed))
            total = time.monotonic() - t_total
            _emit(FlowEvent("complete", label, 0, elapsed=total,
                             result=_truncate(result.text or ""),
                             meta={"steps": step + 1, "signal": None}))
            return AgentResult(
                output=result.text or "",
                signal=None,
                msg=msg,
                steps=step + 1,
            )

        # Preserve assistant narration before tool calls (e.g., "Let me search for that")
        if result.text:
            msg = msg | assistant(result.text)

        # Process tool calls
        for call in result.tool_calls:
            call_label = f"{call.name}"
            _emit(FlowEvent("start", call_label, 2, meta={"tool_id": call.id}))
            tc0 = time.monotonic()

            # Append the tool_use segment
            msg = msg | tool_use(call.id, call.name, call.input)

            # Check if it's a signal tool
            if call.name in signal_tools:
                signal = signal_tools[call.name]
                # Still call the handler if one exists
                if call.name in tools:
                    output = await tools[call.name](call.input)
                else:
                    output = str(call.input)

                tc_elapsed = time.monotonic() - tc0
                _emit(FlowEvent("complete", call_label, 2,
                                 result=f"SIGNAL:{signal} {_truncate(output)}",
                                 elapsed=tc_elapsed))

                msg = msg | tool_result(call.id, output)
                total = time.monotonic() - t_total
                _emit(FlowEvent("complete", step_label, 1, elapsed=time.monotonic() - t0))
                _emit(FlowEvent("complete", label, 0, elapsed=total,
                                 result=f"signal: {signal}",
                                 meta={"steps": step + 1, "signal": signal}))
                return AgentResult(
                    output=output,
                    signal=signal,
                    msg=msg,
                    steps=step + 1,
                )

            # Regular tool — execute and append result.
            # Server-side tools (e.g., Anthropic's web_search) are resolved by
            # the API before we see the response — they never appear as tool_calls.
            # If Anthropic changes this behavior, unknown tool names will get an
            # error tool_result, which the model can recover from.
            if call.name not in tools:
                output = f"Error: unknown tool '{call.name}'"
                msg = msg | tool_result(call.id, output, is_error=True)
                tc_elapsed = time.monotonic() - tc0
                _emit(FlowEvent("complete", call_label, 2,
                                 result=f"error: {output}", elapsed=tc_elapsed))
                continue

            try:
                handler = tools[call.name]
                output = await handler(call.input)
            except Exception as e:
                output = f"Error: {e}"
                msg = msg | tool_result(call.id, output, is_error=True)
                tc_elapsed = time.monotonic() - tc0
                _emit(FlowEvent("error", call_label, 2,
                                 result=str(e), elapsed=tc_elapsed))
                continue

            msg = msg | tool_result(call.id, str(output))
            tc_elapsed = time.monotonic() - tc0
            _emit(FlowEvent("complete", call_label, 2,
                             result=_truncate(str(output)), elapsed=tc_elapsed))

        elapsed = time.monotonic() - t0
        _emit(FlowEvent("complete", step_label, 1, elapsed=elapsed,
                         meta={"tool_calls": len(result.tool_calls)}))

    # Max steps reached
    total = time.monotonic() - t_total
    _emit(FlowEvent("complete", label, 0, elapsed=total,
                     result=f"max steps ({max_steps})",
                     meta={"steps": max_steps, "signal": "max_steps"}))
    return AgentResult(
        output=result.text or "",
        signal="max_steps",
        msg=msg,
        steps=max_steps,
    )
