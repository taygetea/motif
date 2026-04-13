# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`motif` (PyPI: `motif-llm`) is a ~2000-line prompt algebra library for LLM orchestration. It is a standalone, publishable package, distinct from the parent `regulatedconversation` project — do not pull in dependencies on its internals. The README.md is the canonical design document; read it before making non-trivial changes.

## Common commands

```bash
uv sync                              # install (incl. dev/display/tui extras as needed)
uv sync --extra dev --extra display  # full dev install

uv run pytest                        # run all 90 tests
uv run pytest tests/test_algebra.py  # one file
uv run pytest tests/test_flow.py::test_branch -xvs   # one test

uv run python examples/dialectic.py            # smoke-test the library end-to-end
uv run python examples/prism.py                # exercises live display
uv run python examples/deep_research.py "topic"
```

Set `ANTHROPIC_API_KEY` in `.env` (loaded automatically by `llm.py`).

## Architecture (the three layers)

The library is intentionally thin and layered. Changes should respect the layering — lower layers never import from higher ones.

```
src/motif/
    prompt.py    Layer 1 — Msg, Block, segments, render(). Zero dependencies.
    llm.py       Layer 2 — complete / extract / act. Anthropic SDK + observers.
    flow.py      Layer 3 — 9 flow patterns + compaction + events.
    graph.py     Computation graph (Node) built via contextvar — used by flow + tui.
    show.py      Display components + renderers (MarkdownRenderer, etc.).
    display.py   Trace + LiveFlowDisplay (rich-based, optional extra).
    tui.py       Textual-based TUI (optional extra).
```

### Layer 1: `prompt.py` — the monoid

A `Msg` is an immutable sequence of typed `Segment`s (`TextSegment`, `ToolCall`, `ToolResult`). `|` concatenates segment sequences. `Block` is a `str` subclass whose `+` joins with paragraph separators and drops empties. `render(msg, backend=...)` is the **only** function that knows about API formats — it is a homomorphism from Msg to provider payloads (currently `anthropic` and `openai`). Segment types are frozen dataclasses; never mutate them. Property-based tests in `test_algebra.py` enforce monoid laws and render homomorphism — when adding segment kinds or composition rules, run these.

### Layer 2: `llm.py` — three verbs

`complete()` returns text, `extract()` returns structured data via schema, `act()` returns text-or-tool-calls. All take a `Msg` and call `render()` internally. Observers attached via `llm.observe(callback)` receive `(verb, msg, result, model, meta)` for every call — the pipeline stays pure. `CostTracker` is one such observer; pricing table is in `_PRICING` (keep current).

`DEFAULT_MODEL` is the workhorse; `flow._CHEAP = "claude-haiku-4-5"` is reserved for cheap structural decisions (judging, branching). This split is intentional — topology should be cheap, content should be high-quality.

### Layer 3: `flow.py` — nine patterns

`branch`, `fan`, `reduce`, `best_of`, `cascade`, `tree`, `tournament`, `blackboard`, `agent`. Eight have predetermined topology; `agent` generates topology at runtime via tool-use loops. All patterns:

1. Build a `Node` in the computation graph (contextvar in `graph.py`) so nesting is automatic — no executor to thread through.
2. Emit `FlowEvent`s to module-level `_observers` for backward-compat tracing.
3. Are pure functions over `Msg`s — they never mutate caller state.

`compact()` automatically summarizes older turns when `agent()` exceeds token thresholds; it preserves system segments and tool_use/tool_result pairs (never splits them). When editing `agent` or `compact`, run `test_compaction.py`.

### Why nesting "just works"

`flow.blackboard` inside an `agent` tool handler composes without special handling because each pattern builds its own `Msg` from scratch — there is no shared "agent context". The contextvar in `graph.py` makes child nodes attach to the right parent automatically. Preserve this property: do **not** introduce mutable shared state across patterns, and do not pass the agent's `Msg` into nested flow calls.

### Observers vs. graph vs. show — three display layers

- **Graph nodes** (`graph.py`): the complete structural record. TUI polls `node.output` growth via `_version` for live streaming.
- **`flow.show()` components** (`show.py`): the curated narrative the pipeline author chose to emit. Renderers (e.g., `MarkdownRenderer`) consume these.
- **`FlowEvent` observers** (`flow.py`): legacy event stream for `Trace` and `LiveFlowDisplay`.

These serve different needs — keep them separate. The pipeline code stays pure; all display, logging, and tracing happens through observers.

## Conventions to preserve

- **The representation is the thing.** A `ToolCall` has real `.id`/`.name`/`.input` attributes — never encode structured data into strings.
- **`title=` not `label=`.** `_check_label_kwarg` exists to catch the old name; do not reintroduce `label=` as a parameter name on flow functions.
- **No new top-level dependencies** without strong justification. `prompt.py` must remain zero-dep. `display`/`tui` extras stay optional.
- **The library targets Python 3.12+** (uses PEP 695 `type` aliases and `slots=True` frozen dataclasses).
- **The README is load-bearing documentation** — when you change public API, update the README's code samples too.

## Testing

90 tests across 6 files, including Hypothesis property-based tests for monoid laws and render integrity. Always run the full suite before committing changes to `prompt.py`, `flow.py`, or `llm.py` — these are the load-bearing modules. `test_flow_advanced.py` covers nested patterns and the compose-without-debugging property.

## Git workflow

Commit after every logical change. Keep commits granular and descriptive (the recent history in `git log` is a good style reference).
