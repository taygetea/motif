# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`motif` (PyPI: `motif-llm`) is a ~3,800-line prompt algebra library for LLM orchestration. It is a standalone, publishable package, distinct from the parent `regulatedconversation` project (which now depends on it — the engine's fork was retired 2026-07-08). The README.md is the canonical design document **and the system prompt for model authors writing pipelines** — read it before non-trivial changes, and update it when the public API moves. ROADMAP.md holds the direction (the instrument / "real sci-fi interface" vision) and the open-findings ledger.

## The design standard

**The computation-graph author has no mistakes available to make.** The expected author is a model writing a pipeline fresh each time — the library is the prompt. Every change gets tested against: *what mistake does this make possible?* Prefer making errors unrepresentable (algebra, immutability, laziness); where impossible, fail loudly and early with the fix in the message (`_check_label_kwarg` is the house style).

## Common commands

```bash
uv sync --extra dev --extra display --extra tui   # full dev install

uv run pytest                        # all 171 tests
uv run pytest tests/test_algebra.py  # one file
uv run pytest tests/test_flow.py::test_branch -xvs

uv run python examples/dialectic.py               # smoke-test end-to-end
uv run python examples/deep_research_v2.py "topic" --profile deepseek
    # full pipeline on the subsidized endpoint (~$0.15) — needs
    # OPENROUTER_API_KEY and EXA_API_KEY in .env
```

Keys in `.env` (loaded automatically by `llm.py`): `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `EXA_API_KEY`.

## Architecture

Lower layers never import from higher ones. (One documented exception: `llm.stream()` appends chunks to the current graph node — Layer 2 touching graph.py for live display. Review finding B2; invert via observer if it grows.)

```
src/motif/
    prompt.py    Layer 1 — Msg, Block, segments, render(). Zero dependencies.
    llm.py       Layer 2 — complete/extract/act, Endpoint, RoleRef/role/use_profile,
                 Anthropic SDK + OpenAI-compatible httpx transport, CostTracker.
    flow.py      Layer 3 — 9 patterns + call + compaction + agent (finalize, signal
                 tools) + FlowEvents (legacy, slated for retirement).
    graph.py     Computation graph: Node, contextvar nesting, graph.session()
                 (per-run root scoping — always use a session for tests/servers).
    show.py      Salience policy + narrate() (graph→markdown fold) + display
                 components + MarkdownRenderer.
    display.py   Trace + LiveFlowDisplay (rich, optional; consumes legacy FlowEvents).
    tui.py       Textual TUI (optional; polls node._version).
```

### Layer 1: `prompt.py` — the monoid

`Msg` is an immutable sequence of typed `Segment`s; `|` concatenates. `Block` is a str subclass whose `+` paragraph-joins and drops empties. `render(msg, backend=...)` is the **only** function that knows API formats (anthropic / openai / flat) — a homomorphism. Property tests in `test_algebra.py` enforce the monoid laws **including mixed str/Block operands** (`__radd__` must agree with `__add__` — a real bug lived in that blind spot). Never mutate segments; run the algebra tests for any change here.

### Layer 2: `llm.py` — verbs, endpoints, roles

Three verbs take `str | Endpoint | RoleRef` as `model`. `Endpoint(model, base_url, key_env, extra)`: `base_url=None` → Anthropic SDK; any base_url → OpenAI-compatible `/chat/completions` via httpx (transitively available — **no new top-level deps**). RoleRefs resolve lazily at call time against `use_profile()` bindings; defaults: `content` → DEFAULT_MODEL, `structure` → DEFAULT_CHEAP_MODEL.

Rules of the house:
- **Never hardcode a model id where a role belongs.** Cost is a property of the run, not the program. Pipeline code names roles; profiles name models.
- `extract()` on the openai transport degrades json_schema → json_object+prompt → bare prompt; keep that invisible.
- `act()` maps finish_reason to anthropic stop_reason vocabulary (`"length"` → `"max_tokens"`) — agent() depends on it.
- Observer meta may carry `reported_cost` (OpenRouter actual billing) — CostTracker prefers it over the `_PRICING` table. Keep `_PRICING` current.
- Transport tests live at the mocked-HTTP boundary (`httpx.MockTransport`) in `test_llm.py` — test real parsing code, not mocks of our own functions.

### Layer 3: `flow.py` — patterns

All patterns: build a graph Node (contextvar parenting — nesting is automatic), emit legacy FlowEvents, stay pure over Msgs. Model defaults are `llm.role("structure")` (branch, best_of, cascade test, tree split, tournament) or `llm.role("content")` (the rest) — resolved at call time. Every pattern takes `show="shown|collapsed|hidden"` (author display override) and stamps model *labels* into node meta (`_model_label`: RoleRef → `"role:<name>"`) — the salience policy reads these.

`agent()`: compaction is invisible and never splits tool pairs, even non-adjacent (`_compact_split`, pure, tested directly — don't reintroduce a hand-copied version in tests); all tool handlers including signal tools convert exceptions to `is_error` results; `finalize=True` forces a tools-free closing call when max_steps exhausts mid-search.

`title=` not `label=`; labels from data via `label_key=` where possible.

### Graph, sessions, and the fold

`graph.session()` scopes a run's roots (contextvar; concurrent runs don't interleave; tests get an autouse session via conftest.py). `show.narrate(roots)` folds the graph to markdown by salience policy — kind/role/cardinality/error decide, `meta["show"]` overrides, embedded headings demote, errors always surface. narrate is a homomorphism like render: **new output formats are new folds over the same graph**, not new event streams. The three display layers (graph / show components / FlowEvents) are separate on purpose; FlowEvents are legacy and will be retired onto the graph — don't build new features on them.

## Conventions to preserve

- **The representation is the thing.** Real typed attributes, never data-in-strings.
- **No new top-level dependencies** without strong justification. prompt.py stays zero-dep; display/tui stay optional extras.
- **Python 3.12+** (PEP 695 type aliases, slots=True frozen dataclasses).
- **README samples must run.** The README is load-bearing twice over: design doc and author system-prompt.
- Examples are dogfood: they use roles + `--profile`, never hardcoded model ids, and produce their documents via `narrate` (deep_research_v2 has zero display code — keep it that way).

## Testing

171 tests across 8 files. Always run the full suite before committing changes to `prompt.py`, `flow.py`, `llm.py`, `graph.py`, or `show.py`. Known gaps (see ROADMAP): `best_of`, `tournament`, `flow.call`, `label_key`, show components, TUI internals.

## Git workflow

Commit after every logical change. Keep commits granular and descriptive (recent `git log` is the style reference).
