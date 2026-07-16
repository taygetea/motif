# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`motif` (PyPI: `motif-llm`) is a ~4,400-line prompt algebra library for LLM orchestration. It is a standalone, publishable package, distinct from the parent `regulatedconversation` project (which now depends on it — the engine's fork was retired 2026-07-08). The README.md is the canonical design document **and the system prompt for model authors writing pipelines** — read it before non-trivial changes, and update it when the public API moves. ROADMAP.md holds the direction (the instrument / "real sci-fi interface" vision) and the open-findings ledger.

## The design standard

**The computation-graph author has no mistakes available to make.** The expected author is a model writing a pipeline fresh each time — the library is the prompt. Every change gets tested against: *what mistake does this make possible?* Prefer making errors unrepresentable (algebra, immutability, laziness); where impossible, fail loudly and early with the fix in the message (`_check_label_kwarg` is the house style).

## Common commands

```bash
uv sync --extra dev --extra display --extra tui   # full dev install

uv run pytest                        # all 233 tests
uv run pytest tests/test_algebra.py  # one file
uv run pytest tests/test_flow.py::test_branch -xvs

uv run python examples/dialectic.py               # smoke-test end-to-end
uv run python examples/deep_research_v2.py "topic" --profile deepseek
    # full pipeline on the subsidized endpoint (~$0.15) — needs
    # OPENROUTER_API_KEY and EXA_API_KEY in .env
```

Keys in `.env` (loaded automatically by `llm.py`): `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `EXA_API_KEY`.

## Architecture

Lower layers never import from higher ones. (The old B2 exception — `llm.stream()` mutating the current graph node — was eliminated 2026-07-16: llm.py no longer imports graph at all; record.py injects the projection and llm's observer-scope registration downward via `llm._projection` / `graph._register_scoped`.)

```
src/motif/
    prompt.py    Layer 1 — Msg, Block, segments, render(). Zero dependencies.
    llm.py       Layer 2 — complete/extract/act/stream, Endpoint,
                 RoleRef/role/use_profile, call-lifecycle events
                 (CallStarted/CallChunk/CallCompleted/CallFailed — per-call
                 identity, retained input Msg) with observe_calls() and the
                 legacy observe() five-tuple derived by a stateful adapter,
                 Anthropic SDK + OpenAI-compatible httpx transport,
                 CostTracker. Knows nothing about the graph.
    record.py    The llm↔graph bridge — projects call events into graph
                 nodes (kind "llm_call", node id == call_id, Node.msg
                 retained, usage in meta, attachments preserved).
                 Installed at import through the llm._projection slot;
                 exactly one projection may exist.
    delegated.py Leaf module (stdlib only, no motif imports) — the codex
                 CLI adapter behind DelegatedEndpoint. The hermetic
                 invocation is load-bearing: --ephemeral
                 --ignore-user-config -s read-only AND -c mcp_servers={}
                 (without the last, the harness has live remote-write
                 app connectors — notes/2026-07-16-canary-results.md).
                 Prompt via stdin ("-"), own process group, deadline
                 kill; JSONL is adapter protocol, CLI version
                 fingerprinted on every result.
    flow.py      Layer 3 — 9 patterns + group + call + compaction + agent
                 (finalize, signal tools) + FlowEvents (legacy, slated for
                 retirement).
    graph.py     Computation graph: Node, contextvar nesting, graph.session()
                 (per-run scoping of roots AND observer registration; always
                 use a session for tests/servers). attach() places leaf
                 records without making them current.
    show.py      Salience policy + narrate() (graph→markdown fold) + display
                 components + MarkdownRenderer.
    display.py   Trace + LiveFlowDisplay (rich, optional; consumes legacy FlowEvents).
    tui.py       Textual TUI (optional; polls node._version; reads streamed
                 output through to llm_call record children).
```

### Layer 1: `prompt.py` — the monoid

`Msg` is an immutable sequence of typed `Segment`s; `|` concatenates. `Block` is a str subclass whose `+` paragraph-joins and drops empties. `render(msg, backend=...)` is the **only** function that knows API formats (anthropic / openai / flat) — a homomorphism. Property tests in `test_algebra.py` enforce the monoid laws **including mixed str/Block operands** (`__radd__` must agree with `__add__` — a real bug lived in that blind spot). Never mutate segments; run the algebra tests for any change here.

### Layer 2: `llm.py` — verbs, endpoints, roles

Three verbs take `str | Endpoint | RoleRef` as `model`. `Endpoint(model, base_url, key_env, extra)`: `base_url=None` → Anthropic SDK; any base_url → OpenAI-compatible `/chat/completions` via httpx (transitively available — **no new top-level deps**). RoleRefs resolve lazily at call time against `use_profile()` bindings; defaults: `content` → DEFAULT_MODEL, `structure` → DEFAULT_CHEAP_MODEL.

Rules of the house:
- **Never hardcode a model id where a role belongs.** Cost is a property of the run, not the program. Pipeline code names roles; profiles name models.
- `extract()` on the openai transport degrades json_schema → json_object+prompt → bare prompt; keep that invisible.
- `act()` maps finish_reason to anthropic stop_reason vocabulary (`"length"` → `"max_tokens"`) — agent() depends on it. Truncation raises `Truncated` from complete/stream (opt out with allow_truncation) and unconditionally from extract; act() returns it as stop_reason instead.
- Every verb emits its lifecycle exactly once: CallStarted, chunks, then one of CallCompleted/CallFailed — even on truncation (Completed carries stop_reason, then the verb raises) and stream abandonment (Failed). Preserve that invariant in any transport change; cost visibility depends on it.
- Observer meta may carry `reported_cost` (OpenRouter actual billing) — CostTracker prefers it over the `_PRICING` table (matching is exact-or-suffix, never bare prefix). Keep `_PRICING` current.
- Registration is scope-local, emission additive: observers attached inside a graph.session() detach with it; process-global observers see every run. clear_observers() clears only the active scope and can never silence the graph projection.
- Delegated endpoints serve complete/extract only, one semantic call node per author operation under every profile (recording granularity follows effect radius, not step count). flow.agent NEVER routes to a harness. requires= preflight fails before spending. Never weaken the hermetic argv.
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

233 tests across 11 files. Always run the full suite before committing changes to `prompt.py`, `flow.py`, `llm.py`, `graph.py`, `record.py`, or `show.py`. Known gaps (see ROADMAP): show components, TUI internals, the anthropic streaming path (the SDK fake has no `.stream()`).

## Git workflow

Commit after every logical change. Keep commits granular and descriptive (recent `git log` is the style reference).
