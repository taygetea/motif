# Context: motif codebase review

## What this project is

`motif` (PyPI: `motif-llm`) is a ~2000-line prompt algebra library for LLM orchestration.
Standalone publishable package living inside the `regulatedconversation` repo. The README.md
is the **canonical design document** — deviations between README claims and actual code are
findings, not nitpicks.

## Stated design goals (from README + CLAUDE.md — audit against these)

1. **The representation is the thing.** Msg is an immutable sequence of typed segments.
   ToolCall has real `.id`/`.name`/`.input` attributes — never structured data encoded in strings.
2. **Msg is a monoid** — `|` associative, `Msg()` identity. Block is a second monoid.
   `render()` is a homomorphism and the ONLY function that knows API formats
   (backends: anthropic, openai). Property tests enforce this.
3. **Strict layering**: prompt.py (zero deps) ← llm.py (anthropic SDK) ← flow.py.
   Lower layers never import higher ones. graph.py/show.py/display.py/tui.py are display-side.
4. **Purity**: flow patterns are pure functions over Msgs, no caller-state mutation.
   All side effects (display/logging/tracing) via observers. Observers observe, never intervene.
5. **Nesting "just works"**: no shared agent context; each pattern builds its own Msg;
   graph.py contextvar parents nodes automatically. No mutable shared state across patterns.
6. **Compaction is invisible** and never splits tool_use/tool_result pairs, preserves system segments.
7. **Cheap topology / quality content**: `flow._CHEAP` (haiku) for structural decisions,
   DEFAULT_MODEL for content.
8. **API conventions**: `title=` not `label=`; Python 3.12+ (PEP 695, slots=True frozen dataclasses);
   no new top-level deps; prompt.py stays zero-dep.
9. **README code samples must stay current with public API** (README is load-bearing).

## Current state / recent churn

Recent commits added: computation graph (graph.py), show.py display components, graph-driven
TUI (tui.py), `flow.call` helper, branch `label_key`, deep_research_v2 example. README's
"What's in the box" does NOT yet mention graph.py/show.py/tui.py and line counts have drifted
(flow.py is now 1138 lines vs "~980" claimed). Test count claims ("90 tests") may be stale.

## File sizes (actual)

- src/motif: __init__ 38, prompt 338, llm 452, flow 1138, graph 155, show 248, display 388, tui 377
- tests: test_algebra 273, test_compaction 177, test_flow 315, test_flow_advanced 407,
  test_render_edges 114, test_trace 109

## Priorities for this review

Bugs and design-goal violations first; then API inconsistencies and doc drift; then test gaps
(graph.py, show.py, tui.py appear to have NO dedicated tests). Repo hygiene last
(stray research-*.md / robotics_research.md / .hypothesis/ in repo root).

## Constraints

- Don't propose new top-level dependencies.
- Don't propose merging the three display layers (graph nodes / show components / FlowEvents) —
  their separation is intentional.
- Run full test suite before any changes to prompt.py, flow.py, llm.py (load-bearing).
- Commit granularity: one logical change per commit.
