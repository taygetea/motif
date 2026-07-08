# Findings: motif whole-codebase review (2026-07-08)

## STATUS LEDGER (updated end of day 2026-07-08)

| finding | status |
|---------|--------|
| A1 openai render None-content crash | FIXED e4b4799 |
| A2 Block.__radd__ identity violation | FIXED 590ecdc (+ mixed-operand property tests) |
| A3 compact() splits non-adjacent pairs | FIXED 8e95a7f (tests now use the real _compact_split) |
| A4 unguarded signal-tool path | FIXED 3ef654a |
| A5 backend="openai" silently broken | FIXED 277425b (kwarg removed; real transport added) |
| B1 global graph/show roots | FIXED d29fe80 (graph.session; show observers still global — see B1b) |
| B1b observer lists still module-global | OPEN — the Runtime seam (ROADMAP scale 1) |
| B2 llm→graph coupling in stream() | ACCEPTED + documented in CLAUDE.md; invert via observer if it grows |
| C1 CostTracker prefix false-positive | OPEN (mitigated: reported_cost preferred when present) |
| C2 extract() truncation legibility | OPEN |
| D1 llm.py zero direct coverage | FIXED (test_llm.py, 29 tests at mocked-HTTP boundary) |
| D2 untested flow surface | OPEN: best_of, tournament, flow.call, label_key |
| D3 graph/show/tui untested | PARTIAL: graph sessions + narrate + policy tested; show components + tui still open |
| D4 property-test blind spot | FIXED (mixed-operand strategies) |
| E doc drift | FIXED (README/CLAUDE.md rewritten 2026-07-08; keep them current) |
| F stray files / output paths | FIXED (gitignore + examples/output default) |

New since the review (same standard — author error surface): shape rehearsal,
flow.group, schema ergonomics, nested-Msg convention enforcement, FlowEvent
retirement, TUI-reads-policy. Tracked in ROADMAP.md Scale 1.

---

Three parallel explorers (bugs / spec-adherence / tests+display). Deduped and ranked.
Findings marked **[repro]** were confirmed with executed reproduction scripts, not just code reading.

## A. Confirmed correctness bugs

### A1. [repro] `render(backend="openai")` crashes on assistant text after a tool call
`src/motif/prompt.py:291-295` — the merge-into-previous-message branch does
`messages[-1]["content"] += "\n\n" + text` without checking for `None` content; a message
created purely from a `ToolCall` has `content: None` (OpenAI shape).
Repro: `render(tool_use('t1','search',{'q':'x'}) | assistant('text'), backend='openai')`
→ `TypeError`. This is a hole in the "render is a homomorphism" guarantee for the openai backend.

### A2. [repro] `Block.__radd__` violates the monoid identity law
`src/motif/prompt.py:74-81` — `__radd__` checks the left (str) operand for emptiness but not
`self`: `"prefix" + Block("")` → `Block('prefix\n\n')` instead of `'prefix'`.
The Hypothesis tests never exercise `__radd__` (both operands are always Blocks), so the
property suite has a blind spot exactly where the law breaks. Block is one of the two
advertised monoids — this is a direct violation of the core algebra claim.

### A3. [repro] `compact()` can split a non-adjacent tool_use/tool_result pair
`src/motif/flow.py:213-238` — docstring promises pairs are never split, but the backward walk
only inspects the single segment adjacent to the split point. A ToolCall separated from its
ToolResult by intervening text gets split; the resulting Msg produces an Anthropic 400
(orphaned tool_result). Doesn't manifest via `agent()`'s own adjacent-append usage, but
`compact()` is public and documented as general-purpose.

### A4. `agent()` signal-tool path has no exception handling
`src/motif/flow.py:1061-1088` vs guarded regular-tool path at 1101-1111 — a raising signal-tool
handler (FINISH/DELEGATE/ESCALATE) crashes the whole loop and leaves `tool_node`/`step_node`
permanently `state="running"` in the graph, instead of becoming an `is_error` tool_result like
every other tool failure. No test exercises signal-tool failure.

### A5. `backend="openai"` on the llm verbs is silently broken
`src/motif/llm.py:196,217` (and mirrored in extract/act) — all three verbs accept `backend=`
and render an OpenAI-shaped payload, then send it through the hardcoded
`anthropic.AsyncAnthropic()` client. Zero mentions of an OpenAI client in llm.py; zero tests.
**Open question for Olivia:** incomplete feature (wire a real OpenAI client) or accidental
kwarg threading (remove/gate the kwarg)?

## B. Design-goal violations (all three explorers converged on B1)

### B1. Global mutable state in graph.py/show.py; `Trace.graph` aliases it
`src/motif/graph.py:90` (`_root_nodes`), `src/motif/show.py:163` (`_show_observers`),
`src/motif/display.py:109-117` (`Trace.graph` = passthrough to `graph.root_nodes()`).
`_current_node` is correctly a ContextVar (bugs explorer specifically traced all four
gather/TaskGroup patterns and found NO race there), but the root-node registry and show
observers are plain module lists shared process-wide. Consequences:
- Two concurrent top-level pipelines interleave root nodes/components (the exact shape
  deep_research_v2.py uses).
- `Trace.graph` returns every run's roots, not "this trace's graph" as documented.
- Nodes accumulate forever in long-running processes unless `graph.reset()` is called.
Evidence it's a known pain: ~28-30 manual `graph.reset()` calls across the flow tests and
`tui.py:284`. Direct violation of the stated "no mutable shared state across patterns" goal.
Fix direction: per-run scoping (ContextVar or snapshot-at-Trace-construction), plus an
autouse pytest fixture replacing the manual resets. README already names this seam
("where an explicit Runtime object would go").

### B2. Layering violation: llm.py (Layer 2) imports graph.py
`src/motif/llm.py:21,277-283` — `stream()` does `node.append_output(text)` directly.
CLAUDE.md classifies graph as "used by flow + tui" and llm.py as SDK+observers only.
Either document as an accepted exception or invert: stream stays graph-agnostic, a
graph-aware observer appends. (Interacts with B1 — same subsystem.)

## C. Robustness (medium)

### C1. CostTracker prefix matching can false-positive
`src/motif/llm.py:96-100` — `model.startswith(name) or name.startswith(model)`; a short
alias can match a more specific unrelated pricing entry. Telemetry-only impact. No unit
tests verify dollar amounts.

### C2. `extract()` has no truncation/malformed-output handling
`src/motif/llm.py:322-335` — never inspects `stop_reason` (unlike `act()`); max_tokens
truncation surfaces as a bare `json.JSONDecodeError`.

## D. Test gaps

### D1. llm.py's real implementation has zero direct coverage
All flow tests mock at `motif.flow.llm.*`. Response parsing, `_usage()` mapping, extract's
retry loop, act's ToolRequest construction, CostTracker math: untested. Test against a mocked
Anthropic SDK client boundary instead.

### D2. Untested flow surface
`flow.call` (new), `branch(label_key=)` (new), `best_of`, `tournament` — no tests at all.
Two of the nine advertised patterns are uncovered.

### D3. graph.py / show.py / tui.py have no test files
Worth covering: Node state machine + `exit_node` overwrite guard (graph.py:135-137),
`Panels.__post_init__` validation (show.py:90-93), MarkdownRenderer Progress replace-in-place
(show.py:230-237), tui `_visit()` fan-child special-casing (tui.py:332-338). Node and show
components are pure data — cheap to test.

### D4. Property-test blind spot
test_algebra.py never constructs `str + Block` (see A2). Extend Hypothesis strategies to
mixed-operand cases.

## E. Doc drift (README is load-bearing per CLAUDE.md)

- "What's in the box" omits graph.py (155), show.py (248), tui.py (377); listed line counts
  stale; actual total ~3134 vs claimed "~2000" (README.md:203-213).
- No README coverage of `flow.call`, `branch(label_key=)`, the entire show.py component
  system, the `[tui]` extra (pyproject defines it; tui.py's docstring references it), or the
  `flat` render backend (prompt.py:209-230).
- Examples section lists 4 of 10 scripts; deep_research_v2.py (newest, different pattern)
  unlisted.
- Verified NOT drifted: "90 tests" (exact), "five constructors" (exact), all README samples
  execute, no title=/label= violations, no stringly-typed data.

## F. Repo hygiene

- Stray root files from a deep_research_v2 run: `research-llm-driven-robotics-*.md/.trace.json`,
  `robotics_research.md`, `.hypothesis/`. None committed yet; one `git add .` away.
- Root cause: `deep_research_v2.py:500-502` `default_output_path()` writes to CWD instead of
  `examples/output/`. Fix both: .gitignore entries (`research-*.md`, `*.trace.json`,
  `.hypothesis/`) + change the default output dir.
