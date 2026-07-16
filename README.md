# motif

A prompt algebra for LLM orchestration. ~3,800 lines. Does what 50,000-line
frameworks do — and the code reads as its own specification.

> *The thing about encoding frustration into a library is that if the
> frustration was correct — if it was pointed at real problems and not just
> taste — the result is something that feels inevitable.* — Claude

## The design standard

Motif is built for a specific author: **a model writing a pipeline fresh,
every time.** No IDE, no debugger, no accumulated familiarity — the library
is the prompt, and every API decision is a prompt-engineering decision. So
the standard is stricter than "documented" or "intuitive":

**The computation-graph author should have no mistakes available to make.**

- Composition is a monoid — `|` is associative, there is no invalid order,
  no "call X before Y" protocol to violate.
- Everything is immutable — no state to corrupt, no action at a distance.
- Five constructors, one operator, three verbs — the whole language fits in
  one context window, and it coincides with mathematical objects (monoid,
  fold, homomorphism) a model already knows from its priors. The code the
  author would guess is the code that works.
- What a call is *for* is declared (roles), never priced (model ids) — the
  author cannot pick a wrong cost.
- Display falls out of the computation graph by policy — the author cannot
  build a broken interface, only override a good default.
- Mistakes that can't be made unrepresentable fail loudly and early
  (`label=` raises with the correct spelling; unbound roles name the fix).

Every change gets held against this test: *what mistake does it make
possible?*

```python
from motif import system, user, flow, llm

# Discover research angles, investigate each in parallel,
# discuss across researchers, synthesize a report
angles = await flow.branch(decomposer | user(topic), schema=ANGLES,
                           title="decompose", label_key="name")
briefs = await flow.fan(angles, lambda a: researcher | user(a["question"]),
                        title="research")
board, discussion = await flow.blackboard(researchers, seed=findings,
                                          rounds=2, title="panel")
report = await flow.call(synthesizer | user(board), title="synthesis",
                         model=llm.role("content"))
```

This is a real pipeline — [deep_research_v2.py](examples/deep_research_v2.py)
runs reconnaissance, adversarial critique, and synthesis with genuine
multi-step web-searching agents, and its output document is produced entirely
by folding the computation graph (no display code in the pipeline at all).
On a subsidized endpoint the whole run costs about **$0.15**. The same
library composes [philosophical debates](examples/dialectic.py),
[expert panels](examples/blackboard.py), and — in the parent project — full
psychologically-modeled conversations with synthetic nervous systems.

## What makes it different

Most orchestration libraries build elaborate runtime machinery — chains,
agents, memory systems, callback managers — and the actual prompt
composition is an afterthought. Motif inverts this: **the representation is
the machinery.**

A `Msg` is an immutable sequence of typed segments. `|` composes them. The
type is a monoid — composition is associative, branching is free, and flow
patterns are just functions over messages. Agents aren't a framework
concept; they're what happens when tool results feed back into the Msg.
The immutability is deep where it matters: `tool_use()` copies its input
dict, so mutating your dict later — or a tool handler that `pop()`s its
arguments — can never rewrite what a Msg records as having been invoked.

```python
from motif import system, user, assistant, tool_use, tool_result

# Build prompts with |
prompt = system(persona, cache=True) | user(context)

# Multi-turn: just keep composing
prompt = prompt | assistant(response) | user(followup)

# Tool calls: same algebra
prompt = prompt | tool_use("id", "search", {"q": "..."}) | tool_result("id", "found it")

# The Msg grew. The monoid held. No special API for any of this.
```

## The layers

### Layer 1: Prompt composition (`prompt.py`)

Five constructors, one operator. Provider-agnostic. Zero dependencies.

```python
from motif import system, user, assistant, tool_use, tool_result, Block, render

# | composes Msgs
prompt = system("persona", cache=True) | system("felt world") | user("context")

# Block composes text within a segment (drops empties, paragraph-joins)
context = Block(signature) + Block(history) + Block(latest)

# Block.join composes results for synthesis (optionally labeled)
combined = Block.join(analyses, labels=["rhetoric", "logic", "psychology"])

# render() at the boundary — the only function that knows about API formats
render(prompt, backend="anthropic")   # system content blocks, cache_control
render(prompt, backend="openai")      # system message + turns, tool_calls
render(prompt, backend="flat")        # plain system/prompt strings (llm CLI etc.)
```

Property-based tests (Hypothesis) enforce the monoid laws — including the
mixed-operand laws, because `__add__`/`__radd__` are one operation
implemented twice and must be pinned to each other.

### Layer 2: LLM calls (`llm.py`)

Three verbs. Msg in, value out.

```python
from motif import llm

text = await llm.complete(prompt)                       # text out
data = await llm.extract(prompt, schema=MY_SCHEMA)      # structured data out
result = await llm.act(prompt, tools=TOOL_SCHEMAS)      # text or tool calls out
```

**Endpoints.** A model bound to the place it runs. `base_url=None` is the
Anthropic SDK transport; any `base_url` is an OpenAI-compatible
`/chat/completions` endpoint — OpenRouter, a local llama.cpp server, vLLM.
All four verbs (including SSE streaming) dispatch on it.

```python
from motif import Endpoint

DEEPSEEK = Endpoint("deepseek/deepseek-v4-pro",
                    base_url="https://openrouter.ai/api/v1",
                    key_env="OPENROUTER_API_KEY",
                    extra={"provider": {"order": ["DeepSeek"]},
                           "reasoning": {"enabled": False}})
LOCAL = Endpoint("gemma-4-26b-a4b", base_url="http://localhost:11500/v1")
```

**Roles.** Cost is a property of the run, not the program. Pipelines name
what a call is *for*; a profile binds names to endpoints per deployment.
RoleRefs are lazy — resolved at call time — so swapping the entire pipeline
between Anthropic, OpenRouter, and a local model is a profile change, never
a code edit.

```python
from motif import llm, role, use_profile

use_profile({
    "structure": "claude-haiku-4-5",     # topology decisions
    "content":   "claude-opus-4-5",      # the actual work
    "swarm":     DEEPSEEK,               # a thousand cheap samples
})

vote  = await llm.extract(prompt, schema=S, model=role("structure"))
brief = await llm.complete(prompt, model=role("content"))
```

Flow patterns default to `role("structure")` for structural decisions
(branching, judging, routing, splitting) and `role("content")` for
generation. The split is intentional — they are different kinds of work,
whatever they happen to cost this month.

The profile binding is a ContextVar, not a process global: `use_profile()`
applies to the current async context and the tasks it spawns, so concurrent
runs bound to different profiles cannot switch each other's models mid-fan.

Robustness the author never thinks about: `extract()` degrades invisibly
from `json_schema` → `json_object` + schema-in-prompt → bare prompt on
endpoints without structured-output support; `act()` maps finish reasons to
one vocabulary so agent loops run identically on every transport;
`CostTracker` believes provider-reported billed dollars (OpenRouter) over
its own pricing table. And silent partial answers don't exist: a response
that hits `max_tokens` raises `Truncated` (partial text on the exception,
fix in the message) from `complete()` and `stream()` alike unless the
caller passes `allow_truncation=True` — and from `extract()`
unconditionally, because a JSON document cut mid-stream is never a valid
partial.

**The call-lifecycle seam.** Every verb invocation emits typed facts:
`CallStarted(call_id, verb, msg, declared, endpoint, params, meta)` →
`CallChunk*` → (`CallCompleted` | `CallFailed`). Each call has its own
identity — five parallel judgments are five facts, identical resamples are
distinguishable — and `CallStarted` retains the actual input `Msg`, so
replay and lineage can read what the call really saw. Attach with
`llm.observe_calls(fn)`; the older `llm.observe(fn)` five-tuple signature
`(verb, msg, result, model, meta)` still works, derived from the same
events. A failed call is a fact too (`CallFailed` carries usage when the
transport billed before failing). You never emit these yourself — the
verbs do it.

### Layer 3: Flow patterns (`flow.py`)

Nine named patterns for multi-call orchestration — eight with predetermined
topology, one that generates topology at runtime — plus `group(title)` to
put a titled section around any work, and `call()` to put an author's title
on a single verb call.

| Pattern | Does |
|---------|------|
| `branch` | One call discovers structure → list of items. The schema declares exactly one top-level array (`items_key=` selects when there are several; `label_key=` derives display labels from the data) |
| `fan` | Items → parallel calls → results (with concurrency control) |
| `reduce` | Results → labeled synthesis → one output |
| `best_of` | Parallel judging → pick the winner |
| `cascade` | Try cheap model first, escalate if insufficient |
| `tree` | Recursive decomposition — split, analyze leaves, merge back up |
| `tournament` | Bracket-style elimination with pairwise comparison |
| `blackboard` | Expert panel with shared state and selective visibility (`filter_fn`) |
| `agent` | Tool-use loop — the Msg grows until the model finishes or a signal tool fires |

The agent loop earns its keep on the failure modes: compaction is automatic
and invisible (tool_use/tool_result pairs are never split, even
non-adjacent ones); a raising tool handler — signal tools included — becomes
an `is_error` result instead of a crash; and when `max_steps` runs out with
the model still reaching for tools, a **finalize turn** strips the tools and
forces a written answer, so the agent's last word is never a half-finished
search.

Patterns are not required. Every verb call records itself into the graph
automatically (an `llm_call` node under whatever is running, with usage and
the input Msg), so raw loops over the verbs trace and display exactly as
well as the blessed shapes. `call()` is an annotation — a title and a
salience override for a call that is recorded anyway — and `group()` gives
a raw loop a titled home:

```python
with flow.group("turn 1"):
    thought = await llm.complete(persona | user(state))
    speech = await llm.complete(persona | user(thought))
# narrate() renders the group as a section, the calls as its content
```

### The graph and the fold (`graph.py`, `show.py`)

Every pattern builds a node in a computation graph via contextvar — nesting
is automatic, there is no executor to thread through. Scope a run with a
session; fold the graph into a document with `narrate`:

```python
from motif import graph
from motif.show import narrate

with graph.session() as s:
    await pipeline()

document = narrate(s.roots)     # markdown, for free
```

`narrate` is to the graph what `render` is to the Msg: a homomorphism into
an output format. Salience is decided by what a node **is** — pattern kind,
declared role, cardinality, error state — with `show="shown|collapsed|hidden"`
on any pattern as the author override:

| default | nodes |
|---------|-------|
| hidden | `compact` (invisible by its own contract); `llm_call` records under a parent that narrates its own output (the record stays in the graph for drill-in) |
| collapsed | `branch` / `best_of` / `cascade` / `tournament` (topology, not content), calls declared `role:structure` |
| shown | fan items, `reduce`, calls, agent finals, blackboard rounds; `llm_call` records at the root or inside output-less containers like `group` — there, the records are the content |
| always | errors surface regardless of policy (a failed call shows through any parent); fans wider than `fan_limit` collapse to a preview list — salience moves from the nodes to the aggregate |

In practice the defaults carry: the deep-research example produces its
entire 90KB output document with **zero** display code and zero overrides.
`flow.show()` components (`Section`, `Panels`, `Chat`, `Code`, `Table`) and
the `MarkdownRenderer` remain as the escape hatch for bespoke curation, and
`Trace` / `LiveFlowDisplay` / the Textual TUI consume the same graph live.

**Observers observe; they don't intervene.** The pipeline stays pure —
display, logging, tracing, cost all attach via `llm.observe_calls()` /
`llm.observe()` / `flow.observe()`. Attach inside a `graph.session()` and
the observer is scoped to that run: it detaches when the session closes,
and concurrent runs cannot see each other's events. Attach outside any
session and it's process-global — it sees every run (a startup-time
`CostTracker` keeps billing whatever sessions come and go). Scripts need no
ceremony; servers get isolation from the same `with` block that scopes the
graph.

## Why it composes

The [capstone example](examples/agent_compose.py) puts flow patterns inside
agent tool handlers — a blackboard discussion running inside a tool call,
inside an agent loop — and it works on the first run:

```
research analyst (agent loop)
  step 1:
    → model calls expert_panel tool
      → blackboard(4 experts × 2 rounds)       ← flow pattern inside tool handler
        → 8 parallel LLM calls
      → returns discussion to agent
  step 2:
    → model searches, writes sourced report
```

The reason this composes without debugging is that there's nothing to
compose. A tool handler is an async function that takes a dict and returns a
string. Inside, `flow.blackboard()` builds its own Msgs from scratch —
independent of the agent's Msg. When the handler returns, the result becomes
a `tool_result` segment. The two histories never interact. They can't
conflict because they're in different scopes.

Do **not** pass the agent's Msg into nested flow calls — each pattern builds
its own context. (This is currently a convention rather than a structural
guarantee; see the roadmap.)

## Why it works

Msg is a monoid: `|` is associative, `Msg()` is the identity. Block is a
second monoid. `render()` is a homomorphism to API payloads; `narrate()` is
a homomorphism to documents. The reason composition feels right is that it
*is* right — the operations mean what they look like they mean because the
underlying structure is algebraic.

**The representation is the thing.** A ToolCall has `.id`, `.name`, `.input`
as real attributes, not data encoded in strings. Labels derive from data
(`label_key=`), not author diligence.

**The call site is the documentation.** `branch → fan → reduce` says what it
does — and because display reads the same call sites, the pipeline code is
also the interface specification.

**Cost is a property of the run, not the program.** Roles in the pipeline,
economics in the profile.

### Two kinds of joining

`Block + Block` — within-segment composition, for building one segment from parts.
`Block.join(items, labels=)` — between-results composition, for presenting
multiple outputs to a synthesis call.

## Installation

```bash
pip install motif-llm              # core
pip install motif-llm[display]     # + rich live terminal display
pip install motif-llm[tui]         # + Textual TUI

# or with uv
uv add motif-llm
```

Installs as `motif-llm`, imports as `motif`. Keys via environment or `.env`
(loaded automatically): `ANTHROPIC_API_KEY` for the Anthropic transport,
plus whatever your Endpoints name in `key_env` (e.g. `OPENROUTER_API_KEY`).

## Examples

```bash
python examples/dialectic.py                    # Nietzsche vs Schopenhauer debate
python examples/prism.py                        # Multi-lens analysis, live display
python examples/blackboard.py                   # Expert panel with shared state
python examples/tree_decomposition.py           # Recursive decompose/merge
python examples/temporal_analysis.py            # Multi-stage document analysis
python examples/tui_demo.py                     # Textual TUI, live graph
python examples/deep_research.py "topic"        # Deep research v1
python examples/deep_research_v2.py "topic" --profile deepseek
    # Reconnaissance + adversarial critique + synthesis. Roles + profiles,
    # real multi-step web agents (client-side Exa tools, examples/websearch.py),
    # output document folded from the graph. ~$0.15 on the deepseek profile.
```

## What's in the box

```
src/motif/
    prompt.py    ~350 lines   Msg, Block, segments, render — zero dependencies
    llm.py      ~1040 lines   three verbs, Endpoint, roles/profiles, call-
                              lifecycle events, Anthropic + OpenAI-compatible
                              transports, CostTracker
    record.py    ~120 lines   folds call events into graph nodes
    flow.py     ~1430 lines   9 patterns + group + call + compaction + agent
    graph.py     ~240 lines   computation graph, contextvar nesting, sessions
    show.py      ~480 lines   salience policy, narrate fold, display components
    display.py   ~390 lines   Trace, LiveFlowDisplay (rich, optional)
    tui.py       ~400 lines   Textual TUI (optional)
```

Total: ~4,400 lines. 233 tests, including Hypothesis property tests for the
monoid laws (both operand orders), render integrity, compaction referential
integrity, transport behavior at the mocked-HTTP boundary, call-lifecycle
emission and projection, session isolation (graph roots and observers alike)
under concurrency, and the salience policy.

## Where this is going

See [ROADMAP.md](ROADMAP.md) — the short version: motif is the substrate for
an instrument where a model authors pipelines on request and the interface
constructs itself from the computation graph. The library exists so that
author has no mistakes available to make.
