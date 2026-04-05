# motif

A prompt algebra for LLM orchestration. ~2000 lines. Does what 50,000-line frameworks do — and the code reads as its own specification.

```python
from motif import system, user, flow

# Discover research angles, investigate each with web search,
# have the researchers discuss findings, synthesize a report
angles = await flow.branch(decomposer | user(topic), schema=ANGLES_SCHEMA)
briefs = await asyncio.gather(*[research_agent(a) for a in angles])
board, discussion = await flow.blackboard(researchers, seed=findings, rounds=2)
report = await llm.complete(synthesizer | user(board))
```

This is a real pipeline — it produced a [research report on mechanistic interpretability](examples/output/mechinterp_research.md) ([code](examples/deep_research.py)) with web search, cross-researcher discussion, and emergent findings no single agent saw alone. The same library composes [philosophical debates](examples/dialectic.py), [expert panels](examples/blackboard.py), and [prismatic multi-lens analysis](examples/prism.py).

## What makes it different

Most orchestration libraries build elaborate runtime machinery — chains, agents, memory systems, callback managers — and the actual prompt composition is an afterthought. Motif inverts this: **the representation is the machinery.**

A `Msg` is an immutable sequence of typed segments. `|` composes them. That's it. The type is a monoid — composition is associative, branching is free, and the flow patterns are just functions over messages. Agents aren't a framework concept; they're what happens when tool results feed back into the Msg.

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

## Three layers

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
render(prompt, backend="anthropic")   # content blocks, cache_control
render(prompt, backend="openai")      # system message + turns
```

### Layer 2: LLM calls (`llm.py`)

Three verbs. Msg in, value out.

```python
from motif import llm

text = await llm.complete(prompt)                          # text out
data = await llm.extract(prompt, schema=MY_SCHEMA)         # structured data out
result = await llm.act(prompt, tools=TOOL_SCHEMAS)         # action out (text or tool calls)
```

### Layer 3: Flow patterns (`flow.py`)

Nine named patterns for multi-call orchestration. Eight with predetermined topology, one that generates topology at runtime.

```python
from motif import flow

# Discover structure dynamically
methods = await flow.branch(enumerator | user(doc), schema=SCHEMA)

# Apply in parallel (with concurrency control)
analyses = await flow.fan(methods, lambda m: analyst | user(m["name"]), max_concurrency=5)

# Converge
synthesis = await flow.reduce(analyses, lambda t: synth | user(t), labels=[...])

# Agent loop — the Msg grows until the model finishes or a signal tool fires
result = await flow.agent(prompt, tools=TOOLS, tool_schemas=SCHEMAS)

# Expert panel with shared state and selective visibility
board, history = await flow.blackboard(experts, seed=question, rounds=2, filter_fn=degrade)
```

All patterns emit events to observers. Watch the computation graph execute live:

```python
from motif.display import LiveFlowDisplay, Trace

trace = Trace()
display = LiveFlowDisplay()
flow.observe(trace, display)

async with display:
    result = await flow.fan(items, fn)

trace.save("run.json")     # serializable execution record
```

```
Flow
├── ◆ discover lenses → 6 branches (14.3s)
│   ├── ✓ Rhetorical register analysis (9.6s)
│   │     The passage's most consequential move is not its emotional rhetoric...
│   ├── ✓ Ventriloquism and voice theory (10.1s)
│   │     Land's ventriloquism is not decorative...
│   ├── ⠋ Performative contradiction (12s)
│   ...
```

## The nine flow patterns

| Pattern | Does |
|---------|------|
| `branch` | One call discovers structure → list of items |
| `fan` | Items → parallel calls → results (with concurrency control) |
| `reduce` | Results → labeled synthesis → one output |
| `best_of` | Parallel judging → pick the winner |
| `cascade` | Try cheap model first, escalate if insufficient |
| `tree` | Recursive decomposition — split, analyze leaves, merge back up |
| `tournament` | Bracket-style elimination with pairwise comparison |
| `blackboard` | Expert panel with shared state and selective visibility (`filter_fn`) |
| `agent` | Tool-use loop — the Msg grows until a signal tool fires |

## Why it composes

The [capstone example](examples/agent_compose.py) puts flow patterns inside agent tool handlers — a blackboard discussion running inside a tool call, inside an agent loop — and it [works on the first run](examples/output/agent_compose.md):

```
research analyst (agent loop)
  step 1:
    → model calls expert_panel tool
      → blackboard(4 experts × 2 rounds)       ← flow pattern inside tool handler
        → 8 parallel LLM calls
      → returns discussion to agent
  step 2:
    → model does web search (server-side)       ← Anthropic's tool, transparent
    → model writes 3000-word sourced report
```

The reason this composes without debugging is that there's nothing to compose. A tool handler is an async function that takes a dict and returns a string. Inside, `flow.blackboard()` builds its own Msgs from scratch — they're independent of the agent's Msg. When the handler returns, the result becomes a `tool_result` segment in the agent's Msg. The two histories never interact. They can't conflict because they're in different scopes.

The Msg monoid is load-bearing here: the agent's Msg grows with `|`. Inside the tool handler, the flow pattern's Msgs are separate values. There's no "agent context" to thread through, no "flow executor" that needs to know it's nested, no shared mutable state. It's just functions calling functions.

## Why it works

Msg is a monoid: `|` is associative, `Msg()` is the identity. Block is a second monoid. `render()` is a homomorphism to API payloads. The reason composition feels right is that it *is* right — the operations mean what they look like they mean because the underlying structure is algebraic.

**The representation is the thing.** A Msg doesn't describe a message — it is the message. A ToolCall segment has `.id`, `.name`, `.input` as real attributes, not encoded in a string.

**The call site is the documentation.** `branch → fan → reduce` says what it does. The pipeline code reads as a description of what it does, not how it manages concurrency or formats prompts.

**Observers observe; they don't intervene.** The pipeline stays pure. Display, logging, tracing — all through observers. Cost tracking, retries — that's middleware, a separate concern. Observer lists are module-level global state — ergonomic for scripts, but shared across concurrent pipelines. If you need isolation (e.g., a server), this is the seam where an explicit Runtime object would go.

**Compaction is invisible.** The agent loop automatically summarizes older turns when approaching context limits. The user never thinks about it.

### Two kinds of joining

`Block + Block` — within-segment composition. For building one segment from parts.
`Block.join(items, labels=)` — between-results composition. For presenting multiple outputs to a synthesis call.

## Installation

```bash
pip install motif-llm              # core
pip install motif-llm[display]     # + live terminal display

# or with uv
uv add motif-llm
```

Installs as `motif-llm`, imports as `motif`:

```python
from motif import system, user, flow
```

Set `ANTHROPIC_API_KEY` in `.env` or environment. Currently Anthropic-only for LLM calls; `render()` supports OpenAI format for use with your own client.

## Examples

```bash
python examples/dialectic.py                    # Nietzsche vs Schopenhauer debate
python examples/prism.py                        # Multi-lens analysis with live display
python examples/blackboard.py                   # Expert panel with shared state
python examples/deep_research.py "your topic"   # Deep research with web search + discussion
```

## What's in the box

```
src/motif/
    prompt.py    ~340 lines   Msg, Block, segments, render — zero dependencies
    llm.py       ~385 lines   complete, extract, act, CostTracker — anthropic SDK
    flow.py      ~980 lines   9 flow patterns + compaction + events
    display.py   ~310 lines   Trace, LiveFlowDisplay — rich (optional)
```

Total: ~2000 lines. 90 tests (including property-based tests via Hypothesis) verify the monoid laws, render homomorphism, compaction integrity, and all flow patterns.
