"""Capstone: an agent whose tools are flow patterns.

The agent has three capabilities:
    - web_search (server-side, Anthropic)
    - multi_analyze (branch → fan → reduce)
    - expert_panel (blackboard with 2 rounds)

It decides when to search, when to analyze findings in depth,
and when to convene experts for discussion. The flow patterns
run inside tool handlers — DAGs inside the agent loop.

    uv run python examples/agent_compose.py "your question"
    uv run python examples/agent_compose.py  # uses default question
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motif import system, user, Block
from motif import llm, flow
from motif.display import Trace, LiveFlowDisplay

# --- Configuration ---

MODEL_AGENT = "claude-sonnet-4-6"   # the orchestrator
MODEL_TOOLS = "claude-haiku-4-5"    # internal flow patterns (cheap)

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}

DEFAULT_QUESTION = (
    "What are the strongest arguments for and against open-sourcing "
    "frontier AI models? What does the most recent evidence suggest?"
)

# --- Tool: multi_analyze (branch → fan → reduce) ---

LENS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["name", "focus"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


async def multi_analyze(inp):
    """Run multi-perspective analysis. Internally: branch → fan → reduce."""
    text = inp.get("text", "")

    lenses = await flow.branch(
        system("Pick 3 distinct analytical lenses for this text. "
               "Each should reveal something the others can't.")
        | user(text),
        schema=LENS_SCHEMA,
        model=MODEL_TOOLS,
        title="discover lenses",
        depth=3,
    )

    analyses = await flow.fan(
        lenses,
        lambda l: (
            system(f"You analyze through the lens of: {l['name']} ({l['focus']}). "
                   "Be specific and concise. 2-3 sentences.")
            | user(text)
        ),
        model=MODEL_TOOLS,
        title="apply lenses",
        depth=3,
    )

    synthesis = await flow.reduce(
        analyses,
        lambda combined: (
            system("Synthesize these analyses into one paragraph. "
                   "Note tensions between the perspectives.")
            | user(combined)
        ),
        labels=[l["name"] for l in lenses],
        model=MODEL_TOOLS,
        title="synthesize lenses",
        depth=3,
    )

    return synthesis


# --- Tool: expert_panel (blackboard) ---

async def expert_panel(inp):
    """Convene an expert panel discussion. Internally: blackboard."""
    topic = inp.get("topic", "")
    expert_types = inp.get("experts", ["policy analyst", "technical researcher"])

    agents = []
    for expert in expert_types:
        agents.append((
            expert,
            lambda board, _e=expert: (
                system(f"You are a {_e}. Respond to the discussion with your "
                       "perspective. Build on others' points. 2-3 sentences.")
                | user(board)
            ),
        ))

    board, history = await flow.blackboard(
        agents=agents,
        seed=f"Discussion topic: {topic}\n\nPlease share your initial perspective.",
        title="expert panel",
        rounds=2,
        model=MODEL_TOOLS,
        depth=3,
    )

    # Extract just the discussion, not the seed
    discussion_parts = []
    for rnd, round_data in enumerate(history, 1):
        for name, contribution in round_data.items():
            discussion_parts.append(f"[{name}, round {rnd}]: {contribution}")

    return "\n\n".join(discussion_parts)


# --- Tool schemas ---

TOOL_SCHEMAS = [
    WEB_SEARCH,
    {
        "name": "multi_analyze",
        "description": (
            "Run multi-perspective analysis on a piece of text or findings. "
            "Uses 3 analytical lenses chosen for the specific content, "
            "analyzes through each, and synthesizes. Good for understanding "
            "complex or contested claims from multiple angles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text or findings to analyze",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "expert_panel",
        "description": (
            "Convene a panel of experts to discuss a topic. They see each "
            "other's contributions and build on them across 2 rounds. "
            "Good for exploring tensions, generating nuanced positions, "
            "or stress-testing an argument."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or question for discussion",
                },
                "experts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of experts (e.g., 'economist', 'ethicist'). Default: policy analyst + technical researcher.",
                },
            },
            "required": ["topic"],
        },
    },
]


async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION

    trace = Trace()
    display = LiveFlowDisplay()
    flow.observe(trace, display)

    prompt = (
        system(
            "You are a research analyst with three capabilities:\n"
            "1. Web search — find current information\n"
            "2. Multi-perspective analysis — analyze text through multiple lenses\n"
            "3. Expert panel — convene experts to discuss and debate\n\n"
            "Use these strategically: search for facts, analyze complex findings, "
            "convene experts when you need diverse perspectives or want to "
            "stress-test a position. Produce a thorough, well-sourced answer."
        )
        | user(question)
    )

    async with display:
        result = await flow.agent(
            prompt,
            tools={
                "multi_analyze": multi_analyze,
                "expert_panel": expert_panel,
            },
            tool_schemas=TOOL_SCHEMAS,
            model=MODEL_AGENT,
            title="research analyst",
        )

    flow.clear_observers()

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(result.output)
    print()
    print(f"Steps: {result.steps}")
    print(f"Segments: {len(result.msg.segments)}")
    print(f"Trace events: {len(trace)}")
    print(f"Wall time: {trace.total_elapsed:.1f}s")

    trace.save("/tmp/agent_compose_trace.json")
    print("Trace saved to /tmp/agent_compose_trace.json")


if __name__ == "__main__":
    asyncio.run(main())
