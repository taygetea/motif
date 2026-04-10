"""Deep research with discussion.

Standard deep research: fan-out queries → aggregate results.
This: fan-out research agents → discuss findings → synthesize.

The discussion phase is where this differs. Each researcher
sees what the others found and can say "that contradicts X"
or "your finding about Y supports my hypothesis." Emergent
understanding that aggregation can't produce.

Architecture:
    branch  → discover research angles
    fan     → each angle is an agent with web search
    blackboard → researchers discuss findings (2 rounds)
    reduce  → final synthesis

    uv run python examples/deep_research.py "topic here"
"""

import asyncio
import sys
import os

# Add parent to path for running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motif import system, user, Block
from motif import llm, flow
from motif.display import Trace, LiveFlowDisplay

# --- Configuration ---

MODEL = "claude-sonnet-4-6"
MODEL_BRANCH = "claude-sonnet-4-6"  # needs to be good at decomposition

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

ANGLES_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short label for this research angle"},
                    "question": {"type": "string", "description": "The specific question to investigate"},
                    "why": {"type": "string", "description": "Why this angle matters for the overall topic"},
                },
                "required": ["name", "question", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

# --- Prompts ---

DECOMPOSER = system("""You decompose research topics into 3-4 complementary angles.
Each angle should investigate something the others can't — together they
should cover the topic from perspectives that will be productive when
the researchers later discuss their findings with each other.
Prefer angles that might tension or surprise each other.""", cache=True)

RESEARCHER = system("""You are a research agent with web search. Your job:
1. Search for information relevant to your assigned question
2. Follow leads — if a result suggests a more specific query, search again
3. When you have enough, write a concise research brief (300-500 words)

Be specific. Cite what you found. Flag uncertainties and contradictions
in sources. Your brief will be shared with other researchers investigating
related angles of the same topic — write for an informed peer, not a
general audience.

When you're done researching, just write your brief as your final response.""", cache=True)

DISCUSSANT_TEMPLATE = """You are a researcher who just completed an investigation into: {angle}.

Your findings:
{findings}

You're now in a discussion with other researchers who investigated different
angles of the same topic. Read their findings carefully. In each round:
- Note where your findings connect to, support, or contradict theirs
- Identify gaps — things nobody investigated that now seem important
- Refine your own understanding based on what you've learned
- Be specific about evidence, not just opinions

150-250 words per round. Build on what others said, don't repeat yourself."""


SYNTHESIZER = system("""You synthesize a multi-researcher discussion into a comprehensive
research report. You have:
- The original research briefs from each angle
- Two rounds of cross-discussion where researchers engaged with each other's findings

Your report should:
1. State the key findings (what the research established)
2. Identify the most important tensions or contradictions between angles
3. Note what emerged from the discussion that no single researcher saw alone
4. Flag open questions and uncertainties
5. Be specific — cite the evidence, not just the conclusions

Write for someone who wants to understand the topic deeply. 500-800 words.""", cache=True)


async def research_angle(angle: dict) -> str:
    """Run a research agent on one angle. Returns the research brief."""
    result = await flow.agent(
        RESEARCHER | user(
            f"Research angle: {angle['name']}\n"
            f"Question: {angle['question']}\n"
            f"Why this matters: {angle['why']}"
        ),
        tools={},  # no custom tools — web search is server-side
        tool_schemas=[WEB_SEARCH_TOOL],
        model=MODEL,
        max_steps=8,
        title=angle["name"],
    )
    return result.output


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/deep_research.py \"topic\"")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])

    trace = Trace()
    display = LiveFlowDisplay()
    flow.observe(trace, display)

    async with display:
        # Phase 1: Discover research angles
        angles = await flow.branch(
            DECOMPOSER | user(f"Topic: {topic}"),
            schema=ANGLES_SCHEMA,
            model=MODEL_BRANCH,
            title="decompose topic",
        )

        # Phase 2: Research each angle in parallel (agents with web search)
        briefs = await asyncio.gather(*[
            research_angle(angle) for angle in angles
        ])

        # Phase 3: Discussion (blackboard with researcher personas)
        # Build a persona for each researcher based on their angle and findings
        discussants = []
        for angle, brief in zip(angles, briefs):
            prompt_text = DISCUSSANT_TEMPLATE.format(
                angle=angle["name"],
                findings=brief,
            )
            discussants.append((
                angle["name"],
                lambda board, _sys=system(prompt_text, cache=True): _sys | user(board),
            ))

        seed_board = "RESEARCH DISCUSSION\n\n"
        seed_board += Block.join(
            [f"[{a['name']}] investigated: {a['question']}\n\nFindings:\n{b}"
             for a, b in zip(angles, briefs)]
        )
        seed_board += "\n\nPlease discuss: what do your findings mean together?"

        board, discussion = await flow.blackboard(
            agents=discussants,
            seed=seed_board,
            title="researcher discussion",
            rounds=2,
            model=MODEL,
        )

        # Phase 4: Synthesize
        synth_input = Block.join(
            [board, "Now synthesize the complete findings and discussion into a report."]
        )
        report = await llm.complete(
            SYNTHESIZER | user(synth_input),
            model=MODEL,
        )

    flow.clear_observers()

    # Output
    print("\n" + "=" * 70)
    print(f"DEEP RESEARCH: {topic}")
    print("=" * 70)

    print(f"\nAngles investigated: {', '.join(a['name'] for a in angles)}")
    print(f"Research agents: {len(angles)} parallel")
    print(f"Discussion: {len(discussion)} rounds × {len(angles)} researchers")
    print(f"Total trace events: {len(trace)}")
    print(f"Wall time: {trace.total_elapsed:.1f}s")

    print("\n" + "-" * 70)
    print("REPORT")
    print("-" * 70)
    print(report)

    # Save trace
    trace_path = "/tmp/deep_research_trace.json"
    trace.save(trace_path)
    print(f"\nTrace saved to {trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
