"""TUI demo: branch → fan → reduce with streaming.

Shows the full topology: branch discovers lenses (single panel),
fan applies them in parallel (horizontal row), reduce synthesizes
(single panel below). Three sequential steps, the middle one parallel.

    uv run python examples/tui_full.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motif import system, user, Block
from motif import flow
from motif.tui import FlowApp

ENUMERATOR = system("""Given a document, identify 4 distinct analytical lenses that would
reveal something non-obvious. Each chosen because THIS document rewards it.""", cache=True)

ANALYST = system("""Apply the assigned methodology rigorously to the document.
Surface something a generic reading would miss. 3-5 sentences.""", cache=True)

SYNTHESIZER = system("""Multiple analyses of the same document from different lenses.
What does the intersection reveal that no single lens could?
Where do lenses contradict? 3-5 sentences.""", cache=True)

METHODS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

DOC = """\
You burned my soul again
Crashed me within, I know I should've known
I burn your kingdoms
At fault for what you made alone
And so I pray
Clasp my hands together, hear me now
And hope someday
Come whatever, I may fade away
I never die
"""


async def run_pipeline():
    # Step 1: Branch — discover lenses (single panel)
    methods = await flow.branch(
        ENUMERATOR | user(f"Lenses for:\n\n{DOC}"),
        schema=METHODS_SCHEMA,
        model="claude-sonnet-4-6",
        title="discover lenses",
    )

    # Step 2: Fan — apply each lens in parallel (horizontal row)
    analyses = await flow.fan(
        methods,
        lambda m: ANALYST | user(
            f"Methodology: {m['name']}\n{m['rationale']}\n\nDocument:\n{DOC}"
        ),
        model="claude-sonnet-4-6",
        streaming=True,
        title="parallel analysis",
    )

    # Step 3: Reduce — synthesize (single panel below)
    synthesis = await flow.reduce(
        analyses,
        lambda combined: SYNTHESIZER | user(combined),
        labels=[m["name"] for m in methods],
        model="claude-sonnet-4-6",
        streaming=True,
        title="synthesis",
    )

    return synthesis


def main():
    app = FlowApp(title="Branch → Fan → Reduce")
    app.run_pipeline(run_pipeline)
    app.run()


if __name__ == "__main__":
    main()
