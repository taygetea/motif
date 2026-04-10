"""TUI demo: streaming flow viewer.

Watch a fan-out of analyses stream live in parallel panels.

    uv run python examples/tui_demo.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motif import system, user
from motif import flow
from motif.tui import FlowApp

ANALYST = system("Analyze through this lens. Be specific. 3-4 sentences.", cache=True)

DOC = """\
Unless suffering is the direct and immediate object of life, our existence must
entirely fail of its aim. It is absurd to look upon the enormous amount of pain
that abounds everywhere in the world, and originates in needs and necessities
inseparable from life itself, as serving no purpose at all and the result of mere
chance. Each separate misfortune, as it comes, seems, no doubt, to be something
exceptional; but misfortune in general is the rule.
"""

LENSES = [
    {"name": "Rhetoric", "focus": "argumentative structure and persuasive technique"},
    {"name": "Psychology", "focus": "what the text reveals about its author's inner life"},
    {"name": "Philosophy", "focus": "logical structure, assumptions, and validity"},
    {"name": "Theology", "focus": "religious imagery and its function"},
    {"name": "Linguistics", "focus": "word choice, sentence structure, register shifts"},
]


async def run_analysis():
    results = await flow.fan(
        LENSES,
        lambda l: ANALYST | user(f"Lens: {l['name']} ({l['focus']})\n\n{DOC}"),
        title="multi-lens analysis",
        model="claude-sonnet-4-6",
        streaming=True,
    )

    synthesis = await flow.reduce(
        results,
        lambda combined: system("Synthesize these analyses. What do the lenses reveal together?")
        | user(combined),
        title="synthesis",
        labels=[l["name"] for l in LENSES],
        model="claude-sonnet-4-6",
        streaming=True,
    )

    return synthesis


def main():
    app = FlowApp(title="Multi-Lens Analysis")
    app.run_pipeline(run_analysis)
    app.run()


if __name__ == "__main__":
    main()
