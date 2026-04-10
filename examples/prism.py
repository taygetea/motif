"""Prismatic analysis with live display.

Discovers analytical lenses, applies them in parallel, synthesizes.
Shows: branch → fan → reduce with flow events and rich tree display.

    uv run python examples/prism.py
"""

import asyncio
from motif import system, user
from motif import flow
from motif.display import LiveFlowDisplay

ENUMERATOR = system("""Given a document, identify 4-6 distinct analytical lenses that would
reveal something non-obvious. Each chosen because THIS document rewards it.""", cache=True)

ANALYST = system("""Apply the assigned methodology rigorously to the document.
Surface something a generic reading would miss. 150-250 words.""", cache=True)

SYNTHESIZER = system("""Multiple analyses of the same document from different lenses.
What does the intersection reveal that no single lens could?
Where do lenses contradict? 200-300 words.""", cache=True)

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

# Default document — Schopenhauer, "On the Sufferings of the World" (opening)
DEFAULT_DOC = """\
Unless suffering is the direct and immediate object of life, our existence must
entirely fail of its aim. It is absurd to look upon the enormous amount of pain
that abounds everywhere in the world, and originates in needs and necessities
inseparable from life itself, as serving no purpose at all and the result of mere
chance. Each separate misfortune, as it comes, seems, no doubt, to be something
exceptional; but misfortune in general is the rule.

I know of no greater absurdity than that propounded by most systems of philosophy
in declaring evil to be negative in its character. Evil is just what is positive;
it makes its own existence felt. It is the good which is negative; in other words,
happiness and satisfaction always imply some desire fulfilled, some state of pain
brought to an end.

The pleasure in this world, it has been said, outweighs the pain; or, at any rate,
there is an even balance between the two. If the reader wishes to see shortly
whether this statement is true, let him compare the respective feelings of two
animals, one of which is engaged in eating the other.

In early youth, as we contemplate our coming life, we are like children in a theatre
before the curtain is raised, sitting there in high spirits and eagerly waiting for
the play to begin. It is a blessing that we do not know what is really going to
happen. Could we foresee it, there are times when children might seem like innocent
prisoners, condemned, not to death, but to life, and as yet all unconscious of what
their sentence means.
"""

MODEL = "claude-sonnet-4-6"


async def main():
    doc = DEFAULT_DOC

    display = LiveFlowDisplay()
    flow.observe(display)

    async with display:
        methods = await flow.branch(
            ENUMERATOR | user(f"Lenses for:\n\n{doc}"),
            schema=METHODS_SCHEMA,
            model=MODEL,
            title="discover lenses",
        )

        analyses = await flow.fan(
            methods,
            lambda m: ANALYST | user(
                f"Methodology: {m['name']}\n{m['rationale']}\n\nDocument:\n{doc}"
            ),
            model=MODEL,
            title="parallel analysis",
        )

        synthesis = await flow.reduce(
            analyses,
            lambda combined: SYNTHESIZER | user(combined),
            model=MODEL,
            title="synthesis",
        )

    flow.clear_observers()
    print(f"\n{'=' * 70}")
    print("SYNTHESIS:")
    print(f"{'=' * 70}")
    print(synthesis)


if __name__ == "__main__":
    asyncio.run(main())
