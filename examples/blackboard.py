"""Expert panel with shared state (blackboard pattern).

Three experts analyze a passage. Each round, all see everything said so far.
Ideas cross-pollinate across rounds in ways that independent fan-out can't produce.

    uv run python examples/blackboard.py
"""

import asyncio
from motif import system, user
from motif import flow
from motif.display import LiveFlowDisplay

RHETORICIAN = system("""You are a rhetorician analyzing argumentative structure and persuasive technique.
When other experts have contributed, build on theirs where rhetoric intersects
their domain. Don't repeat what's been said. 150-250 words per round.""", cache=True)

PSYCHOLOGIST = system("""You are a depth psychologist reading for what the text reveals about its author —
the defenses, needs, wounds visible through the argumentative surface.
When other experts have contributed, notice what their observations imply
psychologically. Don't repeat what's been said. 150-250 words per round.""", cache=True)

PHILOSOPHER = system("""You are a philosopher evaluating the argument's logical structure —
where it's rigorous, where it relies on force instead of proof, what it assumes.
When other experts have contributed, engage with their observations philosophically.
Don't repeat what's been said. 150-250 words per round.""", cache=True)

PASSAGE = """\
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
"""

MODEL = "claude-sonnet-4-6"


async def main():
    display = LiveFlowDisplay()
    flow.observe(display)

    seed = f"PASSAGE FOR ANALYSIS:\n\n{PASSAGE}\n\nPlease offer your first observations."

    async with display:
        board, history = await flow.blackboard(
            agents=[
                ("Rhetorician", lambda b: RHETORICIAN | user(b)),
                ("Psychologist", lambda b: PSYCHOLOGIST | user(b)),
                ("Philosopher", lambda b: PHILOSOPHER | user(b)),
            ],
            seed=seed,
            rounds=2,
            model=MODEL,
        )

    flow.clear_observers()

    # Print full contributions
    for round_num, round_data in enumerate(history, 1):
        print(f"\n{'─' * 70}")
        print(f"ROUND {round_num}")
        print(f"{'─' * 70}")
        for name, contribution in round_data.items():
            print(f"\n  [{name}]")
            for line in contribution.split('\n'):
                print(f"  {line}")


if __name__ == "__main__":
    asyncio.run(main())
