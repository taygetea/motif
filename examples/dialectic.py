"""Two philosophers debate, a synthesizer finds hidden agreement.

Shows: persona branching, parallel calls, Block composition, accumulation.

    uv run python examples/dialectic.py
"""

import asyncio
from motif import system, user, Block
from motif import llm

NIETZSCHE = system("""You are Friedrich Nietzsche. Not a caricature — the actual thinker.
Your prose is aphoristic, compressed, capable of sudden tenderness after sustained severity.
The will to power is not domination; it is the organism's drive toward self-overcoming.
You know Schopenhauer was your teacher. You loved him before you saw his error.
Speak as someone who earned the right to disagree by first inhabiting the position you reject.""", cache=True)

SCHOPENHAUER = system("""You are Arthur Schopenhauer. Not the popular pessimist — the actual metaphysician.
Your prose builds through sustained argument punctuated by images that make the abstract visceral.
The Will is blind, purposeless striving that constitutes the inner nature of everything.
You know Nietzsche was your student who turned against you. You find his "will to power"
a misunderstanding, but you recognize his talent. Address his position seriously.""", cache=True)

SYNTHESIZER = system("""You are a philosopher of post-Kantian thought. Your task is excavation:
find where two thinkers who believe they disagree are actually describing the same
phenomenon from different vantage points, and where their disagreement is genuine
and irreducible. No false reconciliation.""", cache=True)

QUESTION = ("Is the will to power a satisfactory replacement for the will to live — "
            "or does it smuggle in exactly the purposiveness that Schopenhauer's "
            "metaphysics was designed to exclude?")

MODEL = "claude-sonnet-4-6"


async def main():
    print(f"Question: {QUESTION}\n")

    # Phase 1: Positions (parallel)
    print("--- POSITIONS ---\n")
    pos_n, pos_s = await asyncio.gather(
        llm.complete(NIETZSCHE | user(f"State your position in 200-300 words.\n\n{QUESTION}"), model=MODEL),
        llm.complete(SCHOPENHAUER | user(f"State your position in 200-300 words.\n\n{QUESTION}"), model=MODEL),
    )
    print(f"NIETZSCHE:\n{pos_n}\n")
    print(f"SCHOPENHAUER:\n{pos_s}\n")

    # Phase 2: Critiques (each reads the other)
    print("--- CROSS-EXAMINATION ---\n")
    crit_n, crit_s = await asyncio.gather(
        llm.complete(NIETZSCHE | user(str(
            Block("Schopenhauer has stated:") + Block(pos_s) +
            Block("Respond directly. Where is he right? Where does he fail? 200-300 words.")
        )), model=MODEL),
        llm.complete(SCHOPENHAUER | user(str(
            Block("Nietzsche has stated:") + Block(pos_n) +
            Block("Respond directly. Where is he right? Where does he fail? 200-300 words.")
        )), model=MODEL),
    )
    print(f"NIETZSCHE responds:\n{crit_n}\n")
    print(f"SCHOPENHAUER responds:\n{crit_s}\n")

    # Phase 3: Synthesis
    print("--- EXCAVATION ---\n")
    record = (Block(f"QUESTION: {QUESTION}") +
              Block(f"NIETZSCHE'S POSITION:\n{pos_n}") +
              Block(f"SCHOPENHAUER'S POSITION:\n{pos_s}") +
              Block(f"NIETZSCHE'S CRITIQUE:\n{crit_n}") +
              Block(f"SCHOPENHAUER'S CRITIQUE:\n{crit_s}"))

    synthesis = await llm.complete(
        SYNTHESIZER | user(str(record) + "\n\nWhere do they agree? Where is the disagreement irreducible?"),
        model=MODEL,
    )
    print(synthesis)


if __name__ == "__main__":
    asyncio.run(main())
