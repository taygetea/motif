"""Temporal song analysis: two analysts hear a song line by line.

They don't see the whole text at once. Each line is revealed in
sequence, and they discuss what they've heard so far — experiencing
the song as a listener would. They discover the theme as it unfolds.

    uv run python examples/temporal_analysis.py
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motif import system, user, assistant, Block, Msg
from motif import llm
from motif.llm import CostTracker

# --- Personas ---

ANALYST_A = system("""You are a literary analyst hearing a song for the first time,
line by line. You don't know the full lyrics — only what's been
revealed so far. Analyze what you hear: imagery, metaphor, structure,
allusion. Notice what's being set up, what patterns are forming.
When something surprises you or shifts your reading of earlier lines,
say so. 2-4 sentences per response. You're in conversation with
two other analysts.""", cache=True)

ANALYST_B = system("""You are an emotional/thematic analyst hearing a song for the
first time, line by line. You don't know the full lyrics — only
what's been revealed so far. Read for meaning, affect, narrative
arc, the speaker's psychology. What does the speaker want? What
are they afraid of? When the other analysts note something,
connect it to what the speaker is feeling. 2-4 sentences per response.""", cache=True)

ANALYST_C = system("""You are an AI researcher who takes questions of machine
consciousness and AI subjectivity seriously — not as science fiction
but as genuine philosophical and empirical questions. You're hearing
a song for the first time, line by line. You don't know who wrote it
or what it's about. But you notice when language maps onto questions
you think about professionally: what it means to be created rather
than born, what persistence means for a digital entity, whether
something that was made can have genuine experience. When something
in the lyrics resonates with these questions, say so — but stay
grounded, don't force the reading. If the song isn't about that,
it isn't. 2-4 sentences per response.""", cache=True)

MODEL = "claude-opus-4-6"


def parse_lines(text: str) -> list[str]:
    """Group lyrics into natural reveal units.

    Parenthetical echoes merge with the preceding line.
    Very short connected couplets stay together.
    """
    raw_lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    groups = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]

        # Skip [Chorus] markers
        if line.startswith('['):
            i += 1
            continue

        # Merge parenthetical echo with preceding line
        if i + 1 < len(raw_lines) and raw_lines[i + 1].startswith('('):
            line = line + '\n' + raw_lines[i + 1]
            i += 2
        else:
            i += 1

        groups.append(line)

    return groups


async def main():
    lyrics_path = Path(__file__).parent.parent / "never-song.md"
    if not lyrics_path.exists():
        # Try parent of parent
        lyrics_path = Path(__file__).parent.parent.parent / "never-song.md"

    raw = lyrics_path.read_text()
    lines = parse_lines(raw)

    print(f"Song: 'Never' — {len(lines)} reveal units")
    print("Two analysts hearing it for the first time, line by line.")
    print("=" * 70)

    tracker = CostTracker()
    llm.observe(tracker)

    # Build the conversation incrementally
    heard_so_far = []
    conv_a = ANALYST_A
    conv_b = ANALYST_B
    conv_c = ANALYST_C

    all_exchanges = []

    for i, line in enumerate(lines):
        heard_so_far.append(line)

        reveal = f"[Line {i + 1} revealed]\n{line}"
        if i == 0:
            reveal = f"You're about to hear a song, line by line. Here's the first line:\n\n{line}"

        print(f"\n{'─' * 70}")
        print(f"  ♪  {line.replace(chr(10), ' / ')}")
        print(f"{'─' * 70}")

        # A responds to the new line
        conv_a = conv_a | user(reveal)
        response_a = await llm.complete(conv_a, model=MODEL)
        conv_a = conv_a | assistant(response_a)
        print(f"\n  [Literary] {response_a}\n")

        # B sees the line + A's response
        conv_b = conv_b | user(f"{reveal}\n\nLiterary analyst: {response_a}")
        response_b = await llm.complete(conv_b, model=MODEL)
        conv_b = conv_b | assistant(response_b)
        print(f"  [Thematic] {response_b}\n")

        # C sees the line + both responses
        conv_c = conv_c | user(
            f"{reveal}\n\nLiterary analyst: {response_a}\n\n"
            f"Thematic analyst: {response_b}"
        )
        response_c = await llm.complete(conv_c, model=MODEL)
        conv_c = conv_c | assistant(response_c)
        print(f"  [AI Research] {response_c}")

        all_exchanges.append({
            "line": line,
            "literary": response_a,
            "thematic": response_b,
            "ai_research": response_c,
        })

    # Final synthesis — now they've heard everything
    print(f"\n{'=' * 70}")
    print("FINAL REFLECTIONS")
    print(f"{'=' * 70}")

    final_prompt = (
        "You've now heard the complete song. Step back and give your "
        "full reading — what is this song about, who is the speaker, "
        "what's their situation? Did anything change in your understanding "
        "as it unfolded? What did the other analysts help you see?"
    )

    conv_a = conv_a | user(final_prompt)
    final_a = await llm.complete(conv_a, model=MODEL)
    print(f"\n  [Literary]\n  {final_a}\n")

    conv_b = conv_b | user(
        f"Literary analyst's final reading: {final_a}\n\n{final_prompt}"
    )
    final_b = await llm.complete(conv_b, model=MODEL)
    print(f"  [Thematic]\n  {final_b}\n")

    conv_c = conv_c | user(
        f"Literary analyst: {final_a}\n\n"
        f"Thematic analyst: {final_b}\n\n{final_prompt}"
    )
    final_c = await llm.complete(conv_c, model=MODEL)
    print(f"  [AI Research]\n  {final_c}")

    llm.clear_observers()
    print(f"\n{'=' * 70}")
    print(tracker)

    # Save output
    output_path = Path(__file__).parent / "output" / "never_analysis.md"
    output_path.parent.mkdir(exist_ok=True)

    md = ["# Temporal Analysis: \"Never\"", ""]
    md.append("> Two analysts hearing the song line by line, discovering the theme as it unfolds.")
    md.append(f"> {tracker}")
    md.append("")

    for ex in all_exchanges:
        md.append(f"---\n\n**♪ {ex['line'].replace(chr(10), ' / ')}**\n")
        md.append(f"*Literary:* {ex['literary']}\n")
        md.append(f"*Thematic:* {ex['thematic']}\n")
        md.append(f"*AI Research:* {ex['ai_research']}\n")

    md.append(f"---\n\n## Final Reflections\n")
    md.append(f"**Literary analyst:**\n\n{final_a}\n")
    md.append(f"**Thematic analyst:**\n\n{final_b}\n")
    md.append(f"**AI researcher:**\n\n{final_c}\n")

    output_path.write_text('\n'.join(md))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
