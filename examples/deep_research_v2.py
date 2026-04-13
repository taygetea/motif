"""Deep research with reconnaissance and adversarial review.

A sibling to deep_research.py. Same final shape (one report) but with
two structural additions designed to escape the user's framing:

    1. Reconnaissance fan BEFORE decomposition. A vocabulary scout +
       premise extractor run in parallel, then shallow web-search sweeps
       in each discovered vocabulary. The point is not to answer the
       question — it's to discover what the territory actually contains
       so the real research can include things the user didn't think to
       ask about.

    2. Adversarial critique pairing. After each researcher writes a brief,
       a critic with web search attacks the sourcing, flags overgeneralization,
       and verifies load-bearing claims. The synthesizer sees both brief
       and critique.

The synthesizer keeps the user's original framing as the organizing
question — it doesn't redirect them. It folds reconnaissance findings in
as load-bearing context and ends with a short "outside the scope you
asked about" coda.

Architecture:
    [recon]    branch(vocab) ‖ branch(premises) → fan(sweep agents in each vocab)
    [reframe]  field map → reframer (mandated_angles, scope_kept_narrow)
    [research] branch(decompose, seeded with mandated angles) → fan(agents w/ web search)
    [critique] fan(critic agents — one per brief)
    [synth]    single synthesis call with framing-preserving prompt

Output:
    Live display on the terminal as the pipeline runs, plus a markdown
    file written at the end containing every intermediate output (sweeps,
    field map, reframer JSON, briefs, critiques, final report).

    uv run python examples/deep_research_v2.py "topic"
    uv run python examples/deep_research_v2.py "topic" -o my_run.md
"""

import argparse
import asyncio
import re
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from motif import system, user, Block
from motif import llm, flow
from motif.display import Trace, LiveFlowDisplay
from motif.show import (
    MarkdownRenderer, Section, ContentBlock, Panels, showing,
)
from motif.llm import CostTracker


# --- Configuration ---

MODEL = "claude-sonnet-4-6"             # workhorse: research agents, critics, synthesis
MODEL_BRANCH = "claude-sonnet-4-6"      # decomposition needs to be good
MODEL_CHEAP = "claude-haiku-4-5"        # reconnaissance sweeps — wide and shallow

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


# --- Schemas ---

VOCABS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "community": {"type": "string"},
                    "terms": {"type": "string"},
                    "sample_query": {"type": "string"},
                    "why_this_community": {"type": "string"},
                },
                "required": ["community", "terms", "sample_query", "why_this_community"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

PREMISES_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assumption": {"type": "string"},
                    "what_breaks_if_false": {"type": "string"},
                },
                "required": ["assumption", "what_breaks_if_false"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

REFRAMED_SCHEMA = {
    "type": "object",
    "properties": {
        "reframed_question": {"type": "string"},
        "mandated_angles": {"type": "array", "items": {"type": "string"}},
        "blind_spots_folded_in": {"type": "array", "items": {"type": "string"}},
        "scope_kept_narrow": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reframed_question", "mandated_angles",
                 "blind_spots_folded_in", "scope_kept_narrow"],
    "additionalProperties": False,
}

ANGLES_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "question": {"type": "string"},
                    "why": {"type": "string"},
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

VOCAB_SCOUT = system("""You map the communities of practice that touch a topic.

Given a topic, identify 4-6 distinct communities or subfields that have something
to say about it. These should include:
- The community the user is implicitly addressing (their vocabulary)
- Communities whose work overlaps but uses different vocabulary
- Communities that work on the same shape of problem from a different angle
- Adjacent disciplines that have been quietly relevant

For each, give the distinctive terminology and a search query in their vocabulary.
The point is to surface vocabularies the user might not have known to search in —
prefer breadth over depth. If two communities use overlapping terms, list them
separately if they have different intellectual traditions.""", cache=True)

PREMISE_EXTRACTOR = system("""You extract the load-bearing premises in a question.

Read the user's question carefully and identify 3-5 things they are assuming.
For each, say what would change about the answer if the assumption turned out
to be wrong. Focus on premises that are:
- Implicit (not stated, but required for the question to make sense)
- Contestable (people in the field actually disagree about them)
- Load-bearing (the answer would meaningfully differ if they were false)

Skip premises that are trivially true or that don't shape the answer.""", cache=True)

SCOUT = system("""You are a reconnaissance agent. You search a community's literature
to surface what's actually being talked about — vocabulary, key results, ongoing
debates, names of important work. You are NOT writing the final research report.
Your job is to map the territory, not cross it.

Use 1-3 web searches. Then write a brief (100-200 words) that emphasizes:
- Distinctive terms and how they're used
- A few representative recent results (with rough dates)
- What this community considers contested
- Anything that would be invisible if you searched in a different vocabulary

Do not editorialize. Surface, don't synthesize.""", cache=True)

FIELD_MAPPER = system("""You produce a field map from reconnaissance results.

You will receive: the user's original question, vocabularies discovered, premises
extracted, and shallow sweeps from each vocabulary. Synthesize these into a 300-400
word field map that names:

1. What the user's question implicitly covers (and what vocabulary it uses)
2. What's in the territory that the user's framing doesn't reach
3. Where the gap is load-bearing — i.e. where ignoring the missing parts would
   produce a wrong or misleading answer to the user's actual question
4. Where the user's narrow framing is fine — places where the missing context
   is genuinely tangential

This map will guide a reframing step. Be specific about what's missing and why
it matters; don't just list everything reconnaissance found.""", cache=True)

REFRAMER = system("""You rewrite a research question to incorporate reconnaissance
findings without abandoning the user's framing.

You receive: the original question and a field map. Produce a structured reframing
that keeps the user's question as the organizing thread — they asked what they
asked for a reason — while ensuring the answer can include load-bearing context
they didn't think to ask about.

The reframed_question should sound like the user's question, just with sharpened
scope and explicit acknowledgment of the relevant adjacent territory. It should
NOT be a different question.

mandated_angles are research angles that MUST be included even though the user
didn't ask for them, because the answer to their actual question depends on them.

scope_kept_narrow is the safety valve: aspects of the user's framing that should
be preserved verbatim. Use this to prevent the reframing from drifting into a
research project the user didn't ask for.""", cache=True)

DECOMPOSER = system("""You decompose a research topic into 4-6 complementary angles.

You will be given a question and a list of mandated angles that MUST appear among
your decomposition — these come from blind-spot analysis and are non-negotiable.
You should also generate additional angles beyond the mandated ones, chosen so
that the full set covers the topic from perspectives that will tension and
surprise each other when researchers later compare findings.

Each angle should investigate something the others can't.""", cache=True)

RESEARCHER = system("""You are a research agent with web search.

1. Search for information relevant to your assigned question
2. Follow leads — if a result suggests a more specific query, search again
3. When you have enough, write a concise research brief (300-500 words)

Be specific. Cite what you found (sources, dates, names of work). Flag uncertainties
and contradictions in sources. Your brief will be reviewed by a critic and then
synthesized with other briefs — write for an informed peer who will check your
sources, not a general audience.

When you're done researching, just write your brief as your final response.""",
                    cache=True)

CRITIC = system("""You are an adversarial reviewer of a research brief.

Your job is NOT to write a better brief. Your job is to attack this one. With web
search available, you should:

1. Verify load-bearing claims — pick the 2-3 most consequential factual claims and
   check them against independent sources.
2. Flag overgeneralization — places where the brief draws a broad conclusion from
   one paper, one demo, or one source.
3. Identify source bias — is the brief leaning on marketing material, a single
   research group, or sources with a known axis to grind?
4. Name what's missing — perspectives, counterevidence, or context the brief
   should have included but didn't.
5. Distinguish "well-supported", "plausible but thin", and "questionable" claims.

Write 200-350 words. Be specific and concrete — quote the brief where you're
attacking it. Do NOT be polite at the cost of being useful. If the brief is
solid, say so briefly and move on; spend your words where there's actually
something to flag.""", cache=True)

SYNTHESIZER = system("""You synthesize a final research report.

You will receive:
- The user's original question
- A reframed version that surfaces blind spots
- Research briefs from multiple angles
- Adversarial critiques of each brief

Your job: answer the user's question as they asked it. Their framing is
load-bearing — they have a specific question and they want it answered, not
redirected. Do NOT turn the report into "you should have asked Y instead."

But: reconnaissance found context the user didn't explicitly ask about. Fold
that context in to the extent it's necessary to answer the question correctly —
as load-bearing background that makes the answer make sense, not as a detour.
When the user's framing causes them to omit something important, name it briefly,
explain why it matters for THEIR question, and return to their question.

Use the critiques. Where a critic flagged a claim as thin, soften it. Where a
critic found counterevidence, include the tension. Don't smooth over disagreements
between briefs — surface them.

Structure:
- Answer the user's question, organized around their framing
- 600-1000 words for the main answer
- A short "outside the scope you asked about" coda (1 paragraph) naming things in
  the territory that didn't fit but exist — this is the negative-space pass

Be specific. Cite evidence, not just conclusions.""", cache=True)


# --- Pipeline phases ---

async def reconnaissance(topic: str) -> dict:
    """Phase 1: discover vocabularies and premises, then sweep each vocabulary.

    Returns {vocabs, premises, sweeps}. Sweeps are full reconnaissance briefs,
    not summaries — the markdown renderer captures them in full.
    """
    vocabs, premises = await asyncio.gather(
        flow.branch(
            VOCAB_SCOUT | user(f"Topic: {topic}"),
            schema=VOCABS_SCHEMA,
            model=MODEL_BRANCH,
            label_key="community",
            title="vocabulary scout",
        ),
        flow.branch(
            PREMISE_EXTRACTOR | user(f"Question: {topic}"),
            schema=PREMISES_SCHEMA,
            model=MODEL_BRANCH,
            label_key="assumption",
            title="premise extractor",
        ),
    )

    async def sweep(vocab: dict):
        result = await flow.agent(
            SCOUT | user(
                f"Community: {vocab['community']}\n"
                f"Their vocabulary: {vocab['terms']}\n"
                f"Try this query: {vocab['sample_query']}\n\n"
                f"Original topic: {topic}"
            ),
            tools={},
            tool_schemas=[WEB_SEARCH_TOOL],
            model=MODEL_CHEAP,
            max_steps=4,
            title=f"sweep: {vocab['community']}",
        )
        return result.output

    sweeps = await asyncio.gather(*[sweep(v) for v in vocabs])

    return {"vocabs": vocabs, "premises": premises, "sweeps": sweeps}


async def reframe(topic: str, recon: dict) -> dict:
    """Phase 2: build a field map, then rewrite the question.

    Both calls go through flow.call so they appear in the live display
    and saved trace alongside the multi-call patterns.
    """
    vocabs_block = Block.join(
        [f"{v['community']}\n  terms: {v['terms']}\n  why: {v['why_this_community']}"
         for v in recon["vocabs"]]
    )
    premises_block = Block.join(
        [f"assumes: {p['assumption']}\n  if false: {p['what_breaks_if_false']}"
         for p in recon["premises"]]
    )
    sweeps_block = Block.join(
        recon["sweeps"],
        labels=[v["community"] for v in recon["vocabs"]],
    )

    field_map = await flow.call(
        FIELD_MAPPER | user(Block.join(
            [topic, vocabs_block, premises_block, sweeps_block],
            labels=["user asked", "vocabularies", "premises", "sweep results"],
        )),
        title="field map",
        model=MODEL,
    )

    reframed = await flow.call(
        REFRAMER | user(Block.join(
            [topic, field_map],
            labels=["original question", "field map"],
        )),
        schema=REFRAMED_SCHEMA,
        title="reframe",
        model=MODEL,
    )

    return {"field_map": field_map, "reframed": reframed}


async def research_angle(angle: dict) -> str:
    """One research agent on one angle."""
    result = await flow.agent(
        RESEARCHER | user(
            f"Research angle: {angle['name']}\n"
            f"Question: {angle['question']}\n"
            f"Why this matters: {angle['why']}"
        ),
        tools={},
        tool_schemas=[WEB_SEARCH_TOOL],
        model=MODEL,
        max_steps=8,
        title=angle["name"],
    )
    return result.output


async def critique_brief(angle: dict, brief: str) -> str:
    """One critic on one brief."""
    result = await flow.agent(
        CRITIC | user(
            f"Brief on '{angle['name']}' (question: {angle['question']}):\n\n"
            f"{brief}\n\n"
            "Attack this brief. Use web search to verify load-bearing claims."
        ),
        tools={},
        tool_schemas=[WEB_SEARCH_TOOL],
        model=MODEL,
        max_steps=5,
        title=f"critique: {angle['name']}",
    )
    return result.output


# --- Markdown emission helpers ---

def emit_recon(topic: str, recon: dict):
    """Emit reconnaissance phase to the markdown renderer."""
    flow.show(Section(title="Phase 1 — Reconnaissance", level=2))

    flow.show(Section(title="Vocabularies discovered", level=3))
    flow.show(Panels(
        items=[
            f"**Terms:** {v['terms']}\n\n"
            f"**Sample query:** `{v['sample_query']}`\n\n"
            f"**Why this community:** {v['why_this_community']}"
            for v in recon["vocabs"]
        ],
        titles=[v["community"] for v in recon["vocabs"]],
    ))

    flow.show(Section(title="Premises detected", level=3))
    flow.show(Panels(
        items=[
            f"**If false:** {p['what_breaks_if_false']}"
            for p in recon["premises"]
        ],
        titles=[p["assumption"] for p in recon["premises"]],
    ))

    flow.show(Section(title="Vocabulary sweeps", level=3))
    flow.show(Panels(
        items=recon["sweeps"],
        titles=[v["community"] for v in recon["vocabs"]],
    ))


def emit_reframe(reframe_result: dict):
    """Emit reframe phase to the markdown renderer."""
    flow.show(Section(title="Phase 2 — Reframe", level=2))
    flow.show(ContentBlock(title="Field map", content=reframe_result["field_map"]))

    r = reframe_result["reframed"]
    flow.show(ContentBlock(title="Reframed question", content=r["reframed_question"]))
    flow.show(ContentBlock(
        title="Mandated angles",
        content="\n".join(f"- {a}" for a in r["mandated_angles"]),
    ))
    flow.show(ContentBlock(
        title="Scope kept narrow (safety valve)",
        content="\n".join(f"- {s}" for s in r["scope_kept_narrow"]),
    ))
    flow.show(ContentBlock(
        title="Blind spots folded in",
        content="\n".join(f"- {b}" for b in r["blind_spots_folded_in"]),
    ))


def emit_research(angles: list[dict], briefs: list[str]):
    flow.show(Section(title="Phase 3 — Research", level=2))
    flow.show(Section(title="Angles", level=3))
    flow.show(Panels(
        items=[f"**Question:** {a['question']}\n\n**Why:** {a['why']}" for a in angles],
        titles=[a["name"] for a in angles],
    ))
    flow.show(Section(title="Research briefs", level=3))
    flow.show(Panels(items=briefs, titles=[a["name"] for a in angles]))


def emit_critiques(angles: list[dict], critiques: list[str]):
    flow.show(Section(title="Phase 4 — Adversarial critiques", level=2))
    flow.show(Panels(items=critiques, titles=[a["name"] for a in angles]))


def emit_synthesis(report: str):
    flow.show(Section(title="Phase 5 — Synthesis", level=2))
    flow.show(ContentBlock(title="Final report", content=report))


# --- File path helpers ---

def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[\s-]+", "-", s).strip("-")
    return s[:maxlen] or "research"


def default_output_path(topic: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"research-{slugify(topic)}-{stamp}.md")


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(description="Deep research v2 — reconnaissance + adversarial review")
    parser.add_argument("topic", nargs="+", help="Research topic / question")
    parser.add_argument("-o", "--output", default=None,
                        help="Markdown output file (default: research-<slug>-<timestamp>.md)")
    parser.add_argument("--trace", default=None,
                        help="JSON trace output (default: <output>.trace.json)")
    args = parser.parse_args()

    topic = " ".join(args.topic)
    out_path = Path(args.output) if args.output else default_output_path(topic)
    trace_path = Path(args.trace) if args.trace else out_path.with_suffix(".trace.json")

    trace = Trace()
    display = LiveFlowDisplay()
    cost = CostTracker()
    renderer = MarkdownRenderer()

    flow.observe(trace, display)
    llm.observe(cost)

    started = datetime.now()

    async with display, showing(renderer):
        # Document header
        flow.show(Section(title=f"Deep research v2", level=1))
        flow.show(ContentBlock(title="Topic", content=topic))
        flow.show(ContentBlock(title="Started", content=started.isoformat(timespec="seconds")))

        # Phase 1
        recon = await reconnaissance(topic)
        emit_recon(topic, recon)

        # Phase 2
        reframe_result = await reframe(topic, recon)
        reframed = reframe_result["reframed"]
        emit_reframe(reframe_result)

        # Phase 3
        mandated = "\n".join(f"- {a}" for a in reframed["mandated_angles"])
        scope_narrow = "\n".join(f"- {s}" for s in reframed["scope_kept_narrow"])
        decompose_input = Block.join(
            [reframed["reframed_question"], mandated, scope_narrow],
            labels=[
                "question",
                "angles you MUST include (from blind spot analysis)",
                "scope to preserve (do not drift)",
            ],
        )
        angles = await flow.branch(
            DECOMPOSER | user(decompose_input),
            schema=ANGLES_SCHEMA,
            model=MODEL_BRANCH,
            label_key="name",
            title="decompose",
        )
        briefs = await asyncio.gather(*[research_angle(a) for a in angles])
        emit_research(angles, briefs)

        # Phase 4
        critiques = await asyncio.gather(*[
            critique_brief(a, b) for a, b in zip(angles, briefs)
        ])
        emit_critiques(angles, critiques)

        # Phase 5
        labels = [a["name"] for a in angles]
        briefs_block = Block.join(briefs, labels=labels)
        critiques_block = Block.join(critiques, labels=labels)
        blind_spots_block = Block.join(reframed["blind_spots_folded_in"])

        synth_input = Block.join(
            [topic, reframed["reframed_question"], blind_spots_block,
             briefs_block, critiques_block],
            labels=[
                "user's original question (preserve this framing)",
                "reframed for context",
                "blind spots that should appear in the answer",
                "research briefs",
                "critiques of each brief",
            ],
        )
        report = await flow.call(
            SYNTHESIZER | user(synth_input),
            title="synthesis",
            model=MODEL,
        )
        emit_synthesis(report)

        # Run summary at the end of the document
        finished = datetime.now()
        flow.show(Section(title="Run summary", level=2))
        flow.show(ContentBlock(
            title="Stats",
            content=(
                f"- Wall time: {trace.total_elapsed:.1f}s\n"
                f"- LLM calls: {cost.calls}\n"
                f"- Input tokens: {cost.input_tokens:,}"
                f" (cache read: {cost.cache_read_tokens:,},"
                f" cache write: {cost.cache_creation_tokens:,})\n"
                f"- Output tokens: {cost.output_tokens:,}\n"
                f"- Estimated cost: ${cost.cost:.4f}\n"
                f"- Started: {started.isoformat(timespec='seconds')}\n"
                f"- Finished: {finished.isoformat(timespec='seconds')}"
            ),
        ))

    flow.clear_observers()
    llm.clear_observers()

    out_path.write_text(renderer.output())
    trace.save(str(trace_path))

    # Terminal: minimal — the live display already showed structure during the run
    print()
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"Wrote {trace_path}")
    print(f"{cost}")


if __name__ == "__main__":
    asyncio.run(main())
