"""motif — a prompt algebra for LLM orchestration.

    from motif import system, user, assistant, tool_use, tool_result
    from motif import llm, flow

    # Build prompts
    prompt = system(persona, cache=True) | user(context)

    # Three verbs
    text = await llm.complete(prompt)
    data = await llm.extract(prompt, schema=SCHEMA)
    result = await llm.act(prompt, tools=TOOLS)

    # Flow patterns
    items = await flow.branch(prompt, schema=ITEMS_SCHEMA)
    results = await flow.fan(items, lambda i: analyst | user(str(i)))
    synthesis = await flow.reduce(results, lambda t: synth | user(t))
"""

from .prompt import (
    system, user, assistant, tool_use, tool_result,
    Block, Msg, Template, render,
    TextSegment, ToolCall, ToolResult,
)

__all__ = [
    # Constructors
    "system", "user", "assistant", "tool_use", "tool_result",
    # Composition
    "Block", "Msg", "Template",
    # Segment types
    "TextSegment", "ToolCall", "ToolResult",
    # Rendering
    "render",
]
