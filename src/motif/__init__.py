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
from .graph import Node
from .llm import (
    Endpoint, RoleRef, Truncated, role, use_profile,
    CallStarted, CallChunk, CallCompleted, CallFailed,
)
from . import record as _record  # installs the graph projection (llm._projection)
from . import flow as _flow  # registers flow/show observer scopes with graph.session

__all__ = [
    # Constructors
    "system", "user", "assistant", "tool_use", "tool_result",
    # Composition
    "Block", "Msg", "Template",
    # Segment types
    "TextSegment", "ToolCall", "ToolResult",
    # Graph
    "Node",
    # Rendering
    "render",
    # Endpoints and roles
    "Endpoint", "RoleRef", "role", "use_profile",
    # Call-lifecycle events (llm.observe_calls)
    "CallStarted", "CallChunk", "CallCompleted", "CallFailed",
    # Errors
    "Truncated",
]
