"""Prompt composition primitives.

Three constructors: system, user, assistant
Two tool constructors: tool_use, tool_result
One operator: | (pipe)
Three verbs: complete, extract, act (in llm.py)

A Msg is an immutable sequence of segments. Segments are typed
dataclasses — TextSegment for prose, ToolCall for tool invocations,
ToolResult for tool outputs. | combines Msgs by concatenating segments.
render() at the boundary converts to API payloads.

The algebra: Msg is a monoid under |. The monoid operates on the
sequence, not the elements — segments can be any type as long as
render() knows what to do with them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# --- Segments: the elements of a Msg ---

@dataclass(frozen=True, slots=True)
class TextSegment:
    """A system, user, or assistant text segment."""
    role: str        # "system", "user", "assistant"
    text: str
    cache: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """An assistant's tool invocation."""
    id: str
    name: str
    input: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of executing a tool."""
    tool_use_id: str
    content: str
    is_error: bool = False


type Segment = TextSegment | ToolCall | ToolResult


# --- Block: text composition within a segment ---

class Block(str):
    """str subclass for composing text within a segment.

    + joins with paragraph separator, drops None/empty.
    Block is a monoid: + is associative, Block("") is the identity.
    """

    def __new__(cls, text=""):
        return super().__new__(cls, text)

    def __add__(self, other):
        if other is None or other == "":
            return self
        if isinstance(other, str):
            if not self:
                return Block(str(other))
            return Block(str(self) + "\n\n" + str(other))
        return NotImplemented

    def __radd__(self, other):
        if other is None or other == "":
            return self
        if isinstance(other, str):
            if not other:
                return self
            return Block(str(other) + "\n\n" + str(self))
        return NotImplemented

    def __repr__(self):
        text = str(self)
        if len(text) > 60:
            text = text[:57] + "..."
        return f"Block({text!r})"

    @staticmethod
    def join(items: list[str], *, labels: list[str] | None = None,
             sep: str = "\n\n") -> str:
        """Join multiple texts. Distinct from + (within-segment composition).

        Plain join:
            Block.join(results)

        Labeled join (for presenting multiple results to a synthesis call):
            Block.join(analyses, labels=["rhetoric", "logic", "psychology"])

        Custom separator:
            Block.join(chunks, sep="\\n---\\n")
        """
        if labels is not None:
            if len(labels) != len(items):
                raise ValueError(f"labels ({len(labels)}) must match items ({len(items)})")
            parts = [f"[{label}]:\n{text}" for label, text in zip(labels, items)]
        else:
            parts = list(items)

        return sep.join(p for p in parts if p)


# --- Template: deferred text ---

class Template:
    """Text with holes. Fill later, get a string.

        EVAL = Template("Evaluate {name}...")
        prompt = system(EVAL(name="Land"), cache=True) | user(context)
    """

    def __init__(self, text):
        self._text = text

    def __call__(self, **kw) -> str:
        return self._text.format_map(kw)

    def __repr__(self):
        text = self._text
        if len(text) > 60:
            text = text[:57] + "..."
        return f"Template({text!r})"


# --- Msg: the message ---

@dataclass(frozen=True, slots=True)
class Msg:
    """An immutable sequence of segments that becomes an LLM message.

    Segments are TextSegment, ToolCall, or ToolResult. Compose with |.
    Render at the boundary with render().

    Don't construct directly — use system(), user(), assistant(),
    tool_use(), tool_result().
    """
    segments: tuple[Segment, ...] = ()

    def __or__(self, other):
        if isinstance(other, Msg):
            return Msg(segments=self.segments + other.segments)
        return NotImplemented

    def __bool__(self):
        return len(self.segments) > 0

    def __repr__(self):
        parts = []
        for seg in self.segments:
            match seg:
                case TextSegment(role=role, text=text, cache=cache):
                    tag = " [cached]" if cache else ""
                    preview = text[:40] + "..." if len(text) > 40 else text
                    preview = preview.replace("\n", "\\n")
                    parts.append(f"  {role}{tag}: {preview}")
                case ToolCall(id=id, name=name):
                    parts.append(f"  tool_use: {name}({id})")
                case ToolResult(tool_use_id=tid, content=content):
                    preview = content[:40] + "..." if len(content) > 40 else content
                    parts.append(f"  tool_result({tid}): {preview}")
        return f"Msg(\n{chr(10).join(parts)}\n)" if parts else "Msg()"


# --- Constructors ---

def system(text: str, *, cache: bool = False) -> Msg:
    """A system segment. cache=True marks it for provider-side caching."""
    if not text:
        return Msg()
    return Msg(segments=(TextSegment("system", str(text), cache),))


def user(text: str) -> Msg:
    """A user segment."""
    if not text:
        return Msg()
    return Msg(segments=(TextSegment("user", str(text)),))


def assistant(text: str) -> Msg:
    """An assistant segment."""
    if not text:
        return Msg()
    return Msg(segments=(TextSegment("assistant", str(text)),))


def tool_use(id: str, name: str, input: dict | None = None) -> Msg:
    """A tool invocation segment (from the assistant)."""
    return Msg(segments=(ToolCall(id, name, input or {}),))


def tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> Msg:
    """A tool result segment (fed back to the model)."""
    return Msg(segments=(ToolResult(tool_use_id, str(content), is_error),))


# --- Render ---

def render(msg: Msg, *, backend: str = "anthropic") -> dict:
    """Convert a Msg into an API payload.

    This is the only function that knows about provider formats.
    Everything before render is just accumulating segments.

    Backends:
        anthropic: system content blocks with cache_control,
                   messages with text/tool_use/tool_result blocks.
        openai:    system message + messages array.
        flat:      system and prompt as plain strings.
    """
    match backend:
        case "anthropic":
            return _render_anthropic(msg)
        case "openai":
            return _render_openai(msg)
        case "flat":
            return _render_flat(msg)
        case _:
            raise ValueError(f"Unknown backend: {backend!r}")


def _render_anthropic(msg: Msg) -> dict:
    sys_content = []
    messages = []

    for seg in msg.segments:
        match seg:
            case TextSegment(role="system", text=text, cache=cache) if text:
                block = {"type": "text", "text": text}
                if cache:
                    block["cache_control"] = {"type": "ephemeral"}
                sys_content.append(block)

            case TextSegment(role=role, text=text) if text:
                _append_to_messages(messages, role, {"type": "text", "text": text})

            case ToolCall(id=id, name=name, input=input):
                _append_to_messages(messages, "assistant", {
                    "type": "tool_use", "id": id, "name": name, "input": input,
                })

            case ToolResult(tool_use_id=tid, content=content, is_error=is_err):
                block = {"type": "tool_result", "tool_use_id": tid, "content": content}
                if is_err:
                    block["is_error"] = True
                _append_to_messages(messages, "user", block)

    result = {"messages": messages}
    if sys_content:
        result["system"] = sys_content
    return result


def _append_to_messages(messages: list, role: str, content: dict):
    """Append a content block to messages, merging adjacent same-role."""
    if messages and messages[-1]["role"] == role:
        existing = messages[-1]["content"]
        if isinstance(existing, str):
            messages[-1]["content"] = [{"type": "text", "text": existing}, content]
        elif isinstance(existing, list):
            existing.append(content)
        else:
            messages[-1]["content"] = [existing, content]
    else:
        if content["type"] == "text":
            messages.append({"role": role, "content": content["text"]})
        else:
            messages.append({"role": role, "content": [content]})


def _render_openai(msg: Msg) -> dict:
    # OpenAI has no equivalent to ToolResult.is_error — dropped silently.
    sys_parts = []
    messages = []

    for seg in msg.segments:
        match seg:
            case TextSegment(role="system", text=text) if text:
                sys_parts.append(text)

            case TextSegment(role=role, text=text) if text:
                if messages and messages[-1]["role"] == role:
                    messages[-1]["content"] += "\n\n" + text
                else:
                    messages.append({"role": role, "content": text})

            case ToolCall(id=id, name=name, input=input):
                tool_call = {
                    "id": id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(input)},
                }
                if messages and messages[-1]["role"] == "assistant":
                    # Merge into existing assistant message — OpenAI requires
                    # content + tool_calls in one message, not two.
                    messages[-1].setdefault("tool_calls", []).append(tool_call)
                else:
                    messages.append({
                        "role": "assistant", "content": None,
                        "tool_calls": [tool_call],
                    })

            case ToolResult(tool_use_id=tid, content=content):
                messages.append({
                    "role": "tool", "tool_call_id": tid, "content": content,
                })

    if sys_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})
    return {"messages": messages}


def _render_flat(msg: Msg) -> dict:
    # Flat rendering discards assistant, tool_use, and tool_result segments.
    # Intended for simple prompt/completion interfaces (e.g., llm CLI).
    sys_parts = []
    user_parts = []

    for seg in msg.segments:
        match seg:
            case TextSegment(role="system", text=text) if text:
                sys_parts.append(text)
            case TextSegment(role="user", text=text) if text:
                user_parts.append(text)

    return {
        "system": "\n\n".join(sys_parts),
        "prompt": "\n\n".join(user_parts),
    }
