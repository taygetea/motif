"""Property-based tests for the algebraic structure.

Msg is a monoid: | is associative, Msg() is the identity.
Block is a monoid: + is associative, Block("") is the identity.
render is a homomorphism from Msg to API payloads.

These are the properties that make composition work.
If they break, the library's fundamental guarantee is gone.
"""

import pytest
from hypothesis import given, strategies as st, assume

from motif.prompt import (
    system, user, assistant, tool_use, tool_result,
    Msg, Block, TextSegment, ToolCall, ToolResult, render,
)


# --- Strategies ---

# Generate arbitrary text segments
text_segment = st.builds(
    TextSegment,
    role=st.sampled_from(["system", "user", "assistant"]),
    text=st.text(min_size=0, max_size=200),
    cache=st.booleans(),
)

# Generate arbitrary tool calls
tool_call_seg = st.builds(
    ToolCall,
    id=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
    name=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    input=st.fixed_dictionaries({"key": st.text(max_size=50)}),
)

# Generate arbitrary tool results
tool_result_seg = st.builds(
    ToolResult,
    tool_use_id=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
    content=st.text(min_size=0, max_size=200),
    is_error=st.booleans(),
)

# Any segment
any_segment = st.one_of(text_segment, tool_call_seg, tool_result_seg)

# Generate arbitrary Msgs
msg_strategy = st.builds(
    Msg,
    segments=st.tuples(any_segment, any_segment, any_segment).map(tuple)
    | st.tuples(any_segment).map(tuple)
    | st.just(()),
)

# Generate non-empty text for Block
block_text = st.text(min_size=0, max_size=200)


# --- Msg monoid laws ---

class TestMsgMonoid:
    """Msg under | must be a monoid."""

    @given(a=msg_strategy, b=msg_strategy, c=msg_strategy)
    def test_associativity(self, a, b, c):
        """(a | b) | c == a | (b | c)"""
        assert (a | b) | c == a | (b | c)

    @given(a=msg_strategy)
    def test_left_identity(self, a):
        """Msg() | a == a"""
        assert Msg() | a == a

    @given(a=msg_strategy)
    def test_right_identity(self, a):
        """a | Msg() == a"""
        assert a | Msg() == a

    @given(a=msg_strategy, b=msg_strategy)
    def test_pipe_is_segment_concatenation(self, a, b):
        """| is tuple concatenation of segments."""
        result = a | b
        assert result.segments == a.segments + b.segments


# --- Block monoid laws ---

class TestBlockMonoid:
    """Block under + must be a monoid."""

    @given(a=block_text, b=block_text, c=block_text)
    def test_associativity(self, a, b, c):
        """(Block(a) + Block(b)) + Block(c) == Block(a) + (Block(b) + Block(c))
        Universal — holds for empty and non-empty strings."""
        left = (Block(a) + Block(b)) + Block(c)
        right = Block(a) + (Block(b) + Block(c))
        assert left == right

    @given(a=block_text)
    def test_left_identity(self, a):
        """Block("") + Block(a) == Block(a)"""
        assert Block("") + Block(a) == Block(a)

    @given(a=block_text)
    def test_right_identity(self, a):
        """Block(a) + Block("") == Block(a)"""
        assert Block(a) + Block("") == Block(a)

    def test_none_drops(self):
        """Block + None == Block (None is absorbed)."""
        b = Block("hello")
        assert b + None == b

    def test_empty_drops(self):
        """Block + "" == Block (empty absorbed)."""
        b = Block("hello")
        assert b + "" == b

    @given(a=block_text, b=block_text)
    def test_join_is_paragraph_separated(self, a, b):
        """Non-empty blocks join with paragraph separator."""
        assume(a and b)
        result = Block(a) + Block(b)
        assert str(result) == f"{a}\n\n{b}"


# --- Constructors produce valid Msgs ---

class TestConstructors:
    def test_system_produces_msg(self):
        m = system("hello")
        assert len(m.segments) == 1
        assert isinstance(m.segments[0], TextSegment)
        assert m.segments[0].role == "system"
        assert m.segments[0].text == "hello"

    def test_user_produces_msg(self):
        m = user("hello")
        assert m.segments[0].role == "user"

    def test_assistant_produces_msg(self):
        m = assistant("hello")
        assert m.segments[0].role == "assistant"

    def test_tool_use_produces_msg(self):
        m = tool_use("id1", "search", {"q": "test"})
        assert len(m.segments) == 1
        assert isinstance(m.segments[0], ToolCall)
        assert m.segments[0].name == "search"
        assert m.segments[0].input == {"q": "test"}

    def test_tool_result_produces_msg(self):
        m = tool_result("id1", "found it")
        assert isinstance(m.segments[0], ToolResult)
        assert m.segments[0].content == "found it"

    def test_empty_system_produces_empty_msg(self):
        assert system("") == Msg()

    def test_empty_user_produces_empty_msg(self):
        assert user("") == Msg()

    def test_empty_vanishes_on_compose(self):
        """Empty strings produce empty Msgs that vanish on |."""
        m = system("persona") | system("") | user("hello")
        assert len(m.segments) == 2


# --- Render homomorphism ---

class TestRender:
    """render() should be a homomorphism: it preserves structure."""

    def test_anthropic_basic(self):
        msg = system("sys", cache=True) | user("hello")
        p = render(msg, backend="anthropic")
        assert "system" in p
        assert p["system"][0]["text"] == "sys"
        assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert p["messages"][0] == {"role": "user", "content": "hello"}

    def test_openai_basic(self):
        msg = system("sys") | user("hello")
        p = render(msg, backend="openai")
        assert p["messages"][0] == {"role": "system", "content": "sys"}
        assert p["messages"][1] == {"role": "user", "content": "hello"}

    def test_anthropic_adjacent_merge(self):
        """Adjacent same-role text segments merge into one message."""
        msg = user("a") | user("b")
        p = render(msg, backend="anthropic")
        assert len(p["messages"]) == 1
        # Adjacent text segments become a list of content blocks
        content = p["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["text"] == "a"
        assert content[1]["text"] == "b"

    def test_anthropic_tool_call(self):
        msg = tool_use("id1", "search", {"q": "test"})
        p = render(msg, backend="anthropic")
        blocks = p["messages"][0]["content"]
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "search"

    def test_anthropic_tool_result(self):
        msg = tool_result("id1", "found it")
        p = render(msg, backend="anthropic")
        blocks = p["messages"][0]["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["content"] == "found it"

    def test_anthropic_tool_result_error(self):
        msg = tool_result("id1", "failed", is_error=True)
        p = render(msg, backend="anthropic")
        blocks = p["messages"][0]["content"]
        assert blocks[0]["is_error"] is True

    def test_openai_tool_call(self):
        msg = tool_use("id1", "search", {"q": "test"})
        p = render(msg, backend="openai")
        assert p["messages"][0]["tool_calls"][0]["function"]["name"] == "search"

    def test_openai_tool_result(self):
        msg = tool_result("id1", "found it")
        p = render(msg, backend="openai")
        assert p["messages"][0]["role"] == "tool"
        assert p["messages"][0]["tool_call_id"] == "id1"

    def test_flat_discards_tools(self):
        """Flat backend only keeps system and user text."""
        msg = system("sys") | user("q") | tool_use("id", "fn", {}) | assistant("a")
        p = render(msg, backend="flat")
        assert p["system"] == "sys"
        assert p["prompt"] == "q"

    def test_full_conversation_renders(self):
        """A complete tool-use conversation renders without error."""
        msg = (
            system("You can search.", cache=True)
            | user("Find X")
            | tool_use("c1", "search", {"q": "X"})
            | tool_result("c1", "Found X at Y")
            | assistant("X is at Y.")
        )
        for backend in ("anthropic", "openai", "flat"):
            p = render(msg, backend=backend)
            assert "messages" in p or "prompt" in p


# --- Block.join ---

class TestBlockJoin:
    def test_plain_join(self):
        assert Block.join(["a", "b", "c"]) == "a\n\nb\n\nc"

    def test_labeled_join(self):
        result = Block.join(["text1", "text2"], labels=["A", "B"])
        assert "[A]:" in result
        assert "[B]:" in result
        assert "text1" in result

    def test_empty_filtering(self):
        assert Block.join(["a", "", "c"]) == "a\n\nc"

    def test_labels_length_mismatch(self):
        with pytest.raises(ValueError):
            Block.join(["a", "b"], labels=["only_one"])

    def test_custom_separator(self):
        assert Block.join(["a", "b"], sep="\n---\n") == "a\n---\nb"
