"""Edge cases for render() — complex Msg structures, no API calls needed."""

from motif import system, user, assistant, tool_use, tool_result, render, Msg
from motif.prompt import TextSegment, ToolCall, ToolResult as TR


class TestAnthropicEdges:
    def test_multiple_tool_calls_same_assistant_turn(self):
        """Two tool_use segments merge into one assistant message."""
        msg = tool_use("c1", "search", {"q": "a"}) | tool_use("c2", "calc", {"x": "1+1"})
        p = render(msg, backend="anthropic")
        assert len(p["messages"]) == 1
        assert p["messages"][0]["role"] == "assistant"
        content = p["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["name"] == "search"
        assert content[1]["name"] == "calc"

    def test_interleaved_tool_use_and_result(self):
        """tool_use (assistant) then tool_result (user) are separate messages."""
        msg = tool_use("c1", "search", {"q": "x"}) | tool_result("c1", "found it")
        p = render(msg, backend="anthropic")
        assert len(p["messages"]) == 2
        assert p["messages"][0]["role"] == "assistant"
        assert p["messages"][1]["role"] == "user"

    def test_text_then_tool_use_same_assistant(self):
        """Assistant text followed by tool_use merges into one assistant message."""
        msg = assistant("Let me search for that") | tool_use("c1", "search", {"q": "x"})
        p = render(msg, backend="anthropic")
        assert len(p["messages"]) == 1
        assert p["messages"][0]["role"] == "assistant"
        content = p["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"

    def test_full_agent_conversation(self):
        """A complete multi-turn agent conversation renders correctly."""
        msg = (
            system("You can search.", cache=True)
            | user("Find X")
            | assistant("I'll search for X")
            | tool_use("c1", "search", {"q": "X"})
            | tool_result("c1", "X is at Y")
            | assistant("Let me verify")
            | tool_use("c2", "search", {"q": "X at Y"})
            | tool_result("c2", "Confirmed: X is at Y")
            | assistant("X is at Y, confirmed by two searches.")
        )
        p = render(msg, backend="anthropic")
        assert "system" in p
        roles = [m["role"] for m in p["messages"]]
        # user, assistant (text+tool), user (result), assistant (text+tool), user (result), assistant
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]

    def test_tool_result_is_error(self):
        """is_error flag propagates in anthropic format."""
        msg = tool_result("c1", "404 not found", is_error=True)
        p = render(msg, backend="anthropic")
        content = p["messages"][0]["content"]
        assert content[0]["is_error"] is True

    def test_multiple_system_segments(self):
        """Multiple system segments become multiple content blocks."""
        msg = system("persona", cache=True) | system("scene context") | user("go")
        p = render(msg, backend="anthropic")
        assert len(p["system"]) == 2
        assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in p["system"][1]

    def test_no_system_segments(self):
        """Msg with no system segments has no 'system' key."""
        msg = user("hello")
        p = render(msg, backend="anthropic")
        assert "system" not in p

    def test_empty_msg_renders(self):
        """Empty Msg produces empty messages list."""
        p = render(Msg(), backend="anthropic")
        assert p == {"messages": []}


class TestOpenAIEdges:
    def test_multiple_tool_calls(self):
        """Multiple tool_use segments merge into one tool_calls array."""
        msg = tool_use("c1", "search", {"q": "a"}) | tool_use("c2", "calc", {"x": "1"})
        p = render(msg, backend="openai")
        assert len(p["messages"]) == 1
        assert len(p["messages"][0]["tool_calls"]) == 2

    def test_tool_result_role_is_tool(self):
        """OpenAI uses role 'tool' not 'user' for tool results."""
        msg = tool_result("c1", "answer")
        p = render(msg, backend="openai")
        assert p["messages"][0]["role"] == "tool"
        assert p["messages"][0]["tool_call_id"] == "c1"

    def test_system_merges_into_one(self):
        """Multiple system segments join into one system message."""
        msg = system("part 1") | system("part 2") | user("go")
        p = render(msg, backend="openai")
        assert p["messages"][0]["role"] == "system"
        assert "part 1" in p["messages"][0]["content"]
        assert "part 2" in p["messages"][0]["content"]

    def test_text_then_tool_use(self):
        """Assistant text + tool_use merge into one message (OpenAI requires this)."""
        msg = assistant("thinking...") | tool_use("c1", "fn", {})
        p = render(msg, backend="openai")
        # One message with both content and tool_calls
        assert len(p["messages"]) == 1
        assert p["messages"][0]["content"] == "thinking..."
        assert p["messages"][0]["tool_calls"][0]["function"]["name"] == "fn"
