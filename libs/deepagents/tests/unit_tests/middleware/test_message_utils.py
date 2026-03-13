"""Unit tests for message utility functions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware._message_utils import extract_message_content


class TestExtractMessageContent:
    """Tests for extract_message_content function."""

    def test_string_content(self) -> None:
        """Test extraction from message with string content."""
        message = AIMessage(content="Hello, world!")
        assert extract_message_content(message) == "Hello, world!"

    def test_empty_string_content(self) -> None:
        """Test extraction from message with empty string content."""
        message = AIMessage(content="")
        assert extract_message_content(message) == ""

    def test_human_message(self) -> None:
        """Test extraction from HumanMessage."""
        message = HumanMessage(content="User query")
        assert extract_message_content(message) == "User query"

    def test_tool_message(self) -> None:
        """Test extraction from ToolMessage."""
        message = ToolMessage(content="Tool result", tool_call_id="123")
        assert extract_message_content(message) == "Tool result"

    def test_list_content_with_text_block(self) -> None:
        """Test extraction from Anthropic-style content blocks with text."""
        message = AIMessage(content=[{"type": "text", "text": "Hello from Anthropic"}])
        assert extract_message_content(message) == "Hello from Anthropic"

    def test_list_content_with_multiple_text_blocks(self) -> None:
        """Test extraction from multiple text blocks."""
        message = AIMessage(
            content=[
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
            ]
        )
        assert extract_message_content(message) == "First part\nSecond part"

    def test_list_content_with_mixed_blocks(self) -> None:
        """Test extraction ignores non-text blocks (e.g., tool_use)."""
        message = AIMessage(
            content=[
                {"type": "text", "text": "I will use a tool"},
                {"type": "tool_use", "id": "123", "name": "search", "input": {}},
                {"type": "text", "text": "After the tool"},
            ]
        )
        assert extract_message_content(message) == "I will use a tool\nAfter the tool"

    def test_list_content_with_only_tool_use(self) -> None:
        """Test extraction when content is only tool_use blocks (no text)."""
        message = AIMessage(
            content=[
                {"type": "tool_use", "id": "123", "name": "search", "input": {}},
            ]
        )
        assert extract_message_content(message) == ""

    def test_list_content_with_string_items(self) -> None:
        """Test extraction from list with plain string items."""
        message = AIMessage(content=["Hello", "World"])
        assert extract_message_content(message) == "Hello\nWorld"

    def test_list_content_empty(self) -> None:
        """Test extraction from empty list content."""
        message = AIMessage(content=[])
        assert extract_message_content(message) == ""

    def test_legacy_text_attribute(self) -> None:
        """Test fallback to .text attribute when .content is None."""

        class LegacyMessage:
            content = None
            text = "Legacy text"

        message = LegacyMessage()
        assert extract_message_content(message) == "Legacy text"

    def test_fallback_to_str(self) -> None:
        """Test fallback to str() when neither .content nor .text exists."""

        class CustomMessage:
            def __str__(self) -> str:
                return "Custom string representation"

        message = CustomMessage()
        assert extract_message_content(message) == "Custom string representation"

    def test_content_with_text_attr_block(self) -> None:
        """Test extraction from list with objects that have .text attribute."""

        class TextBlock:
            def __init__(self, text: str) -> None:
                self.text = text

        # Can't use AIMessage directly as it validates content types,
        # so we create a mock message object
        class MockMessage:
            content = [TextBlock("Block text")]

        message = MockMessage()
        assert extract_message_content(message) == "Block text"

    def test_non_string_content_fallback(self) -> None:
        """Test that non-string, non-list content falls back to str()."""

        class WeirdMessage:
            content = 12345

        message = WeirdMessage()
        assert extract_message_content(message) == "12345"
