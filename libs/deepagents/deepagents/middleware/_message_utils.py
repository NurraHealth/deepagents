"""Utilities for extracting content from LangChain messages.

This module provides helpers for safely extracting text content from various
LangChain message types, handling differences between providers (OpenAI, Anthropic).

See: https://github.com/langchain-ai/deepagents/issues/979
"""


def extract_message_content(message) -> str:
    """Extract text content from a LangChain message.

    LangChain messages use `.content` (not `.text`) for message content.
    This function handles the various content formats:

    - String content (most common)
    - List content (Anthropic-style content blocks)
    - Legacy `.text` attribute (fallback)

    Args:
        message: A LangChain message object (AIMessage, HumanMessage, ToolMessage, etc.)

    Returns:
        The text content as a string. Returns empty string if no text content found.
    """
    # Get content, trying .content first (LangChain standard), then .text (legacy)
    content = getattr(message, "content", None)
    if content is None:
        content = getattr(message, "text", None)
    if content is None:
        return str(message)

    # Handle string content directly
    if isinstance(content, str):
        return content

    # Handle list content (Anthropic-style content blocks)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                # Content blocks: {"type": "text", "text": "..."} or {"type": "tool_use", ...}
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    # Fallback
    return str(content)
