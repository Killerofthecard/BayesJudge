"""
Conversation formatting utilities.
"""

from typing import List, Dict


def format_conversation(conv: List[Dict[str, str]]) -> str:
    """
    Format a conversation list as a string.

    Args:
        conv: List of turns, each containing 'role' and 'content'.

    Returns:
        Formatted conversation string.
    """
    lines = []
    for turn in conv:
        role = turn.get('role', '').upper()
        content = turn.get('content', '')
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


def truncate_conversation(conv: List[Dict[str, str]], max_length: int = 2000) -> str:
    """
    Truncate a conversation to a maximum length.

    Args:
        conv: Conversation list.
        max_length: Maximum number of characters.

    Returns:
        Truncated conversation string.
    """
    formatted = format_conversation(conv)
    if len(formatted) > max_length:
        return formatted[:max_length] + "\n... [content truncated]"
    return formatted


def format_for_comparison(conv_a: List[Dict], conv_b: List[Dict]) -> tuple:
    """
    Format two conversations as a pair for comparison.

    Returns:
        (str_a, str_b): Two formatted conversation strings.
    """
    return format_conversation(conv_a), format_conversation(conv_b)
