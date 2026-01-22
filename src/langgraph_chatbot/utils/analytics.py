"""
Analytics utilities for conversation statistics.
"""

from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_conversation_stats(messages: list[Any]) -> dict:
    """
    Calculate statistics for a conversation.
    
    Parameters
    ----------
    messages : list[Any]
        List of conversation messages
        
    Returns
    -------
    dict
        Statistics including message counts, tool usage, etc.
    """
    try:
        stats = {
            "total_messages": len(messages),
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_calls": 0,
            "tools_used": Counter(),
            "total_tokens": 0,  # Approximate
        }
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                stats["user_messages"] += 1
                stats["total_tokens"] += len(msg.content.split())
            elif isinstance(msg, AIMessage):
                stats["assistant_messages"] += 1
                stats["total_tokens"] += len(msg.content.split())
            elif isinstance(msg, ToolMessage):
                stats["tool_calls"] += 1
                tool_name = getattr(msg, "name", "unknown")
                stats["tools_used"][tool_name] += 1
        
        # Convert Counter to dict for JSON serialization
        stats["tools_used"] = dict(stats["tools_used"])
        
        logger.info(f"Calculated stats: {stats['total_messages']} messages, {stats['tool_calls']} tool calls")
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating stats: {str(e)}", exc_info=True)
        return {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_calls": 0,
            "tools_used": {},
            "total_tokens": 0,
        }


def format_stats_for_display(stats: dict) -> str:
    """
    Format conversation statistics for display.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from calculate_conversation_stats
        
    Returns
    -------
    str
        Formatted statistics string
    """
    lines = [
        f"**Total Messages:** {stats['total_messages']}",
        f"**User Messages:** {stats['user_messages']}",
        f"**Assistant Messages:** {stats['assistant_messages']}",
        f"**Tool Calls:** {stats['tool_calls']}",
        f"**Approx. Tokens:** {stats['total_tokens']}",
    ]
    
    if stats["tools_used"]:
        lines.append("\n**Tools Used:**")
        for tool, count in stats["tools_used"].items():
            lines.append(f"  - {tool}: {count}x")
    
    return "\n".join(lines)