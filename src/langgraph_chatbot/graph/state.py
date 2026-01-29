"""
State management for the LangGraph chatbot.

Defines the conversation state structure and metadata tracking.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    State definition for the chatbot conversation.

    Attributes
    ----------
    messages : list[BaseMessage]
        Conversation history with message reducer
    conversation_title : Optional[str]
        Human-readable title for the conversation
    allowed_tools : Optional[list[str]]
        List of tool names allowed for current turn (set by intent classifier)
    human_in_loop_enabled : Optional[bool]
        Whether human-in-the-loop approval is enabled for this conversation
    needs_approval : Optional[bool]
        Whether the current tool calls need human approval
    pending_tool_calls : Optional[list[dict]]
        Tool calls awaiting approval
    approved_tools : Optional[list[str]]
        Tool names that have been approved by user
    rejected_tools : Optional[list[str]]
        Tool names that have been rejected by user
    """

    messages: Annotated[list[BaseMessage], add_messages]
    conversation_title: Optional[str]
    allowed_tools: Optional[list[str]]
    human_in_loop_enabled: Optional[bool]
    needs_approval: Optional[bool]
    pending_tool_calls: Optional[list[dict]]
    approved_tools: Optional[list[str]]
    rejected_tools: Optional[list[str]]
