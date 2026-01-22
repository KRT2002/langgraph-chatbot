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
    pending_tool_approval : Optional[dict]
        Tool call awaiting user approval
    conversation_title : Optional[str]
        Human-readable title for the conversation
    allowed_tools : Optional[list[str]]
        List of tool names allowed for current turn (set by intent classifier)
    """
    
    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool_approval: Optional[dict]
    conversation_title: Optional[str]
    allowed_tools: Optional[list[str]]