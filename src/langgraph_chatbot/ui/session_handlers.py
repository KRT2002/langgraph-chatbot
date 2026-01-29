"""
Session and conversation management for the Streamlit chatbot.
"""

from datetime import datetime

import streamlit as st

from langgraph_chatbot.graph.workflow import chatbot, retrieve_all_threads
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_conversation_title(first_message: str) -> str:
    """
    Generate a meaningful title from the first user message.

    Parameters
    ----------
    first_message : str
        The first user message in the conversation

    Returns
    -------
    str
        Truncated title (max 50 chars + ellipsis if longer)
    """
    title = first_message[:50].strip()
    if len(first_message) > 50:
        title += "..."
    return title


def reset_chat():
    """
    Start a new conversation.

    Creates a new thread ID with timestamp and initializes all session state
    variables for a fresh conversation.

    Notes
    -----
    - Thread ID format: chat_YYYYMMDD_HHMMSS
    - Clears message history and approval state
    - Logs the new chat thread creation
    """
    thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["conversation_title"] = "New Conversation"
    st.session_state["stats"] = None
    st.session_state["awaiting_approval"] = False
    st.session_state["approval_decisions"] = {}
    logger.info(f"New chat started: {thread_id}")


def load_conversation(thread_id: str):
    """
    Load an existing conversation from LangGraph state.

    Parameters
    ----------
    thread_id : str
        The thread ID of the conversation to load

    Returns
    -------
    tuple[list, str]
        A tuple of (messages, title) where:
        - messages: List of message objects from the conversation
        - title: Conversation title (generated from first message if not stored)

    Notes
    -----
    - Returns empty list and "Error Loading" if loading fails
    - Automatically generates title from first HumanMessage if not present
    """
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])

        # Generate conversation title from first message
        title = state.values.get("conversation_title", "Conversation")
        if not title or title == "Conversation":
            from langchain_core.messages import HumanMessage

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    title = generate_conversation_title(msg.content)
                    break

        return messages, title
    except Exception as e:
        logger.error(f"Error loading conversation: {str(e)}")
        return [], "Error Loading"


def save_conversation_title(thread_id: str, title: str):
    """
    Save conversation title to LangGraph state.

    Parameters
    ----------
    thread_id : str
        The thread ID of the conversation
    title : str
        The title to save for the conversation

    Notes
    -----
    - Updates the conversation_title field in LangGraph state
    - Logs success or error messages
    - Errors are caught and logged but do not raise exceptions
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Update state with title
        chatbot.update_state(config, {"conversation_title": title})
        logger.info(f"Saved conversation title: {title}")
    except Exception as e:
        logger.error(f"Error saving title: {str(e)}")


def initialize_session_state():
    """
    Initialize all session state variables for the Streamlit app.

    Sets up the following session state variables if they don't exist:
    - message_history: List of chat messages
    - thread_id: Current conversation thread identifier
    - chat_threads: List of all available conversation threads
    - conversation_title: Title of current conversation
    - human_in_loop_enabled: Toggle for approval requirement
    - stats: Conversation statistics
    - show_analytics: Toggle for analytics display
    - awaiting_approval: Flag for pending tool approvals
    - approval_decisions: User decisions on tool approvals

    Notes
    -----
    - Safe to call multiple times (checks existence before initialization)
    - Calls reset_chat() if thread_id doesn't exist
    - Retrieves all threads from chatbot workflow
    """
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "thread_id" not in st.session_state:
        reset_chat()

    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = retrieve_all_threads(chatbot)

    if "conversation_title" not in st.session_state:
        st.session_state["conversation_title"] = "New Conversation"

    if "human_in_loop_enabled" not in st.session_state:
        st.session_state["human_in_loop_enabled"] = False

    if "stats" not in st.session_state:
        st.session_state["stats"] = None

    if "show_analytics" not in st.session_state:
        st.session_state["show_analytics"] = False

    if "awaiting_approval" not in st.session_state:
        st.session_state["awaiting_approval"] = False

    if "approval_decisions" not in st.session_state:
        st.session_state["approval_decisions"] = {}
