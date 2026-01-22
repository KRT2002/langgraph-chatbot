"""
Streamlit UI for the LangGraph Chatbot.
"""

import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_chatbot.config import settings
from langgraph_chatbot.graph.workflow import chatbot, retrieve_all_threads
from langgraph_chatbot.utils.analytics import calculate_conversation_stats, format_stats_for_display
from langgraph_chatbot.utils.export import (
    export_conversation_json,
    export_conversation_markdown,
    export_conversation_pdf,
)
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title="LangGraph Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)


# =========================== Utilities ===========================


def generate_conversation_title(first_message: str) -> str:
    """Generate a meaningful title from the first user message."""
    title = first_message[:50].strip()
    if len(first_message) > 50:
        title += "..."
    return title


def reset_chat():
    """Start a new conversation."""
    thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["conversation_title"] = "New Conversation"
    st.session_state["stats"] = None
    logger.info(f"New chat started: {thread_id}")


def load_conversation(thread_id: str):
    """Load an existing conversation."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])

        # Generate conversation title from first message
        title = state.values.get("conversation_title", "Conversation")
        if not title or title == "Conversation":
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    title = generate_conversation_title(msg.content)
                    break

        return messages, title
    except Exception as e:
        logger.error(f"Error loading conversation: {str(e)}")
        return [], "Error Loading"


def save_conversation_title(thread_id: str, title: str):
    """Save conversation title to state."""
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Update state with title
        chatbot.update_state(config, {"conversation_title": title})
        logger.info(f"Saved conversation title: {title}")
    except Exception as e:
        logger.error(f"Error saving title: {str(e)}")


# ======================= Session Initialization ===================

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


# ============================ Sidebar ============================

with st.sidebar:
    st.title("🤖 LangGraph Chatbot")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    # Human-in-Loop Toggle
    st.session_state["human_in_loop_enabled"] = st.toggle(
        "🔒 Human-in-Loop",
        value=st.session_state["human_in_loop_enabled"],
        help="Require approval before executing sensitive tools",
    )

    # Analytics Toggle
    analytics_toggled = st.toggle(
        "📊 Show Analytics",
        value=st.session_state["show_analytics"],
        help="Display conversation statistics",
    )

    if analytics_toggled != st.session_state["show_analytics"]:
        st.session_state["show_analytics"] = analytics_toggled
        st.rerun()

    st.divider()

    # Export Options
    with st.expander("💾 Export Conversation", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("JSON", use_container_width=True):
                if st.session_state["message_history"]:
                    try:
                        messages, _ = load_conversation(st.session_state["thread_id"])
                        filepath = export_conversation_json(
                            messages,
                            st.session_state["thread_id"],
                            st.session_state["conversation_title"],
                        )
                        st.success(f"Exported to:\n{Path(filepath).name}")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                else:
                    st.warning("No messages to export")

        with col2:
            if st.button("MD", use_container_width=True):
                if st.session_state["message_history"]:
                    try:
                        messages, _ = load_conversation(st.session_state["thread_id"])
                        filepath = export_conversation_markdown(
                            messages,
                            st.session_state["thread_id"],
                            st.session_state["conversation_title"],
                        )
                        st.success(f"Exported to:\n{Path(filepath).name}")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                else:
                    st.warning("No messages to export")

        with col3:
            if st.button("PDF", use_container_width=True):
                if st.session_state["message_history"]:
                    try:
                        messages, _ = load_conversation(st.session_state["thread_id"])
                        filepath = export_conversation_pdf(
                            messages,
                            st.session_state["thread_id"],
                            st.session_state["conversation_title"],
                        )
                        st.success(f"Exported to:\n{Path(filepath).name}")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                else:
                    st.warning("No messages to export")

    st.divider()

    # Analytics Display
    if st.session_state["show_analytics"]:
        with st.expander("📊 Statistics", expanded=True):
            if st.session_state["message_history"]:
                messages, _ = load_conversation(st.session_state["thread_id"])
                stats = calculate_conversation_stats(messages)
                st.markdown(format_stats_for_display(stats))
            else:
                st.info("No messages yet")

    st.divider()

    # Conversation History
    st.subheader("💬 Conversations")

    # Refresh threads button
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state["chat_threads"] = retrieve_all_threads(chatbot)
        st.rerun()

    # Display threads
    for thread_id in reversed(st.session_state["chat_threads"]):
        messages, title = load_conversation(thread_id)

        # Use title or truncated thread ID
        display_name = title if title and title != "Conversation" else thread_id[:20]

        if st.button(f"💬 {display_name}", key=f"thread_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            st.session_state["conversation_title"] = title

            # Convert messages to display format
            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    temp_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    temp_messages.append({"role": "assistant", "content": msg.content})

            st.session_state["message_history"] = temp_messages
            st.rerun()


# ============================ Main UI ============================

# Display conversation title
st.title(st.session_state["conversation_title"])

# Render message history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Generate title from first message
    if len(st.session_state["message_history"]) == 0:
        title = generate_conversation_title(user_input)
        st.session_state["conversation_title"] = title
        save_conversation_title(st.session_state["thread_id"], title)

    # Display user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Configuration
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
    }

    # Assistant response
    with st.chat_message("assistant"):
        status_container = st.empty()
        message_container = st.empty()

        response_content = ""
        tool_used = False
        intent_analyzing = True

        try:
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # Get the node that emitted this message
                node_name = metadata.get("langgraph_node", "")

                # Show intent classification indicator (once)
                if intent_analyzing and node_name == "intent_classifier":
                    status_container.info("🔍 Analyzing query...")
                    intent_analyzing = False

                # Skip messages from intent_classifier (internal only)
                if node_name == "intent_classifier":
                    continue

                # Clear intent indicator when chat_node starts
                if node_name == "chat_node" and not tool_used:
                    status_container.empty()

                # Handle tool execution
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "unknown")
                    tool_used = True

                    # Check if human approval is needed
                    if (
                        st.session_state["human_in_loop_enabled"]
                        and tool_name in settings.tools_requiring_approval
                    ):

                        status_container.warning(
                            f"🔒 Tool `{tool_name}` requires approval. "
                            f"(Human-in-Loop is enabled)"
                        )
                    else:
                        status_container.info(f"🔧 Using tool: `{tool_name}`")

                # Stream AI response (only from chat_node)
                if isinstance(message_chunk, AIMessage) and node_name == "chat_node":
                    response_content += message_chunk.content
                    message_container.markdown(response_content)

            # Clear status after completion
            if tool_used:
                status_container.success("✅ Tool execution complete")
                time.sleep(1)
                status_container.empty()
            else:
                # Clear intent indicator if still showing
                status_container.empty()

            # Save assistant message
            st.session_state["message_history"].append(
                {"role": "assistant", "content": response_content}
            )

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.session_state["message_history"].append({"role": "assistant", "content": error_msg})
