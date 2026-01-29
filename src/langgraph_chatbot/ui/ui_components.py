"""
Reusable UI components for the Streamlit chatbot.
"""

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_chatbot.graph.workflow import chatbot, retrieve_all_threads
from langgraph_chatbot.utils.analytics import calculate_conversation_stats, format_stats_for_display
from langgraph_chatbot.utils.export import (
    export_conversation_json,
    export_conversation_markdown,
    export_conversation_pdf,
)
from langgraph_chatbot.utils.logger import setup_logger
from langgraph_chatbot.utils.tool_descriptions import describe_tool_call

from langgraph_chatbot.ui.approval_handlers import handle_approval_and_resume, handle_auto_resume
from langgraph_chatbot.ui.session_handlers import load_conversation, reset_chat

logger = setup_logger(__name__)


def render_sidebar():
    """Render the sidebar with controls and conversation history."""
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
                st.session_state["awaiting_approval"] = False
                st.session_state["approval_decisions"] = {}

                # Convert messages to display format
                temp_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        temp_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        temp_messages.append({"role": "assistant", "content": msg.content})

                st.session_state["message_history"] = temp_messages
                st.rerun()


def render_chat_messages():
    """Render the message history."""
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_approval_ui(config):
    """
    Render the approval UI for pending tool executions.

    Parameters
    ----------
    config : dict
        LangGraph configuration with thread_id
    """
    state = chatbot.get_state(config)
    pending_tools = state.values.get("pending_tool_calls", [])

    # Case 1: Has pending tools (needs user approval)
    if pending_tools:
        st.warning("⚠️ The following tools require your approval:")

        # Create containers for streaming after approval
        with st.chat_message("assistant"):
            status_container = st.empty()
            message_container = st.empty()

        # Batch approval buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve All", use_container_width=True, type="primary"):
                approved = [tc["name"] for tc in pending_tools]
                response_content, has_more_interrupts = handle_approval_and_resume(
                    config, approved, [], status_container, message_container
                )

                if response_content:
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": response_content}
                    )

                if not has_more_interrupts:
                    st.session_state["awaiting_approval"] = False

                st.rerun()

        with col2:
            if st.button("❌ Reject All", use_container_width=True):
                rejected = [tc["name"] for tc in pending_tools]
                response_content, has_more_interrupts = handle_approval_and_resume(
                    config, [], rejected, status_container, message_container
                )

                if response_content:
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": response_content}
                    )

                if not has_more_interrupts:
                    st.session_state["awaiting_approval"] = False

                st.rerun()

        st.divider()

        # Individual tool approval
        for idx, tc in enumerate(pending_tools):
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            task_desc = describe_tool_call(tool_name, tool_args)

            with st.container():
                st.markdown(f"**{idx + 1}. 🔧 {tool_name}**")
                st.text(f"Task: {task_desc}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "✅ Approve", key=f"approve_{tool_name}_{idx}", use_container_width=True
                    ):
                        st.session_state["approval_decisions"][tool_name] = "approved"

                with col2:
                    if st.button(
                        "❌ Reject", key=f"reject_{tool_name}_{idx}", use_container_width=True
                    ):
                        st.session_state["approval_decisions"][tool_name] = "rejected"

                st.divider()

        # Submit individual decisions
        if st.session_state["approval_decisions"]:
            if st.button("Submit Decisions", type="primary", use_container_width=True):
                approved = [
                    name
                    for name, decision in st.session_state["approval_decisions"].items()
                    if decision == "approved"
                ]
                rejected = [
                    name
                    for name, decision in st.session_state["approval_decisions"].items()
                    if decision == "rejected"
                ]

                # Handle tools with no decision (default to reject for safety)
                all_tool_names = {tc["name"] for tc in pending_tools}
                decided_tools = set(st.session_state["approval_decisions"].keys())
                undecided_tools = all_tool_names - decided_tools
                rejected.extend(list(undecided_tools))

                response_content, has_more_interrupts = handle_approval_and_resume(
                    config, approved, rejected, status_container, message_container
                )

                if response_content:
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": response_content}
                    )

                if not has_more_interrupts:
                    st.session_state["awaiting_approval"] = False
                    st.session_state["approval_decisions"] = {}

                st.rerun()

    # Case 2: No pending tools but state.next == ("tools",) - auto-resume
    elif state.next == ("tools",):
        with st.chat_message("assistant"):
            status_container = st.empty()
            message_container = st.empty()

        response_content, has_more_interrupts = handle_auto_resume(
            config, status_container, message_container
        )

        if response_content:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": response_content}
            )

        if not has_more_interrupts:
            st.session_state["awaiting_approval"] = False

        st.rerun()
