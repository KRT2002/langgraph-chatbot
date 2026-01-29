"""
Streamlit UI for the LangGraph Chatbot.
"""

import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_chatbot.graph.workflow import chatbot
from langgraph_chatbot.utils.logger import setup_logger

from langgraph_chatbot.ui.approval_handlers import handle_auto_resume
from langgraph_chatbot.ui.session_handlers import (
    generate_conversation_title,
    initialize_session_state,
    save_conversation_title,
)
from langgraph_chatbot.ui.ui_components import (
    render_approval_ui,
    render_chat_messages,
    render_sidebar,
)

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title="LangGraph Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)


# ======================= Session Initialization ===================

initialize_session_state()


# ============================ Sidebar ============================

render_sidebar()


# ============================ Main UI ============================

# Display conversation title
st.title(st.session_state["conversation_title"])

# Render message history
render_chat_messages()

# Handle approval UI if awaiting
if st.session_state.get("awaiting_approval", False):
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    render_approval_ui(config)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Reset approval decisions for new turn
    st.session_state["approval_decisions"] = {}

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
        "configurable": {
            "thread_id": st.session_state["thread_id"],
        }
    }

    # Update state with human-in-loop setting
    chatbot.update_state(
        config, {"human_in_loop_enabled": st.session_state["human_in_loop_enabled"]}
    )

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

                # Skip messages from approval_check (internal only)
                if node_name == "approval_check":
                    continue

                # Clear intent indicator when chat_node starts
                if node_name == "chat_node" and not tool_used:
                    status_container.empty()

                # Handle tool execution
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "unknown")
                    tool_used = True
                    status_container.info(f"🔧 Using tool: `{tool_name}`")

                # Stream AI response (only from chat_node)
                if (
                    isinstance(message_chunk, AIMessage)
                    and node_name == "chat_node"
                    and hasattr(message_chunk, "tool_calls")
                    and not message_chunk.tool_calls
                ):
                    response_content += message_chunk.content
                    message_container.markdown(response_content)

            # Check if graph is interrupted for approval
            state = chatbot.get_state(config)
            if state.next == ("tools",) and state.values.get("needs_approval", False):
                status_container.warning("⏸️ Waiting for tool approval...")
                st.session_state["awaiting_approval"] = True
                st.rerun()
            elif state.next == ("tools",):
                # Interrupt but no approval needed - auto-resume with streaming
                additional_content, has_more_interrupts = handle_auto_resume(
                    config, status_container, message_container
                )
                response_content += additional_content

                if has_more_interrupts:
                    st.session_state["awaiting_approval"] = True
                    st.rerun()

            # Clear status after completion
            if tool_used:
                status_container.success("✅ Tool execution complete")
                time.sleep(1)
                status_container.empty()
            else:
                status_container.empty()

            # Save assistant message
            if response_content:
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": response_content}
                )

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.session_state["message_history"].append({"role": "assistant", "content": error_msg})
