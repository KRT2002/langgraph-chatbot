"""
Human-in-loop approval handling for tool execution.
"""

import time

from langchain_core.messages import AIMessage, ToolMessage

from langgraph_chatbot.graph.workflow import chatbot
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def handle_approval_and_resume(
    config, approved_tools, rejected_tools, status_container, message_container
):
    """
    Filter tool calls based on approval/rejection and resume graph execution with streaming.

    Parameters
    ----------
    config : dict
        LangGraph configuration with thread_id
    approved_tools : list[str]
        List of approved tool names
    rejected_tools : list[str]
        List of rejected tool names
    status_container : streamlit container
        Container for status messages
    message_container : streamlit container
        Container for assistant messages

    Returns
    -------
    tuple[str, bool]
        (response_content, has_more_interrupts)
    """
    try:
        # Get current state
        state = chatbot.get_state(config)
        messages = list(state.values.get("messages", []))

        # Find last AIMessage with tool_calls
        last_ai_msg = None

        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and hasattr(messages[i], "tool_calls"):
                if messages[i].tool_calls:
                    last_ai_msg = messages[i]
                    break

        if not last_ai_msg:
            logger.warning("No AIMessage with tool_calls found")
            return "", False

        logger.info(f"Filtering tool calls: approved={approved_tools}, rejected={rejected_tools}")

        # Create ToolMessages for rejected tools
        rejection_messages = []
        for tc in last_ai_msg.tool_calls:
            if tc.get("name") in rejected_tools:
                rejection_messages.append(
                    ToolMessage(
                        content="Tool execution was rejected by the user. Please respond without using this tool.",
                        tool_call_id=tc.get("id", ""),
                        name=tc.get("name", "unknown"),
                    )
                )

        # Rebuild messages list
        new_messages = messages + rejection_messages

        # Update state with new messages and clear approval flags
        chatbot.update_state(
            config,
            {
                "messages": new_messages,
                "approved_tools": [],
                "rejected_tools": [],
                "needs_approval": False,
            },
        )

        # Resume execution with streaming
        logger.info("Resuming graph execution with streaming")
        status_container.info("⚡ Processing...")

        response_content = ""
        tool_used = False

        for message_chunk, metadata in chatbot.stream(None, config=config, stream_mode="messages"):
            node_name = metadata.get("langgraph_node", "")

            # Skip internal nodes
            if node_name in ["intent_classifier", "approval_check"]:
                continue

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

        # Clear status
        if tool_used:
            status_container.success("✅ Tool execution complete")
            time.sleep(1)

        status_container.empty()

        # Check if there's another interrupt
        state = chatbot.get_state(config)
        has_more_interrupts = state.next == ("tools",)

        logger.info(
            f"Resume completed, response length: {len(response_content)}, has_more_interrupts: {has_more_interrupts}"
        )
        return response_content, has_more_interrupts

    except Exception as e:
        logger.error(f"Error in handle_approval_and_resume: {str(e)}", exc_info=True)
        status_container.error(f"Error: {str(e)}")
        return "", False


def handle_auto_resume(config, status_container, message_container):
    """
    Auto-resume graph execution when no approval is needed.

    Parameters
    ----------
    config : dict
        LangGraph configuration with thread_id
    status_container : streamlit container
        Container for status messages
    message_container : streamlit container
        Container for assistant messages

    Returns
    -------
    tuple[str, bool]
        (response_content, has_more_interrupts)
    """
    try:
        logger.info("Auto-resuming (no approval needed)")
        status_container.info("⚡ Executing tools...")

        response_content = ""
        tool_used = False

        # Auto-resume with streaming
        for message_chunk, metadata in chatbot.stream(None, config=config, stream_mode="messages"):
            node_name = metadata.get("langgraph_node", "")

            if node_name in ["intent_classifier", "approval_check"]:
                continue

            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "unknown")
                tool_used = True
                status_container.info(f"🔧 Using tool: `{tool_name}`")

            if (
                isinstance(message_chunk, AIMessage)
                and node_name == "chat_node"
                and hasattr(message_chunk, "tool_calls")
                and not message_chunk.tool_calls
            ):
                response_content += message_chunk.content
                message_container.markdown(response_content)

        # Clear status
        if tool_used:
            status_container.success("✅ Tool execution complete")
            time.sleep(1)

        status_container.empty()

        # Check if there's another interrupt
        state = chatbot.get_state(config)
        has_more_interrupts = state.next == ("tools",)

        logger.info(
            f"Auto-resume completed, response length: {len(response_content)}, has_more_interrupts: {has_more_interrupts}"
        )
        return response_content, has_more_interrupts

    except Exception as e:
        logger.error(f"Error in handle_auto_resume: {str(e)}", exc_info=True)
        status_container.error(f"Error: {str(e)}")
        return "", False
