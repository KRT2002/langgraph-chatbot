"""
Approval check node for human-in-the-loop tool execution.
"""

from typing import Literal

from langchain_core.messages import AIMessage

from langgraph_chatbot.config import settings
from langgraph_chatbot.graph.state import ChatState
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_pending_tools(state: ChatState) -> list[dict]:
    """
    Extract tool calls from the last AIMessage.

    Parameters
    ----------
    state : ChatState
        Current conversation state

    Returns
    -------
    list[dict]
        List of tool calls with name, args, and id
    """
    messages = state["messages"]

    # Find last AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "tool_call"),
                    }
                )
            logger.info(f"Extracted {len(tool_calls)} pending tool calls")
            return tool_calls

    logger.info("No pending tool calls found")
    return []


def check_approval_required(
    tool_calls: list[dict], human_in_loop_enabled: bool
) -> tuple[list[dict], list[dict]]:
    """
    Determine which tools need approval and which can execute directly.

    Parameters
    ----------
    tool_calls : list[dict]
        List of pending tool calls
    human_in_loop_enabled : bool
        Whether human-in-the-loop is enabled

    Returns
    -------
    tuple[list[dict], list[dict]]
        (tools_needing_approval, tools_auto_approved)
    """
    if not human_in_loop_enabled:
        logger.info("Human-in-loop disabled, all tools auto-approved")
        return [], tool_calls

    tools_needing_approval = []
    tools_auto_approved = []

    for tc in tool_calls:
        if tc["name"] in settings.tools_requiring_approval:
            tools_needing_approval.append(tc)
        else:
            tools_auto_approved.append(tc)

    logger.info(
        f"Approval check: {len(tools_needing_approval)} need approval, "
        f"{len(tools_auto_approved)} auto-approved"
    )

    return tools_needing_approval, tools_auto_approved


def approval_check_node(state: ChatState) -> dict:
    """
    Check if any tools need human approval before execution.

    This node decides whether to interrupt for approval or proceed directly to tools.

    Parameters
    ----------
    state : ChatState
        Current conversation state

    Returns
    -------
    dict
        Updated state with approval information
    """
    try:
        # Get human-in-loop setting from state or default to False
        human_in_loop_enabled = state.get("human_in_loop_enabled", False)

        # Extract pending tool calls
        pending_tools = extract_pending_tools(state)

        if not pending_tools:
            logger.info("No tools to approve, proceeding")
            return {
                "needs_approval": False,
                "pending_tool_calls": [],
            }

        # Check which tools need approval
        tools_needing_approval, tools_auto_approved = check_approval_required(
            pending_tools, human_in_loop_enabled
        )

        if tools_needing_approval:
            logger.info(f"Interrupt required for {len(tools_needing_approval)} tools")
            return {
                "needs_approval": True,
                "pending_tool_calls": tools_needing_approval,
            }
        else:
            logger.info("All tools auto-approved, proceeding to execution")
            return {
                "needs_approval": False,
                "pending_tool_calls": [],
            }

    except Exception as e:
        logger.error(f"Error in approval check: {str(e)}", exc_info=True)
        # On error, don't block - allow tools to execute
        return {
            "needs_approval": False,
            "pending_tool_calls": [],
        }


def should_interrupt(state: ChatState) -> Literal["interrupt", "tools"]:
    """
    Routing function to determine if graph should interrupt for approval.

    Parameters
    ----------
    state : ChatState
        Current conversation state

    Returns
    -------
    Literal["interrupt", "tools"]
        Route to take: interrupt for approval or proceed to tools
    """
    needs_approval = state.get("needs_approval", False)

    if needs_approval:
        logger.info("Routing to interrupt for human approval")
        return "interrupt"
    else:
        logger.info("Routing directly to tools execution")
        return "tools"
