"""
Graph nodes for the LangGraph chatbot workflow.
"""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from langgraph_chatbot.config import settings
from langgraph_chatbot.graph.state import ChatState
from langgraph_chatbot.tools import ALL_TOOLS
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_llm(tools: list = None, strict: bool = True) -> ChatGroq:
    """
    Create and configure the LLM instance.

    Parameters
    ----------
    tools : list, optional
        List of tools to bind to LLM, by default None (no tools)
    strict : bool, optional
        Whether to enforce strict schema validation, by default True

    Returns
    -------
    ChatGroq
        Configured Groq LLM instance
    """
    try:
        llm = ChatGroq(
            model_name=settings.model_name,
            temperature=settings.temperature,
            api_key=settings.groq_api_key,
        )

        if tools:
            llm = llm.bind_tools(tools, strict=strict)
            logger.info(f"LLM initialized with {len(tools)} tools (strict={strict})")
        else:
            logger.info("LLM initialized without tools")

        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
        raise


def chat_node(state: ChatState) -> dict:
    """
    LLM node that processes messages with schema error retry and fallback.

    This node implements:
    - Intent-based tool filtering
    - Schema error retry with feedback (max 3 attempts)
    - Final fallback to LLM without tools
    - Rate limit error handling

    Parameters
    ----------
    state : ChatState
        Current conversation state

    Returns
    -------
    dict
        Updated state with AI response
    """
    try:
        messages = state["messages"]
        allowed_tools = state.get("allowed_tools", [])

        logger.info(f"Chat node processing {len(messages)} messages")

        # Add system prompt to messages if not present
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system:
            messages = [SystemMessage(content=settings.main_llm_system_prompt)] + messages

        # Filter tools based on intent classifier
        if allowed_tools:
            filtered_tools = [tool for tool in ALL_TOOLS if tool.name in allowed_tools]
            logger.info(f"Using {len(filtered_tools)} tools based on intent: {allowed_tools}")
        else:
            filtered_tools = ALL_TOOLS
            logger.info("Using all tools (no intent filtering)")

        # Retry loop for schema errors
        max_retries = settings.max_schema_retries
        current_messages = list(messages)  # Copy to avoid mutation

        for attempt in range(max_retries):
            try:
                logger.info(f"LLM invocation attempt {attempt + 1}/{max_retries}")

                # Create LLM with filtered tools and strict validation
                llm = create_llm(tools=filtered_tools, strict=True)
                response = llm.invoke(current_messages)

                # Check for tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in response.tool_calls]
                    logger.info(f"AI requested tools: {tool_names}")
                else:
                    logger.info("AI generated text response")

                # Clear rejected_tools from state after processing
                return {"messages": [response]}

            except Exception as e:
                # Check for rate limit error - don't retry, return to user
                if "groq.RateLimitError" in str(type(e)) or "rate_limit_exceeded" in str(e).lower():
                    logger.error(f"Rate limit error: {str(e)}")
                    error_msg = AIMessage(content=f"Rate limit error occurred: {str(e)}")
                    return {"messages": [error_msg]}

                # Check for schema/function call error - retry with feedback
                if "Failed to call a function" in str(e) or "groq.APIError" in str(type(e)):
                    logger.warning(f"Schema validation error on attempt {attempt + 1}")

                    # Create feedback message without error details
                    error_message = (
                        "You attempted to call a tool but provided an invalid input schema. "
                        "Please retry with a correct tool call. Make sure all required parameters "
                        "are provided and match the expected types."
                    )

                    # Inject feedback as AIMessage
                    current_messages = current_messages + [AIMessage(content=error_message)]
                    logger.info("Injected schema error feedback, retrying...")
                    continue

                # Check for other validation errors
                if isinstance(e, (ValidationError, OutputParserException)):
                    logger.warning(f"Schema validation error on attempt {attempt + 1}: {str(e)}")

                    # Create feedback message for LLM
                    error_message = (
                        "You attempted to call a tool but provided an invalid input schema.\n"
                        f"Schema error:\n{str(e)}\n\n"
                        "Please retry with a correct tool call. Make sure all required parameters "
                        "are provided and match the expected types."
                    )

                    # Inject feedback as AIMessage
                    current_messages = current_messages + [AIMessage(content=error_message)]
                    logger.info("Injected schema error feedback, retrying...")
                    continue

                # For other errors, re-raise to be caught by outer try-catch
                raise

        # After max retries, fallback to LLM without tools
        logger.warning(f"Max retries ({max_retries}) exceeded, falling back to LLM without tools")

        try:
            llm_no_tools = create_llm(tools=None, strict=False)
            response = llm_no_tools.invoke(current_messages)
            logger.info("Fallback LLM (no tools) generated response")
            return {"messages": [response]}

        except Exception as fallback_error:
            logger.error(f"Fallback LLM also failed: {str(fallback_error)}", exc_info=True)
            # Last resort: return error message
            error_msg = AIMessage(
                content="I apologize, but I'm having trouble processing your request right now. "
                "Please try rephrasing your question."
            )
            return {"messages": [error_msg]}

    except Exception as e:
        logger.error(f"Error in chat_node: {str(e)}", exc_info=True)
        error_msg = AIMessage(content=f"An error occurred: {str(e)}")
        return {"messages": [error_msg]}


class FilteredToolNode:
    """
    Custom wrapper around ToolNode that filters out tool calls
    that already have corresponding ToolMessages in the state.

    This prevents duplicate tool execution when ToolMessages are
    manually injected into the state.
    """

    def __init__(self, tools):
        """
        Initialize the FilteredToolNode.

        Parameters
        ----------
        tools : list
            List of tools to be used by the underlying ToolNode
        """
        self.tool_node = ToolNode(tools)

    def __call__(self, state: dict) -> dict:
        """
        Execute tool calls, filtering out those already executed.

        Parameters
        ----------
        state : dict
            The current state containing messages

        Returns
        -------
        dict
            Updated state with new ToolMessages
        """
        logger.info("FilteredToolNode called - starting tool call filtering")
        messages = state.get("messages", [])
        logger.debug(f"Total messages in state: {len(messages)}")

        # Get all existing ToolMessage IDs from state
        existing_tool_ids = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}
        logger.info(
            f"Found {len(existing_tool_ids)} existing ToolMessages with IDs: {existing_tool_ids}"
        )

        # Find the last AIMessage with tool_calls
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_ai_message = msg
                break

        # If no AIMessage with tool_calls, just return empty
        if not last_ai_message:
            logger.warning("No AIMessage with tool_calls found in state - returning empty")
            return {"messages": []}

        logger.info(f"Found AIMessage with {len(last_ai_message.tool_calls)} tool_calls")
        tool_call_ids = [tc["id"] for tc in last_ai_message.tool_calls]
        logger.debug(f"Tool call IDs in AIMessage: {tool_call_ids}")

        # Filter out tool_calls that already have ToolMessages
        filtered_tool_calls = [
            tc for tc in last_ai_message.tool_calls if tc["id"] not in existing_tool_ids
        ]

        filtered_ids = [tc["id"] for tc in filtered_tool_calls]
        skipped_ids = [
            tc["id"] for tc in last_ai_message.tool_calls if tc["id"] in existing_tool_ids
        ]

        logger.info(
            f"Filtered tool_calls: {len(filtered_tool_calls)} to execute, {len(skipped_ids)} already executed"
        )
        if skipped_ids:
            logger.info(f"Skipping tool_calls with IDs (already executed): {skipped_ids}")
        if filtered_ids:
            logger.info(f"Will execute tool_calls with IDs: {filtered_ids}")

        # If all tool_calls already executed, return empty
        if not filtered_tool_calls:
            logger.info("All tool_calls already executed - returning empty")
            return {"messages": []}

        # Create a new AIMessage with only the filtered tool_calls
        filtered_ai_message = AIMessage(
            content=last_ai_message.content,
            tool_calls=filtered_tool_calls,
            id=last_ai_message.id,
            additional_kwargs=last_ai_message.additional_kwargs,
            response_metadata=last_ai_message.response_metadata,
        )

        # Find the index of the last_ai_message in the messages list
        ai_message_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i] is last_ai_message:
                ai_message_index = i
                break

        logger.debug(f"Found AIMessage at index {ai_message_index} in messages list")

        # Replace the original AIMessage with the filtered version
        # This preserves any messages that came after it (like manually added ToolMessages)
        updated_messages = (
            messages[:ai_message_index] + [filtered_ai_message] + messages[ai_message_index + 1 :]
        )

        logger.debug(f"Created updated messages: {len(updated_messages)} total messages")
        logger.debug(
            f"Messages after AIMessage that will be preserved: {len(messages[ai_message_index + 1:])}"
        )

        # Create temporary state with filtered message
        # IMPORTANT: This temp_state is only used for ToolNode execution
        # It is NOT saved to the graph state - only the result (new ToolMessages) is returned
        temp_state = {**state, "messages": updated_messages}

        # Execute the tool node with filtered tool_calls
        # ToolNode will only see and execute the filtered tool_calls
        logger.info(f"Invoking ToolNode with {len(filtered_tool_calls)} filtered tool_calls")
        result = self.tool_node.invoke(temp_state)

        # Result contains only the NEW ToolMessages from executed tool_calls
        # These get appended to the original state by LangGraph
        new_tool_messages = result.get("messages", [])
        logger.info(
            f"ToolNode execution complete - generated {len(new_tool_messages)} new ToolMessages"
        )

        return result


# Tool execution node
tool_node = FilteredToolNode(ALL_TOOLS)
