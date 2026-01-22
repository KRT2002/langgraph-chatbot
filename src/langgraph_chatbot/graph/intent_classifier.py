"""
Intent classification node for determining relevant tools.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from langgraph_chatbot.config import settings
from langgraph_chatbot.graph.state import ChatState
from langgraph_chatbot.tools import ALL_TOOLS
from langgraph_chatbot.utils.logger import setup_logger
from langgraph_chatbot.utils.tool_descriptions import get_cached_tool_descriptions

logger = setup_logger(__name__)


def extract_last_n_turns(messages: list[Any], n: int = 5) -> list[Any]:
    """
    Extract last N conversation turns (HumanMessage + corresponding final AIMessage pairs).

    Parameters
    ----------
    messages : list[Any]
        Full conversation history
    n : int, optional
        Number of turns to extract, by default 5

    Returns
    -------
    list[Any]
        Last N turns as [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]

    Notes
    -----
    A "turn" consists of:
    - HumanMessage (user input)
    - Final AIMessage for that turn (skipping intermediate tool/feedback messages)
    """
    turns = []
    i = len(messages) - 1

    while i >= 0 and len(turns) // 2 < n:
        # Look for AIMessage
        if isinstance(messages[i], AIMessage):
            ai_msg = messages[i]

            # Find the corresponding HumanMessage before it
            j = i - 1
            while j >= 0:
                if isinstance(messages[j], HumanMessage):
                    human_msg = messages[j]
                    # Add this turn (human first, then ai)
                    turns.insert(0, human_msg)
                    turns.insert(1, ai_msg)
                    i = j - 1
                    break
                j -= 1
            else:
                # No HumanMessage found, skip this AIMessage
                i -= 1
        else:
            i -= 1

    logger.info(f"Extracted {len(turns) // 2} turns from conversation history")
    return turns


def intent_classifier_node(state: ChatState) -> dict:
    """
    Classify user intent and determine which tools are relevant.

    Uses retry mechanism for JSON parsing errors with feedback injection.

    Parameters
    ----------
    state : ChatState
        Current conversation state

    Returns
    -------
    dict
        Updated state with allowed_tools list
    """
    try:
        messages = state["messages"]

        # Get last N turns for context
        n_turns = settings.intent_classifier_turns
        context_messages = extract_last_n_turns(messages[:-1], n=n_turns)  # Exclude current message

        # Get current user message
        current_message = messages[-1]
        if not isinstance(current_message, HumanMessage):
            logger.warning("Last message is not HumanMessage, allowing all tools")
            return {"allowed_tools": [tool.name for tool in ALL_TOOLS]}

        # Get tool descriptions
        tool_descriptions = get_cached_tool_descriptions(ALL_TOOLS)

        # Build base intent classifier prompt
        base_classifier_messages = [
            SystemMessage(content=settings.intent_classifier_system_prompt),
            SystemMessage(content=f"Available tools:\n{tool_descriptions}"),
        ]

        # Add conversation context
        if context_messages:
            base_classifier_messages.append(SystemMessage(content="Recent conversation context:"))
            base_classifier_messages.extend(context_messages)

        # Add current query
        base_classifier_messages.append(
            SystemMessage(content=f"Current user query:\n{current_message.content}")
        )
        base_classifier_messages.append(
            SystemMessage(
                content="You must not generate any explanation just respond with JSON array of relevant tool names:"
            )
        )

        # Create LLM for classification
        classifier_llm = ChatGroq(
            model_name=settings.model_name,
            temperature=0.0,  # Deterministic for classification
            api_key=settings.groq_api_key,
        )

        # Retry loop for JSON parsing errors
        max_retries = settings.max_schema_retries
        current_messages = list(base_classifier_messages)

        for attempt in range(max_retries):
            try:
                logger.info(f"Intent classifier attempt {attempt + 1}/{max_retries}")

                # Passing as system prompt
                formatted = []
                for msg in current_messages:
                    role = msg.__class__.__name__.replace("Message", "").lower()
                    formatted.append(f"{role}:\n{msg.content}")

                combined_content = "\n\n".join(formatted)
                prompt = SystemMessage(content=combined_content)

                # Get classification
                response = classifier_llm.invoke([prompt])

                # Extract JSON from response
                content = response.content.strip()

                # Handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                allowed_tool_names = json.loads(content)

                if not isinstance(allowed_tool_names, list):
                    raise ValueError("Response is not a JSON array")

                # Validate tool names
                valid_tool_names = {tool.name for tool in ALL_TOOLS}
                allowed_tool_names = [
                    name for name in allowed_tool_names if name in valid_tool_names
                ]

                logger.info(
                    f"Intent classifier selected {len(allowed_tool_names)} tools: {allowed_tool_names}"
                )

                return {"allowed_tools": allowed_tool_names}

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing error on attempt {attempt + 1}: {str(e)}")

                # Create feedback message
                error_message = (
                    "You provided an invalid JSON response.\n"
                    f"Error: {str(e)}\n\n"
                    "Please respond with ONLY a valid JSON array of tool names.\n"
                    'Example: ["calculator", "get_weather"]\n'
                    "If no tools are needed, return: []"
                )

                # Inject feedback as AIMessage
                current_messages = current_messages + [AIMessage(content=error_message)]
                logger.info("Injected JSON error feedback, retrying...")

                # Continue to next retry
                continue

        # After max retries, fallback to allowing all tools
        logger.warning(
            f"Intent classifier max retries ({max_retries}) exceeded, allowing all tools"
        )
        return {"allowed_tools": [tool.name for tool in ALL_TOOLS]}

    except Exception as e:
        logger.error(f"Error in intent classifier: {str(e)}", exc_info=True)
        # Fallback: allow all tools
        return {"allowed_tools": [tool.name for tool in ALL_TOOLS]}
