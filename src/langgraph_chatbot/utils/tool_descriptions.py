"""
Tool description extraction utilities.

Extracts concise tool descriptions for intent classification.
"""

from typing import Any

from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_tool_description(tool: Any) -> str:
    """
    Extract the first line of a tool's docstring for intent classification.

    Parameters
    ----------
    tool : Any
        LangChain tool object with docstring

    Returns
    -------
    str
        First line of the tool's docstring, or tool name if no docstring

    Examples
    --------
    >>> extract_tool_description(calculator)
    'Perform a basic arithmetic operation on two numbers.'
    """
    try:
        # Get the tool's description or docstring
        if hasattr(tool, "description") and tool.description:
            docstring = tool.description
        elif hasattr(tool, "__doc__") and tool.__doc__:
            docstring = tool.__doc__
        elif hasattr(tool, "func") and hasattr(tool.func, "__doc__"):
            docstring = tool.func.__doc__
        else:
            # Fallback to tool name
            return getattr(tool, "name", str(tool))

        # Clean up the docstring
        docstring = docstring.strip()

        # Extract first line (before any section headers or blank lines)
        lines = docstring.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Parameters") and not line.startswith("---"):
                return line

        # Fallback if no valid line found
        return getattr(tool, "name", "Unknown tool")

    except Exception as e:
        logger.warning(f"Failed to extract description for tool: {e}")
        return getattr(tool, "name", "Unknown tool")


def get_tools_for_intent_classifier(tools: list[Any]) -> str:
    """
    Create a formatted string of tool names and descriptions for intent classifier.

    Parameters
    ----------
    tools : list[Any]
        List of LangChain tools

    Returns
    -------
    str
        Formatted string with tool names and one-line descriptions

    Examples
    --------
    >>> tools_str = get_tools_for_intent_classifier(ALL_TOOLS)
    >>> print(tools_str)
    - calculator: Perform basic arithmetic operations
    - get_weather: Get current weather information
    """
    tool_descriptions = []

    for tool in tools:
        name = getattr(tool, "name", "unknown")
        description = extract_tool_description(tool)
        tool_descriptions.append(f"- {name}: {description}")

    formatted = "\n".join(tool_descriptions)
    logger.info(f"Extracted descriptions for {len(tools)} tools")

    return formatted


# Cache for tool descriptions (regenerate only when tools change)
_CACHED_TOOL_DESCRIPTIONS = None
_CACHED_TOOL_COUNT = 0


def get_cached_tool_descriptions(tools: list[Any]) -> str:
    """
    Get cached tool descriptions or generate if cache is invalid.

    Parameters
    ----------
    tools : list[Any]
        List of LangChain tools

    Returns
    -------
    str
        Formatted tool descriptions string
    """
    global _CACHED_TOOL_DESCRIPTIONS, _CACHED_TOOL_COUNT

    # Regenerate cache if tools changed
    if _CACHED_TOOL_DESCRIPTIONS is None or len(tools) != _CACHED_TOOL_COUNT:
        logger.info("Generating tool descriptions cache")
        _CACHED_TOOL_DESCRIPTIONS = get_tools_for_intent_classifier(tools)
        _CACHED_TOOL_COUNT = len(tools)

    return _CACHED_TOOL_DESCRIPTIONS
