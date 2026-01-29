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


def describe_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Generate a human-readable description of what a tool call will do.

    Parameters
    ----------
    tool_name : str
        Name of the tool
    tool_args : dict
        Arguments for the tool call

    Returns
    -------
    str
        Human-readable description of the tool's task

    Examples
    --------
    >>> describe_tool_call("get_weather", {"city": "London"})
    'Fetch current weather information for London'
    """
    try:
        # Tool-specific descriptions
        if tool_name == "calculator":
            op = tool_args.get("operation", "calculate")
            first = tool_args.get("first_num", "?")
            second = tool_args.get("second_num", "?")
            return f"Calculate {first} {op} {second}"

        elif tool_name == "get_weather":
            city = tool_args.get("city", "unknown location")
            return f"Fetch current weather information for {city}"

        elif tool_name == "unit_converter":
            value = tool_args.get("value", "?")
            from_unit = tool_args.get("from_unit", "?")
            to_unit = tool_args.get("to_unit", "?")
            return f"Convert {value} {from_unit} to {to_unit}"

        elif tool_name == "get_current_time":
            tz = tool_args.get("timezone", "UTC")
            return f"Get current time in {tz} timezone"

        elif tool_name == "date_calculator":
            start = tool_args.get("start_date", "?")
            op = tool_args.get("operation", "?")
            days = tool_args.get("days", "?")
            return f"Calculate date by {op}ing {days} days from {start}"

        elif tool_name == "file_operations":
            operation = tool_args.get("operation", "?")
            filename = tool_args.get("filename", "?")
            if operation == "read":
                return f"Read contents of file '{filename}'"
            elif operation == "write":
                return f"Write content to file '{filename}'"
            elif operation == "append":
                return f"Append content to file '{filename}'"
            elif operation == "delete":
                return f"Delete file '{filename}'"
            elif operation == "list":
                return "List all available files"
            else:
                return f"Perform '{operation}' operation on file '{filename}'"

        elif tool_name == "web_search":
            query = tool_args.get("query", "?")
            max_results = tool_args.get("max_results", 5)
            return f"Search the web for '{query}' (up to {max_results} results)"

        else:
            # Generic fallback
            args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
            return f"Execute {tool_name} with arguments: {args_str}"

    except Exception as e:
        logger.warning(f"Error describing tool call: {e}")
        return f"Execute {tool_name}"


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
