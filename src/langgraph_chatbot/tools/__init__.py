"""
Tool collection for the LangGraph chatbot.
"""

from langgraph_chatbot.tools.calculator import calculator
from langgraph_chatbot.tools.datetime_utils import date_calculator, get_current_time
from langgraph_chatbot.tools.file_operations import file_operations
from langgraph_chatbot.tools.unit_converter import unit_converter
from langgraph_chatbot.tools.weather import get_weather
from langgraph_chatbot.tools.web_search import web_search

# All available tools
ALL_TOOLS = [
    calculator,
    get_weather,
    unit_converter,
    get_current_time,
    date_calculator,
    file_operations,
    web_search,
]

__all__ = [
    "ALL_TOOLS",
    "calculator",
    "get_weather",
    "unit_converter",
    "get_current_time",
    "date_calculator",
    "file_operations",
    "web_search",
]