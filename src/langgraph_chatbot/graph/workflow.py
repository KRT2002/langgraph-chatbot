"""
LangGraph workflow construction and compilation.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from langgraph_chatbot.config import settings
from langgraph_chatbot.graph.intent_classifier import intent_classifier_node
from langgraph_chatbot.graph.nodes import chat_node, tool_node
from langgraph_chatbot.graph.state import ChatState
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_chatbot():
    """
    Create and compile the LangGraph chatbot workflow.

    Workflow:
    1. START → intent_classifier (determines relevant tools)
    2. intent_classifier → chat_node (LLM with filtered tools)
    3. chat_node → tools_condition (check if tools needed)
    4. If tools needed → tool_node → chat_node (loop)
    5. If no tools → END

    Returns
    -------
    CompiledGraph
        Compiled LangGraph workflow with checkpointing
    """
    try:
        logger.info("Creating chatbot workflow with intent classifier")

        # Initialize state graph
        graph = StateGraph(ChatState)

        # Add nodes
        graph.add_node("intent_classifier", intent_classifier_node)
        graph.add_node("chat_node", chat_node)
        graph.add_node("tools", tool_node)

        # Add edges
        graph.add_edge(START, "intent_classifier")
        graph.add_edge("intent_classifier", "chat_node")
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")

        # Setup checkpointer
        db_path = Path(settings.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(database=str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        # Compile graph
        chatbot = graph.compile(checkpointer=checkpointer)
        logger.info("Chatbot workflow compiled successfully with intent classifier")

        return chatbot

    except Exception as e:
        logger.error(f"Failed to create chatbot: {str(e)}", exc_info=True)
        raise


def retrieve_all_threads(chatbot) -> list[str]:
    """
    Retrieve all conversation thread IDs from the checkpointer.

    Parameters
    ----------
    chatbot : CompiledGraph
        Compiled chatbot workflow

    Returns
    -------
    list[str]
        List of thread IDs
    """
    try:
        all_threads = set()
        checkpointer = chatbot.checkpointer

        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(thread_id)

        logger.info(f"Retrieved {len(all_threads)} conversation threads")
        return list(all_threads)

    except Exception as e:
        logger.error(f"Error retrieving threads: {str(e)}", exc_info=True)
        return []


# Global chatbot instance
chatbot = create_chatbot()
