"""
LangGraph workflow construction and compilation.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from langgraph_chatbot.config import settings
from langgraph_chatbot.graph.approval_node import approval_check_node, should_interrupt
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
    4. If tools needed → approval_check (check if approval required)
    5. approval_check routes to:
       - interrupt (if approval needed - human-in-loop)
       - tools (if no approval needed - direct execution)
    6. tools → chat_node (loop back with results)
    7. If no tools → END

    Returns
    -------
    CompiledGraph
        Compiled LangGraph workflow with checkpointing and interrupts
    """
    try:
        logger.info("Creating chatbot workflow with intent classifier and approval node")

        # Initialize state graph
        graph = StateGraph(ChatState)

        # Add nodes
        graph.add_node("intent_classifier", intent_classifier_node)
        graph.add_node("chat_node", chat_node)
        graph.add_node("approval_check", approval_check_node)
        graph.add_node("tools", tool_node)

        # Add edges
        graph.add_edge(START, "intent_classifier")
        graph.add_edge("intent_classifier", "chat_node")

        # From chat_node, check if tools are needed
        graph.add_conditional_edges(
            "chat_node",
            tools_condition,
            {
                "tools": "approval_check",  # If tools needed, check approval first
                END: END,  # If no tools, end
            },
        )

        # From approval_check, route based on whether approval is needed
        graph.add_conditional_edges(
            "approval_check",
            should_interrupt,
            {
                "interrupt": "tools",  # Will interrupt before tools if needed
                "tools": "tools",  # Direct to tools if no approval needed
            },
        )

        # From tools, always go back to chat_node
        graph.add_edge("tools", "chat_node")

        # Setup checkpointer
        db_path = Path(settings.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(database=str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        # Compile graph with interrupt capability
        # Interrupt will happen before "tools" node when needs_approval=True
        chatbot = graph.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        logger.info("Chatbot workflow compiled successfully with approval mechanism")

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
