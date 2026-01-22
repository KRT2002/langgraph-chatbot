"""
Graph nodes for the LangGraph chatbot workflow.
"""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, SystemMessage
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
            api_key=settings.groq_api_key
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
                
                return {"messages": [response]}
                
            except Exception as e:
                # Check for rate limit error - don't retry, return to LLM
                if "groq.RateLimitError" in str(type(e)) or "rate_limit_exceeded" in str(e).lower():
                    logger.error(f"Rate limit error: {str(e)}")
                    error_msg = AIMessage(
                        content=f"Rate limit error occurred: {str(e)}"
                    )
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

# Tool execution node
tool_node = ToolNode(ALL_TOOLS)