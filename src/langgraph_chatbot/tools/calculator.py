"""
Calculator tool for basic arithmetic operations.
"""

from langchain_core.tools import tool

from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.

    Parameters
    ----------
    first_num : float
        First operand
    second_num : float
        Second operand
    operation : str
        Operation to perform: 'add', 'sub', 'mul', 'div'

    Returns
    -------
    dict
        Result dictionary with operation details or error message

    Examples
    --------
    >>> calculator(10, 5, "add")
    {'status': 'success', 'first_num': 10, 'second_num': 5, 'operation': 'add', 'result': 15}
    """
    try:
        logger.info(f"Calculator called: {first_num} {operation} {second_num}")

        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                logger.warning("Division by zero attempted")
                return {
                    "status": "error",
                    "error_type": "division_by_zero",
                    "message": "Division by zero is not allowed",
                }
            result = first_num / second_num
        else:
            logger.warning(f"Unsupported operation: {operation}")
            return {
                "status": "error",
                "error_type": "invalid_operation",
                "message": f"Unsupported operation '{operation}'. Use: add, sub, mul, div",
            }

        logger.info(f"Calculator result: {result}")
        return {
            "status": "success",
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        logger.error(f"Calculator error: {str(e)}", exc_info=True)
        return {"status": "error", "error_type": "unexpected_error", "message": str(e)}
