"""
Date and time utility tools.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


@tool
def get_current_time(timezone: str = "UTC") -> dict:
    """
    Get the current time in a specific timezone.

    Parameters
    ----------
    timezone : str, optional
        Timezone name (e.g., 'UTC', 'America/New_York', 'Asia/Tokyo'), by default 'UTC'
        Timezone must be valid IANA TZ string

    Returns
    -------
    dict
        Current date and time information

    Examples
    --------
    >>> get_current_time("America/New_York")
    {'status': 'success', 'timezone': 'America/New_York', 'datetime': '2024-01-15 10:30:45', ...}
    """
    try:
        logger.info(f"Getting current time for timezone: {timezone}")

        tz = ZoneInfo(timezone)
        now = datetime.now(tz)

        result = {
            "status": "success",
            "timezone": timezone,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "iso_format": now.isoformat(),
        }

        logger.info(f"Current time retrieved: {result['datetime']}")
        return result

    except Exception as e:
        logger.error(f"Error getting current time: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": "invalid_timezone",
            "message": f"Invalid timezone '{timezone}' or error: {str(e)}",
        }


@tool
def date_calculator(start_date: str, operation: str, days: int = 0) -> dict:
    """
    Calculate dates by adding or subtracting days.

    Parameters
    ----------
    start_date : str
        Starting date in YYYY-MM-DD format
    operation : str
        Operation to perform: 'add' or 'subtract'
    days : int, optional
        Number of days to add/subtract, by default 0

    Returns
    -------
    dict
        Calculated date information

    Examples
    --------
    >>> date_calculator("2024-01-15", "add", 30)
    {'status': 'success', 'start_date': '2024-01-15', 'operation': 'add', 'days': 30, 'result_date': '2024-02-14'}
    """
    try:
        logger.info(f"Date calculation: {start_date} {operation} {days} days")

        start = datetime.strptime(start_date, "%Y-%m-%d")

        if operation.lower() == "add":
            result = start + timedelta(days=days)
        elif operation.lower() == "subtract":
            result = start - timedelta(days=days)
        else:
            logger.warning(f"Invalid operation: {operation}")
            return {
                "status": "error",
                "error_type": "invalid_operation",
                "message": f"Invalid operation '{operation}'. Use 'add' or 'subtract'",
            }

        return {
            "status": "success",
            "start_date": start_date,
            "operation": operation,
            "days": days,
            "result_date": result.strftime("%Y-%m-%d"),
            "day_of_week": result.strftime("%A"),
        }

    except ValueError as e:
        logger.error(f"Invalid date format: {str(e)}")
        return {
            "status": "error",
            "error_type": "invalid_date_format",
            "message": "Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-15)",
        }
    except Exception as e:
        logger.error(f"Date calculator error: {str(e)}", exc_info=True)
        return {"status": "error", "error_type": "unexpected_error", "message": str(e)}
