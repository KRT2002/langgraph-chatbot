"""
Logging configuration for the application.

Provides structured logging with file rotation and console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from langgraph_chatbot.config import settings


def setup_logger(
    name: str, log_file: Optional[str] = None, level: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the module)
    log_file : Optional[str], optional
        Custom log file name, by default None (uses timestamp)
    level : Optional[str], optional
        Logging level, by default uses settings.log_level

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    log_level = getattr(logging, level or settings.log_level)
    logger.setLevel(log_level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = logging.Formatter(fmt="%(levelname)s | %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"chatbot_{timestamp}.log"

    log_path = Path(settings.logs_dir) / log_file
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Global logger instance
logger = setup_logger("langgraph_chatbot")
