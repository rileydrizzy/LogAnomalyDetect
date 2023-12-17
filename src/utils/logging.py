"""
Module: Logging Utility

This module provides a simple utility for logging messages to a file and the console.
logging_utility

Usage:
- Import the module: `import logger`
- Log messages with different levels:
  - `logger.debug("Debug message")`
  - `logger.info("Informational message")`
  - `logger.warning("Warning message")`
  - `logger.error("Error message")`
  - `logger.critical("Critical message")`
"""

from pathlib import Path

from loguru import logger

FORMAT_STYLE = (
    "{time:MMMM D, YYYY > HH:mm:ss}  | {level} | {module}: {function}{line} - {message}"
)

LOG_DIR = Path("logs")
log_filepath = Path(LOG_DIR, "running_logs.log")
Path.mkdir(LOG_DIR, exist_ok=True)

logger.add(
    log_filepath,
    format=FORMAT_STYLE,
    level="INFO",
)
