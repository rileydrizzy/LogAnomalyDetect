"""doc

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
