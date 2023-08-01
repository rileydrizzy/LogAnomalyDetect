"""doc

"""

from pathlib import Path

from loguru import logger

Format_style = "{time:YYYY_MM_DD HH:mm:ss} | {level} | {module}: {function}{line} - {message}"

log_dir = Path.cwd() / "logs"
log_filepath = Path(log_dir, "running_logs.log")
log_dir.mkdir(exist_ok=True)

logger.add(
    log_filepath,
    format=Format_style,
    level="INFO",
)
