"""doc

"""

from pathlib import Path
from loguru import logger

Format_style = "{time:YYYY_MM_DD HH:mm:ss} | {level} | {module}: {function}{line} - {message}"

log_dir = 'logs'
log_filepath = Path(log_dir, "running_logs.log")
Path.mkdir(log_dir, exist_ok= True)

#os.makedirs(log_dir, exist_ok= True)

logger.add(log_filepath,
    format= Format_style,
    level= "INFO",)
