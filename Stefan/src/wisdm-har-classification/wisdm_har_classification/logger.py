import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

LOG_FILE = "logs/debug.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",  # overwrite each run ("a" = append)
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger("WISDM")
logger.setLevel(logging.DEBUG)

logger.info("Logger initialized.")
