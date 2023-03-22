from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    getLogger,
)
from typing import Optional

# To avoid a creation of duplicates save an instance
_loggers = {}

def get_logger(
    name="dagflow",
    *,
    filename: Optional[str] = None,
    debug: bool = False,
    console: bool = False,
    formatstr: Optional[str] = "%(asctime)s - %(levelname)s - %(message)s",
) -> Logger:
    if logger := _loggers.get(name):
        return logger
    logger = getLogger(name)

    level = DEBUG if debug else INFO
    logger.setLevel(level)
    formatter = Formatter(formatstr)
    if filename:
        fh = FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if console:
        ch = StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    _loggers[name] = logger
    return logger

logger = get_logger()
SUBINFO    = INFO-10
SUBSUBINFO = INFO-20
