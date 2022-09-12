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
    filename: Optional[str] = None, debug: bool = False, **kwargs
) -> Logger:
    name = kwargs.pop("name", "PyGNA")
    if logger := _loggers.get(name):
        return logger
    logger = getLogger("PyGNA")
    formatstr = (
        fmtstr
        if (fmtstr := kwargs.pop("formatstr", False))
        else "%(asctime)s - %(levelname)s - %(message)s"
    )
    level = DEBUG if debug else INFO
    logger.setLevel(level)
    formatter = Formatter(formatstr)
    if filename:
        fh = FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if kwargs.pop("console", True):
        ch = StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    _loggers[name] = logger
    return logger
