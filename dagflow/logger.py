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


def get_logger(
    filename: Optional[str] = None, debug: bool = False, **kwargs
) -> Logger:
    formatstr = (
        fmtstr
        if (fmtstr := kwargs.pop("formatstr", False))
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = getLogger(nm if (nm := kwargs.pop("name", False)) else "PyGNA")
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
    return logger
