import logging
import sys
from typing import Optional

DEFAULT_LOGGER_NAME = "spot_scam"


def _resolve_logger_name(name: Optional[str]) -> str:
    return name or DEFAULT_LOGGER_NAME


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    if isinstance(level, str):
        name = level
        level = logging.INFO

    logger_name = _resolve_logger_name(name)
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        # Already configured
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
