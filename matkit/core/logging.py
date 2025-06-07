"""Logging configuration for MatKit."""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

# Remove default handler
logger.remove()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format: Optional[str] = None,
    colorize: bool = True,
    serialize: bool = False
) -> None:
    """
    Configure logging for MatKit.
    
    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str or Path, optional
        Path to log file. If None, only console logging is enabled
    format : str, optional
        Custom log format. If None, uses default format
    colorize : bool, default=True
        Whether to colorize console output
    serialize : bool, default=False
        Whether to serialize logs to JSON format
    """
    if format is None:
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        colorize=colorize,
        serialize=serialize
    )
    
    # File handler
    if log_file is not None:
        logger.add(
            log_file,
            format=format,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
            serialize=serialize
        )


def get_logger(name: str) -> logger:
    """
    Get a logger instance for a specific module.
    
    Parameters
    ----------
    name : str
        Name of the module
        
    Returns
    -------
    logger
        Configured logger instance
    """
    return logger.bind(name=name)


# Setup default logging
setup_logging()