"""
Logging configuration module for Micro Post application.

This module provides centralized logging configuration with both
console and file output support.
"""

import logging
import os
from datetime import datetime


def setup_logger(log_dir: str = None) -> logging.Logger:
    """
    Set up and configure the application logger.

    Args:
        log_dir: Directory to save log files. If None, logs only to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("micro_post")

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG level) if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"micro_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file created: {log_file}")

    return logger


def get_logger() -> logging.Logger:
    """
    Get the application logger instance.

    Returns:
        Logger instance for micro_post.
    """
    return logging.getLogger("micro_post")
