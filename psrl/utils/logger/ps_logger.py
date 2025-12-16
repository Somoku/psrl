"""
Logger configuration for PS (Parameter Server) related modules.

This module provides a centralized logger configuration for all PS-related components
to ensure consistent logging behavior across ps_manager, request_status_tracker,
staleness_controller, and ps_storage_worker.
"""

import logging
import os

from .ray_logger import DualOutputHandler

# Unified logger for all PS related modules
PS_LOGGER_NAME = "psrl.workers.ps"
_ps_logger: logging.Logger | None = None


def get_ps_logger() -> logging.Logger:
    """Get the unified PS logger instance."""
    global _ps_logger
    if _ps_logger is None:
        _ps_logger = logging.getLogger(PS_LOGGER_NAME)
        _ps_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))
    return _ps_logger


def setup_ps_logger(logging_path: str, log_prefix: str = "PSManager") -> logging.Logger:
    """
    Setup the PS logger with DualOutputHandler.

    This should be called once during PSManager initialization to configure
    logging for all PS-related modules.

    Args:
        logging_path (str): Path to the log directory
        log_prefix (str): Prefix for the log file name

    Returns:
        logging.Logger: The configured logger instance
    """
    logger = get_ps_logger()

    # Check if handler is already added to avoid duplicate handlers
    if not any(isinstance(handler, DualOutputHandler) for handler in logger.handlers):
        handler = DualOutputHandler(logging_path, log_prefix)
        logger.addHandler(handler)

    return logger
