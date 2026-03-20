#!/usr/bin/env python3
"""
Structured Logging — Proper Python logging with rotating file output.

Log levels:
  - DEBUG: tick-level data, price updates, iteration details
  - INFO: trades, signals, startup/shutdown, config
  - WARNING: reconnections, stale feeds, rate limits
  - ERROR: API failures, unexpected exceptions
  - CRITICAL: kill switch triggers, fatal errors
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src.config import Config, PROJECT_ROOT


# Custom format with timestamps, module, and level
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: Config = None) -> logging.Logger:
    """
    Configure logging for the entire bot.

    Sets up:
    - Console handler (INFO+ by default)
    - Rotating file handler (DEBUG+, 10MB, 5 backups)

    Args:
        config: Bot configuration. Uses defaults if None.

    Returns:
        The root 'bot' logger.
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    # Create the root bot logger
    root_logger = logging.getLogger("bot")
    root_logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # Clear any existing handlers (prevent duplicates on re-init)
    root_logger.handlers.clear()

    # --- Console Handler ---
    console_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # --- File Handler ---
    log_file = config.resolve_path(config.logging.log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_level = getattr(logging, config.logging.file_level.upper(), logging.DEBUG)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named child logger under the 'bot' namespace.

    Usage:
        logger = get_logger("trader")
        logger.info("Order placed")
        # Logs as: bot.trader | Order placed

    Args:
        name: Module/component name.

    Returns:
        A child logger.
    """
    return logging.getLogger(f"bot.{name}")
