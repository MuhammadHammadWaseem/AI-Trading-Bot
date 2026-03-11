"""
config/logger.py
----------------
Centralized logging - colored terminal + rotating file logs.
Emoji-free for Windows terminal compatibility.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import colorlog

from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Terminal handler (colored, no emojis)
    terminal_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(cyan)s%(name)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG":    "white",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    )
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(terminal_formatter)
    terminal_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    logger.addHandler(terminal_handler)

    # File handler (rotating)
    if settings.log_to_file:
        logs_dir = settings.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "trading_bot.log"
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger