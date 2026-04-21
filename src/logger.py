"""
src/logger.py
==============
Central logging configuration for the Agentic Doctor System.

Every module imports get_logger() from here to get a consistent logger
that writes to BOTH the terminal (with color) and a rotating log file.

Log file location:  logs/agentic_doctor.log
Log rotation:       5 MB per file, keeps last 3 files
Log format:
    2025-01-08 10:45:23 | INFO     | vector_store_manager | Batch 3/130 | docs 100-149 | 50 docs
    2025-01-08 10:45:31 | WARNING  | vector_store_manager | Quota hit — waiting 60s (attempt 1/3)
    2025-01-08 10:46:31 | ERROR    | vector_store_manager | All retries exhausted. Checkpoint saved.

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Starting ingestion")
    logger.warning("Quota hit")
    logger.error("Failed after retries")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
LOG_DIR   = BASE_DIR / "logs"
LOG_FILE  = LOG_DIR / "agentic_doctor.log"

LOG_FORMAT        = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
LOG_DATE_FORMAT   = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES     = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT  = 3                  # keep last 3 rotated files


class _ColorFormatter(logging.Formatter):
    """
    Adds ANSI color codes to terminal output only.
    File output stays plain text (no escape codes in log files).
    """
    COLORS = {
        logging.DEBUG:    "\033[37m",     # white
        logging.INFO:     "\033[36m",     # cyan
        logging.WARNING:  "\033[33m",     # yellow
        logging.ERROR:    "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",   # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color  = self.COLORS.get(record.levelno, "")
        result = super().format(record)
        return f"{color}{result}{self.RESET}"


def _setup_logger() -> logging.Logger:
    """
    Creates and configures the root application logger.
    Called once at import time — subsequent get_logger() calls
    return child loggers that inherit this configuration.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("agentic_doctor")
    root_logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if module is re-imported
    if root_logger.handlers:
        return root_logger

    # ── Terminal handler (INFO and above, with color) ─────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        _ColorFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )

    # ── File handler (DEBUG and above, plain text, rotating) ──────
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


_root_logger = _setup_logger()


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a child logger for a specific module.
    Inherits all handlers from the root agentic_doctor logger.

    Args:
        module_name: typically __name__ from the calling module
                     e.g. 'src.rag.vector_store_manager'

    Usage:
        from src.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    short_name = module_name.replace("src.", "").replace(".", ".")
    return logging.getLogger(f"agentic_doctor.{short_name}")