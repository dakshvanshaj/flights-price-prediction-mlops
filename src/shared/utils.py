# src/shared/utils.py
import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


def setup_logger(
    name: Optional[str] = None,
    verbose: bool = True,
    log_file: str = "logs/app.log",
    mode: str = "a",
    use_json: bool = False,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
    use_utc: bool = False,
) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name: Name of the logger (None for root logger).
        verbose: If True, sets logger level to DEBUG. Otherwise, INFO.
        log_file: Path to the log file.
        mode: File write mode ('a' or 'w').
        use_json: If True, formats file logs in JSON (requires `python-json-logger`).
        max_bytes: Maximum size (in bytes) for each log file.
        backup_count: Number of rotated backups to keep.
        use_utc: If True, log timestamps in UTC; else use local time.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False  # Avoid duplicate logs if logger is child

    # Quiet noisy libs
    logging.getLogger("great_expectations").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tzlocal").setLevel(logging.WARNING)

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers (idempotency)
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- File Handler (Rotating) ---
    file_handler = RotatingFileHandler(
        log_path, mode=mode, maxBytes=max_bytes, backupCount=backup_count
    )

    if use_json and HAS_JSON_LOGGER:
        file_formatter = jsonlogger.JsonFormatter()
    else:
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] [%(name)-35s] [%(filename)s:%(lineno)d] %(message)s"
        )
        file_formatter.converter = time.gmtime if use_utc else time.localtime

    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        fmt="[%(levelname)-8s] [%(name)-25s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logger initialized.")
    return logger
