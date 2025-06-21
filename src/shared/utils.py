# src/shared/utils.py
import logging
import sys
from pathlib import Path


def setup_logger(verbose: bool = True, log_file: str = "app.log", mode: str = "w"):
    """
    Configures the root logger for the application.

    This function sets up handlers for both console and file logging,
    and adjusts log levels for noisy third-party libraries.

    Args:
        verbose: If True, sets the root logger level to DEBUG. Otherwise, INFO.
        log_file: The path to the log file.
        mode: The file mode for the log file ('w' for write, 'a' for append).
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # --- Reduce Log Noise from Third-Party Libraries ---
    # Set noisy libraries to a higher log level to silence their DEBUG/INFO messages.
    logging.getLogger("great_expectations").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tzlocal").setLevel(logging.WARNING)

    # Create the log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Detailed formatter for file logs
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Simple formatter for console logs
    console_formatter = logging.Formatter("%(levelname)s: %(name)s - %(message)s")

    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.debug(f"Root logger initialized (Verbose: {verbose}).")
