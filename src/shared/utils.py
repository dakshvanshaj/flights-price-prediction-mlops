import logging
import sys


def setup_logger(verbose: bool = True, log_file: str = "app.log", mode: str = "w"):
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Set the great_expectations logger to INFO level to hide its verbose DEBUG messages
    logging.getLogger("great_expectations").setLevel(logging.INFO)
    # You can add other libraries here if they are also too noisy
    # logging.getLogger("another_library").setLevel(logging.WARNING)

    # Detailed formatter for file logs
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )  # Simplified for console readability

    # Simple formatter for console logs
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.debug(
        f"Root logger initialized (Verbose: {verbose}). GE logger set to INFO."
    )
