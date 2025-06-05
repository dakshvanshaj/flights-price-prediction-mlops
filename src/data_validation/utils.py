import sys
from load_data import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    get_or_create_batch_definition,
    load_batch_from_definition,
)
import logging


def setup_logger(verbose: bool = True, log_file: str = "app.log", mode: str = "w"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Detailed formatter for file logs
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Simple formatter for console logs
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(console_formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(
        f"Logger initialized. Verbose: {verbose}, Log file: {log_file}, Mode: {mode}"
    )


def initialize_ge_components(
    root_dir: str,
    source_name: str,
    base_dir: str,
    asset_name: str,
    batch_name: str,
    batch_path: str,
):
    """
    Initialize and return core GE components including validation definition.

    Returns:
        tuple: (
            context,
            data_source,
            csv_asset,
            batch_definition,
            batch,
        )
    """
    context = get_ge_context(project_root_dir=root_dir)
    data_source = get_or_create_datasource(context, source_name, base_dir)
    csv_asset = get_or_create_csv_asset(data_source, asset_name)
    batch_definition = get_or_create_batch_definition(csv_asset, batch_name, batch_path)
    batch = load_batch_from_definition(batch_definition)

    return (context, data_source, csv_asset, batch_definition, batch)
