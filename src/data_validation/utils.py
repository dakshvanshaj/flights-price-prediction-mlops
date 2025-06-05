from load_data import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    get_or_create_batch_definition,
    load_batch_from_definition,
)
import logging

logger = logging.getLogger(__name__)


def setup_logger(verbose: bool = True):
    """
    Configure logger level based on verbose flag.
    Call this once at the start of your app.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="./logs/run_expectations_suite.log",
        filemode="w",
    )
    logger.info(f"Logger initialized. Verbose mode: {verbose}")


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
