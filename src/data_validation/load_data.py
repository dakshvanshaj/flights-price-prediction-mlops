import logging
import great_expectations as gx
from great_expectations import expectations as gxe

# create a logger object
logger = logging.getLogger(__name__)


def get_ge_context(project_root_dir: str):
    """
    Initialize and return Great Expectations context.

    Args:
        project_root_dir: Path to the GE project root directory.

    Returns:
        Great Expectations DataContext object.
    """
    logger.info(f"Initializing GE context with root dir: {project_root_dir}")
    return gx.get_context(mode="file", project_root_dir=project_root_dir)


def get_or_create_datasource(context, data_source_name: str, base_directory: str):
    """
    Retrieve existing data source by name or create a new pandas filesystem data source.

    Args:
        context: Great Expectations DataContext.
        data_source_name: Name of the data source.
        base_directory: Base directory path for the data source.

    Returns:
        DataSource object.
    """
    existing_datasources = context.list_datasources()
    for ds_config in existing_datasources:
        if ds_config["name"] == data_source_name:
            logger.info(f"Using existing data source: {data_source_name}")
            return context.data_sources.get(data_source_name)

    logger.info(f"Initializing new data source: {data_source_name}")
    try:
        data_source = context.data_sources.add_pandas_filesystem(
            name=data_source_name, base_directory=base_directory
        )
        logger.info("Data source created successfully")
        return data_source
    except Exception as e:
        logger.error(f"Failed to create data source: {e}")
        raise


def get_or_create_csv_asset(data_source, asset_name: str):
    """
    Retrieve existing CSV asset or create a new one in the data source.

    Args:
        data_source: Great Expectations DataSource object.
        asset_name: Name of the CSV asset.

    Returns:
        CSV asset object.
    """
    try:
        file_csv_asset = data_source.get_asset(asset_name)
        logger.info(f"Using existing asset: {asset_name}")
        return file_csv_asset
    except Exception:
        try:
            file_csv_asset = data_source.add_csv_asset(name=asset_name)
            logger.info(f"CSV asset '{asset_name}' created")
            return file_csv_asset
        except Exception as e:
            logger.error(f"Failed to create asset '{asset_name}': {e}")
            raise


def get_or_create_batch_definition(
    file_csv_asset, batch_definition_name: str, path_to_batch_file: str
):
    """
    Create or load a batch definition for a specific file.

    Args:
        file_csv_asset: CSV asset object.
        batch_definition_name: Name for the batch definition.
        path_to_batch_file: Path to the CSV file.

    Returns:
        BatchDefinition object.
    """
    try:
        batch_definition = file_csv_asset.get_batch_definition(batch_definition_name)
        logger.info(
            f"Batch definition '{batch_definition_name}' already exists. Using existing one."
        )
    except Exception:
        batch_definition = file_csv_asset.add_batch_definition_path(
            name=batch_definition_name, path=path_to_batch_file
        )
        logger.info(f"Batch definition '{batch_definition_name}' created.")
    return batch_definition


def load_batch_from_definition(batch_definition):
    """
    Load a batch dataset from a given batch definition.

    Args:
        batch_definition: BatchDefinition object.

    Returns:
        Loaded batch dataset.
    """
    try:
        batch = batch_definition.get_batch()
        logger.info("Batch loaded successfully")
        logger.info(f"Sample data preview:\n{batch.head()}")
        return batch
    except FileNotFoundError:
        logger.error(f"Batch file not found: {batch_definition.path}")
        logger.error("Ensure data pipeline has generated the required training file")
        raise
    except Exception as e:
        logger.error(f"Batch loading failed: {e}")
        logger.error("Check data source configuration and file permissions")
        raise


def expect_column_max_to_be_between(
    batch, column: str, min_value=1, max_value=2000, strict_max=False, strict_min=False
):
    """
    Add an expectation that the max value of a column is between min_value and max_value.

    Args:
        batch: Great Expectations batch or dataset object.
        column: Column name to validate.
        min_value: Minimum allowed max value.
        max_value: Maximum allowed max value.

    Returns:
        Validation result dictionary.
    """
    range_expectation = gxe.ExpectColumnMaxToBeBetween(
        column=column,
        min_value=min_value,
        max_value=max_value,
        strict_max=strict_max,
        strict_min=strict_min,
    )

    result = batch.validate(range_expectation)
    logger.info(f"Validation result for column '{column}': {result}")
    return result
