# src/data_validation/ge_components.py

import logging
from pathlib import Path
from typing import List, Any

import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite, ValidationDefinition
from great_expectations.checkpoint import Checkpoint, CheckpointResult
from great_expectations.checkpoint.actions import UpdateDataDocsAction
from great_expectations.data_context import FileDataContext
from great_expectations.datasource.fluent import PandasDatasource, DataAsset

logger = logging.getLogger(__name__)


def get_ge_context(project_root_dir: Path) -> FileDataContext:
    """
    Initializes and returns a Great Expectations FileDataContext.

    This function serves as the primary entry point for interacting with a
    Great Expectations project on the filesystem.

    Args:
        project_root_dir: The absolute path to the Great Expectations
                          project root directory (i.e., the `gx` folder).

    Returns:
        An initialized FileDataContext object.
    """
    logger.info(f"Initializing GE context from directory: {project_root_dir}")
    return gx.get_context(mode="file", project_root_dir=project_root_dir)


def get_or_create_datasource(
    context: FileDataContext, source_name: str, data_dir: Path
) -> PandasDatasource:
    """
    Ensures a datasource exists by deleting it if present and recreating it.

    This "delete-and-recreate" pattern guarantees that the pipeline starts
    with a clean, known datasource configuration for each run, preventing
    issues from leftover state. It is designed for file-based datasources
    in an automated pipeline context.

    Args:
        context: The Great Expectations FileDataContext object.
        source_name: The name for the datasource.
        data_dir: The base directory for the filesystem datasource.

    Returns:
        The newly created or recreated PandasDatasource object.
    """
    logger.debug(f"Attempting to delete and recreate datasource '{source_name}'.")
    try:
        context.data_sources.delete(source_name)
        logger.info(f"Deleted existing datasource '{source_name}' for a clean state.")
    except (ValueError, KeyError):
        logger.info(f"Datasource '{source_name}' not found. Creating new...")

    datasource = context.data_sources.add_pandas_filesystem(
        name=source_name, base_directory=data_dir
    )
    logger.info(f"Datasource '{source_name}' created/recreated successfully.")
    return datasource


def get_or_create_csv_asset(datasource: PandasDatasource, asset_name: str):
    """
    Ensures a CSV asset exists on a datasource by deleting and recreating it.

    This ensures a clean asset state for each pipeline run.

    Args:
        datasource: The PandasDatasource to add the asset to.
        asset_name: The name for the CSV asset.
    """
    logger.debug(f"Attempting to delete and recreate asset '{asset_name}'.")
    try:
        datasource.delete_asset(asset_name)
        logger.info(f"Deleted existing asset '{asset_name}' for a clean state.")
    except (LookupError, KeyError):
        logger.info(f"Asset '{asset_name}' not found. Will create new.")

    asset = datasource.add_csv_asset(name=asset_name)
    logger.info(f"Asset '{asset_name}' created/recreated successfully.")
    return asset


def get_or_create_batch_definition(
    asset: DataAsset, batch_definition_name: str, file_name: str
):
    """
    Adds or updates a batch definition to point to a specific file.

    Using `add_batch_definition_path` is idempotent for a given name if it
    points to a new path.

    Args:
        asset: The DataAsset to which the batch definition will be added.
        batch_definition_name: The name for the batch definition.
        file_name: The relative path of the file to be included in the batch.
    """
    logger.debug(
        f"Attempting to add or update batch definition '{batch_definition_name}'."
    )
    batch_definition = asset.add_batch_definition_path(
        name=batch_definition_name, path=file_name
    )
    logger.info(
        f"Batch definition '{batch_definition_name}' points to file named: {file_name}"
    )
    return batch_definition


def get_or_create_expectation_suite(
    context: FileDataContext, suite_name: str
) -> ExpectationSuite:
    """
    Ensures a fresh, empty Expectation Suite exists for the pipeline run.

    If a suite with the same name already exists, it is deleted and
    recreated to guarantee a clean state.

    Args:
        context: The Great Expectations FileDataContext object.
        suite_name: The name for the expectation suite.

    Returns:
        The newly created, empty ExpectationSuite object.
    """
    try:
        logger.debug(f"Checking for existing suite '{suite_name}' to recreate.")
        context.suites.delete(name=suite_name)
        logger.info(f"Deleted existing suite '{suite_name}' for a clean state.")
    except gx.exceptions.DataContextError:
        # This is expected if the suite doesn't exist yet
        logger.info(f"Expectation suite '{suite_name}' not found. Will create new.")

    suite = ExpectationSuite(name=suite_name)
    context.suites.add(suite=suite)
    logger.info(f"Expectation suite '{suite_name}' created/recreated successfully.")
    return suite


def add_expectations_to_suite(suite: ExpectationSuite, expectation_list: List):
    """
    Adds a list of new expectation objects to an existing suite.

    Args:
        suite: The ExpectationSuite object to be populated.
        expectation_list: A list of Expectation objects to add.
    """
    logger.info(
        f"Adding {len(expectation_list)} expectations to suite '{suite.name}'..."
    )
    for expectation in expectation_list:
        suite.add_expectation(expectation=expectation)
    logger.info("All expectations added successfully.")


def get_or_create_validation_definition(
    context: FileDataContext,
    definition_name: str,
    batch_definition: Any,  # Using Any to avoid new GX imports
    suite: ExpectationSuite,
) -> ValidationDefinition:
    """
    Creates or recreates a Validation Definition linking data to an Expectation Suite.

    Args:
        context: The Great Expectations FileDataContext object.
        definition_name: The name for the validation definition.
        batch_definition: The batch definition representing the data to be validated.
        suite: The ExpectationSuite containing the rules.

    Returns:
        The newly created ValidationDefinition object.
    """
    try:
        context.validation_definitions.delete(definition_name)
        logger.info(
            f"Deleted existing validation definition '{definition_name}' for a clean state."
        )
    except (gx.exceptions.DataContextError, AttributeError, KeyError):
        logger.info(
            f"Validation definition '{definition_name}' not found. Creating new one..."
        )

    validation_def = ValidationDefinition(
        name=definition_name, data=batch_definition, suite=suite
    )
    context.validation_definitions.add(validation_def)
    logger.info(f"Validation definition '{definition_name}' created/recreated.")
    return validation_def


def get_action_list() -> list:
    """
    Returns a default list of actions for checkpoints.

    Currently configured to update all Data Docs sites after a validation run.
    """
    return [
        UpdateDataDocsAction(
            name="update_all_data_docs",
        ),
    ]


def get_or_create_checkpoint(
    context: FileDataContext,
    checkpoint_name: str,
    validation_definition_list: List,
    action_list: List,
    result_format: dict = {"result_format": "COMPLETE"},
) -> Checkpoint:
    """
    Creates or updates a Checkpoint configuration in the Data Context.

    Args:
        context: The Great Expectations FileDataContext object.
        checkpoint_name: The name for the checkpoint.
        validation_definition_list: A list of ValidationDefinition objects to include.
        action_list: A list of actions to perform after validation.
        result_format: The format for the validation results.

    Returns:
        The configured Checkpoint object from the context.
    """
    checkpoint_config = {
        "name": checkpoint_name,
        "validation_definitions": validation_definition_list,
        "actions": action_list,
        "result_format": result_format,
    }
    checkpoint = gx.Checkpoint(**checkpoint_config)

    context.checkpoints.add_or_update(checkpoint)
    checkpoint = context.checkpoints.get(name=checkpoint_name)
    logger.info(f"Checkpoint '{checkpoint_name}' added or updated successfully.")
    return checkpoint


def run_checkpoint(checkpoint: Checkpoint) -> CheckpointResult:
    """
    Executes a Great Expectations Checkpoint and returns the result.

    Includes error handling to prevent pipeline crashes during the run.

    Args:
        checkpoint: The Checkpoint object to run.

    Returns:
        A CheckpointResult object, or None if a critical error occurs.
    """
    logger.info(f"Running checkpoint: '{checkpoint.name}'...")
    try:
        result = checkpoint.run()
        logger.info(f"Checkpoint run completed. Success: {result.success}")
        return result
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during checkpoint run: {e}", exc_info=True
        )
        return None


def run_checkpoint_on_dataframe(
    project_root_dir: str,
    datasource_name: str,
    asset_name: str,
    batch_definition_name: str,
    suite_name: str,
    validation_definition_name: str,
    checkpoint_name: str,
    dataframe_to_validate: pd.DataFrame,
    expectation_list: List,
) -> CheckpointResult:
    """
    Creates and runs a checkpoint to validate an in-memory DataFrame.

    This orchestrates the creation of all necessary temporary GE objects
    (datasource, asset, definitions) to perform the validation.

    Args:
        project_root_dir: The root directory of the Great Expectations project.
        datasource_name: Name for the temporary pandas datasource.
        asset_name: Name for the temporary dataframe asset.
        batch_definition_name: Name for the temporary batch definition.
        suite_name: The name of the expectation suite to use.
        validation_definition_name: Name for the validation definition.
        checkpoint_name: The name for the temporary checkpoint.
        dataframe_to_validate: The pandas DataFrame to be validated.
        expectation_list: A list of Expectation objects to validate against.

    Returns:
        A CheckpointResult object, or None if a critical error occurs.
    """
    context = get_ge_context(project_root_dir=project_root_dir)

    # 1. Add a temporary 'in-memory' datasource.
    try:
        context.data_sources.delete(name=datasource_name)
        logger.info(
            f"Deleted existing datasource '{datasource_name}' for a clean state."
        )
    except (ValueError, KeyError):
        logger.info(f"Datasource '{datasource_name}' not found. Creating new...")
    datasource = context.data_sources.add_pandas(name=datasource_name)

    # 2. Add the DataFrame as a Data Asset
    try:
        datasource.delete_asset(name=asset_name)
    except (LookupError, KeyError):
        pass  # Asset didn't exist, which is fine
    data_asset = datasource.add_dataframe_asset(name=asset_name)

    # 3. Create a Batch Definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name=batch_definition_name
    )

    # 4. Create and populate the Expectation Suite
    suite = get_or_create_expectation_suite(context, suite_name)
    add_expectations_to_suite(suite, expectation_list)
    suite.save()

    # 5. Create the Validation Definition
    validation_definition = get_or_create_validation_definition(
        context=context,
        definition_name=validation_definition_name,
        batch_definition=batch_definition,
        suite=suite,
    )

    # 6. Create the Checkpoint
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=checkpoint_name,
        validation_definition_list=[validation_definition],
        action_list=get_action_list(),
    )

    # 7. Run the checkpoint, passing the DataFrame as a runtime parameter
    try:
        result = checkpoint.run(
            batch_parameters={"dataframe": dataframe_to_validate},
        )
        return result
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during in-memory checkpoint run: {e}",
            exc_info=True,
        )
        return None
