# delete and recreate stratify for on the fly working of pipeline

import logging
from pathlib import Path
from typing import List
import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite, ValidationDefinition
from great_expectations.checkpoint import Checkpoint, CheckpointResult
from great_expectations.checkpoint.actions import UpdateDataDocsAction
from great_expectations.data_context import FileDataContext
from great_expectations.datasource.fluent import PandasDatasource

logger = logging.getLogger(__name__)


def get_ge_context(project_root_dir: Path) -> FileDataContext:
    logger.info(f"Initializing GE context from directory: {project_root_dir}")
    return gx.get_context(mode="file", project_root_dir=project_root_dir)


def get_or_create_datasource(
    context: FileDataContext, source_name: str, data_dir: Path
) -> PandasDatasource:
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
    logger.debug(f"Attempting to delete and recreate asset '{asset_name}'.")
    try:
        datasource.delete_asset(asset_name)
        logger.info(f"Deleted existing asset '{asset_name}' for a clean state.")
    except (LookupError, KeyError):
        logger.info(f"Asset '{asset_name}' not found. Will create new.")

    asset = datasource.add_csv_asset(name=asset_name)
    logger.info(f"Asset '{asset_name}' created/recreated successfully.")
    return asset


def get_or_create_batch_definition(asset, batch_definition_name: str, file_name: str):
    logger.debug(
        f"Attempting to add or update batch definition '{batch_definition_name}'."
    )
    # Using add_batch_definition_path is idempotent for a given name if it points to a new path
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
    """
    try:
        logger.debug(f"Checking for existing suite '{suite_name}' to recreate.")
        context.suites.delete(name=suite_name)
        logger.info(f"Deleted existing suite '{suite_name}' for a clean state.")
    except gx.exceptions.DataContextError:
        # This is expected if the suite doesn't exist yet
        logger.info(f"Expectation suite '{suite_name}' not found. Will create new.")

    # Create the new suite
    suite = ExpectationSuite(name=suite_name)
    context.suites.add(suite=suite)
    logger.info(f"Expectation suite '{suite_name}' created/recreated successfully.")
    return suite


def add_expectations_to_suite(suite: ExpectationSuite, expectation_list: List):
    """
    Adds a list of new expectations to a suite. Assumes the suite is
    ready to be populated (e.g., is empty).
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
    batch_definition,
    suite: ExpectationSuite,
):
    try:
        context.validation_definitions.delete(definition_name)
        logger.info(
            f"Deleted existing validation definition '{definition_name}' for a clean state."
        )
    except (gx.exceptions.DataContextError, AttributeError, KeyError):
        logger.info(
            f"Validation definition '{definition_name}' not found. . Creating new one..."
        )

    validation_def = ValidationDefinition(
        name=definition_name, data=batch_definition, suite=suite
    )
    context.validation_definitions.add(validation_def)
    logger.info(f"Validation definition '{definition_name}' created/recreated.")
    return validation_def


def get_action_list() -> list:
    """Returns a default list of actions for checkpoints."""
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
    result_format={"result_format": "COMPLETE"},
) -> Checkpoint:
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
    logger.info(f"Running checkpoint: '{checkpoint.name}'...")
    result = checkpoint.run()
    logger.info(f"Checkpoint run completed. Success: {result.success}")
    return result


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
    Creates and runs a checkpoint to validate an in-memory DataFrame, following
    the Validation Definition pattern.

    Args:
        project_root_dir: The root directory of the Great Expectations project.
        suite_name: The name of the expectation suite to use.
        checkpoint_name: The name for the temporary checkpoint.
        dataframe_to_validate: The pandas DataFrame to be validated.
        expectation_list: A list of Expectation objects to validate against.

    Returns:
        A CheckpointResult object containing the validation results.
    """
    context = get_ge_context(project_root_dir=project_root_dir)

    # 1. Add a temporary 'in-memory' datasource. We use the delete/recreate pattern.
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
        pass
    data_asset = datasource.add_dataframe_asset(name=asset_name)

    # 3. Create a Batch Definition that points to our DataFrame asset
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name=batch_definition_name
    )

    # 4. Get or create the Expectation Suite and add our rules to it
    suite = get_or_create_expectation_suite(context, suite_name)
    add_expectations_to_suite(suite, expectation_list)
    suite.save()

    # 5. Create a Validation Definition linking the data and the suite
    validation_definition = get_or_create_validation_definition(
        context=context,
        definition_name=validation_definition_name,
        batch_definition=batch_definition,  # pass batch directly with dataframe_to_validate
        suite=suite,
    )

    # 6. Create the Checkpoint using the Validation Definition
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=checkpoint_name,
        validation_definition_list=[validation_definition],
        action_list=get_action_list(),
    )

    # 7. Run the checkpoint and return the result
    try:
        logger.info(f"Running checkpoint: '{checkpoint.name}'...")
        result = checkpoint.run(
            batch_parameters={"dataframe": dataframe_to_validate},
        )
        logger.info(f"Checkpoint run completed. Success: {result.success}")
    except Exception as e:
        logger.error(f"Error running checkpoint: {e}", exc_info=True)

    return result
