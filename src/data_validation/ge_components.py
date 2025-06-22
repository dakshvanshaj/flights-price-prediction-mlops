# delete and recreate stratify for on the fly working of pipeline

import logging
from pathlib import Path
from typing import List

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
