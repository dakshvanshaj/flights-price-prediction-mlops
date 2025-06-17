# src/data_validation/ge_components.py
import logging
from pathlib import Path
from typing import List, Optional, Union
import copy

import great_expectations as gx
from great_expectations.checkpoint import Checkpoint, CheckpointResult
from great_expectations.checkpoint.actions import UpdateDataDocsAction
from great_expectations.core import ExpectationSuite, ValidationDefinition
from great_expectations.expectations.expectation_configuration import (
    ExpectationConfiguration,
)
from great_expectations.data_context import FileDataContext
from great_expectations.datasource.fluent import PandasDatasource

# Create a logger object for this module
logger = logging.getLogger(__name__)


# --- Context and Datasource Components ---


def get_ge_context(project_root_dir: Path) -> FileDataContext:
    """Initialize and return a file-based Great Expectations DataContext."""
    logger.info(f"Initializing GE context from directory: {project_root_dir}")
    return gx.get_context(mode="file", project_root_dir=project_root_dir)


def get_or_create_datasource(
    context: FileDataContext, source_name: str, data_dir: Path
) -> PandasDatasource:
    """Retrieve an existing datasource or create a new one."""
    try:
        datasource = context.get_datasource(source_name)
        logger.info(f"Loaded existing datasource: '{source_name}'")
    except ValueError:
        logger.info(f"Datasource '{source_name}' not found. Creating new one...")
        datasource = context.data_sources.add_pandas_filesystem(
            name=source_name, base_directory=data_dir
        )
        logger.info(f"Datasource '{source_name}' created successfully.")
    return datasource


# --- Asset and Batch Definition Components ---


def get_or_create_csv_asset(datasource: PandasDatasource, asset_name: str):
    """Retrieve an existing CSV asset or create a new one."""
    try:
        csv_asset = datasource.get_asset(asset_name)
        logger.info(f"Loaded existing asset: '{asset_name}'")
    except LookupError:
        logger.info(f"Asset '{asset_name}' not found. Creating new one...")
        csv_asset = datasource.add_csv_asset(name=asset_name)
        logger.info(f"Asset '{asset_name}' created successfully.")
    return csv_asset


def get_or_create_batch_definition(asset, batch_definition_name: str, file_path: Path):
    """Create or load a static batch definition that points to a specific file."""
    try:
        batch_definition = asset.get_batch_definition(batch_definition_name)
        logger.info(f"Loaded existing batch definition: '{batch_definition_name}'")
        # Update the path in case the file location changes between runs for the same def
        batch_definition.path = str(file_path)
    except Exception:
        logger.info(
            f"Batch definition '{batch_definition_name}' not found. Creating new one..."
        )
        batch_definition = asset.add_batch_definition_path(
            name=batch_definition_name, path=str(file_path)
        )
        logger.info(
            f"Batch definition '{batch_definition_name}' created for path: {file_path}"
        )
    return batch_definition


# --- Expectation Suite Components ---


def get_or_create_expectation_suite(
    context: FileDataContext, suite_name: str
) -> ExpectationSuite:
    """Retrieve an existing Expectation Suite or create a new one."""
    try:
        suite = context.suites.get(name=suite_name)
        logger.info(f"Loaded existing expectation suite: '{suite_name}'")
    except gx.exceptions.DataContextError:
        logger.info(f"Expectation suite '{suite_name}' not found. Creating new one...")
        suite = ExpectationSuite(name=suite_name)
        context.suites.add(suite=suite)
        logger.info(f"Expectation suite '{suite_name}' created successfully.")
    return suite


def upsert_expectation(
    suite: ExpectationSuite,
    expectation: Union[ExpectationConfiguration, List[ExpectationConfiguration]],
):
    """
    Add or update expectations in a suite using a robust dictionary-based method.
    This ensures idempotency and avoids list modification errors.
    """
    if not isinstance(expectation, list):
        expectations_to_add = [expectation]
    else:
        expectations_to_add = expectation

    # Build a dictionary of existing expectations for efficient lookup
    # The value is the config object, which is what we need to add back.
    existing_expectations_dict = {
        exp.meta.get("name"): exp.configuration for exp in suite.expectations
    }

    # Add or update expectations in our dictionary
    for exp_config in expectations_to_add:
        name = exp_config.meta.get("name")
        if not name:
            raise ValueError(
                "Expectation must have a 'meta.name' key for upsert logic."
            )
        # This will add new or overwrite existing expectations by name
        existing_expectations_dict[name] = exp_config
        logger.info(f"Prepared upsert for expectation: '{name}'")

    # Clear the suite's current expectations
    suite.expectations = []

    # Re-add all expectations from our clean dictionary
    for exp_config in existing_expectations_dict.values():
        suite.add_expectation(exp_config)


# --- Validation and Checkpoint Components ---


def get_or_create_validation_definition(
    context: FileDataContext,
    definition_name: str,
    batch_definition,
    suite: ExpectationSuite,
) -> ValidationDefinition:
    """Create or load a static link between a BatchDefinition and an ExpectationSuite."""
    try:
        validation_def = context.validation_definitions.get(definition_name)
        logger.info(f"Loaded existing validation definition: '{definition_name}'")
    except (gx.exceptions.DataContextError, AttributeError):
        logger.info(
            f"Validation definition '{definition_name}' not found. Creating new one..."
        )
        validation_def = ValidationDefinition(
            name=definition_name, data=batch_definition, suite=suite
        )
        context.validation_definitions.add(validation_def)
        logger.info(f"Validation definition '{definition_name}' created.")
    return validation_def


def get_default_actions() -> list:
    """Returns a default list of actions for checkpoints."""
    return [UpdateDataDocsAction()]


def get_or_create_checkpoint(
    context: FileDataContext,
    checkpoint_name: str,
    validation_definition_names: List[str],
    action_list: Optional[List] = None,
) -> Checkpoint:
    """Retrieve or create a checkpoint configured to run a list of ValidationDefinitions."""
    if action_list is None:
        action_list = get_default_actions()

    try:
        checkpoint = context.checkpoints.get(name=checkpoint_name)
        logger.info(f"Loaded existing checkpoint: '{checkpoint_name}'. Updating...")
        checkpoint.validation_definitions = validation_definition_names
        checkpoint.action_list = action_list
        context.add_or_update_checkpoint(checkpoint=checkpoint)

    except gx.exceptions.CheckpointNotFoundError:
        logger.info(f"Checkpoint '{checkpoint_name}' not found. Creating new one...")
        checkpoint = Checkpoint(
            name=checkpoint_name,
            validation_definitions=validation_definition_names,
            action_list=action_list,
            data_context=context,
        )
        context.add_or_update_checkpoint(checkpoint=checkpoint)
        logger.info(f"Checkpoint '{checkpoint_name}' created successfully.")
    return checkpoint


def run_checkpoint(checkpoint: Checkpoint) -> CheckpointResult:
    """Runs a Great Expectations checkpoint and returns the result."""
    logger.info(f"Running checkpoint: '{checkpoint.name}'...")
    result = checkpoint.run()
    logger.info(f"Checkpoint run completed. Success: {result.success}")
    return result
