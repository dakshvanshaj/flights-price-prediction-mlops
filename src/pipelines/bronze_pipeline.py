# src/pipelines/bronze_pipeline.py
import logging
import argparse
from pathlib import Path
import sys
from shared.config import config_bronze, core_paths, config_logging
from data_validation.expectations.bronze_expectations import build_bronze_expectations
from data_validation.ge_components import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    get_or_create_batch_definition,
    get_or_create_expectation_suite,
    add_expectations_to_suite,
    get_or_create_validation_definition,
    get_action_list,
    get_or_create_checkpoint,
    run_checkpoint,
)
from shared.utils import setup_logging_from_yaml, handle_file_based_on_validation

# Create a logger object for this module
logger = logging.getLogger(__name__)


def run_bronze_pipeline(file_name: str) -> bool:
    """
    Runs the Bronze data validation pipeline on a specified raw data file
    to ensure it meets the quality standards defined in the Bronze expectations.

    Args:
        file_name: The name of the raw data file in folder to validate.

    Returns:
        True if the validation succeeds, False otherwise.
    """
    file_path = Path(file_name)
    file_path = config_bronze.RAW_DATA_SOURCE / file_path
    logger.info(f"Validating file: {file_path}")
    if not file_path.exists():
        logger.error(f"File not found at {file_path}. Aborting Bronze pipeline.")
        return False

    # --- 1. Initialize GE Context and Datasource ---
    context = get_ge_context(project_root_dir=core_paths.GE_ROOT_DIR)
    datasource = get_or_create_datasource(
        context=context,
        source_name=config_bronze.RAW_DATA_SOURCE_NAME,
        data_dir=config_bronze.RAW_DATA_SOURCE,
    )
    asset = get_or_create_csv_asset(
        datasource=datasource, asset_name=config_bronze.RAW_ASSET_NAME
    )

    # --- 2. Create Definitions to Link Data and Rules ---
    batch_definition = get_or_create_batch_definition(
        asset=asset,
        batch_definition_name=config_bronze.BRONZE_BATCH_DEFINITION_NAME,
        file_name=file_name,
    )

    suite = get_or_create_expectation_suite(
        context=context, suite_name=config_bronze.BRONZE_SUITE_NAME
    )
    bronze_expectations = build_bronze_expectations()
    add_expectations_to_suite(suite=suite, expectation_list=bronze_expectations)
    suite.save()

    logger.info(
        f"Bronze expectation suite '{config_bronze.BRONZE_SUITE_NAME}' is built successfully."
    )

    validation_definition = get_or_create_validation_definition(
        context=context,
        definition_name=config_bronze.BRONZE_VALIDATION_DEFINITION_NAME,
        batch_definition=batch_definition,
        suite=suite,
    )
    action_list = get_action_list()

    # --- 3. Run the Checkpoint ---
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=config_bronze.BRONZE_CHECKPOINT_NAME,
        validation_definition_list=[validation_definition],
        action_list=action_list,
    )
    result = run_checkpoint(checkpoint=checkpoint)

    # --- 4. Move File Based on Result ---
    file_op_successful = handle_file_based_on_validation(
        result=result,
        file_path=file_path,
        success_dir=config_bronze.BRONZE_PROCESSED_DIR,
        failure_dir=config_bronze.BRONZE_QUARANTINE_DIR,
    )

    # The pipeline's true success depends on BOTH validation AND the file move
    pipeline_successful = result.success and file_op_successful

    # --- 5. Log Final Status ---
    if pipeline_successful:
        logger.info(f"--- Bronze Validation Pipeline: PASSED for {file_path.name} ---")
    else:
        logger.warning(
            f"--- Bronze Validation Pipeline: FAILED for {file_path.name} ---"
        )

    # Return the final, combined success status
    return pipeline_successful


def main():
    """
    Main entry point for running the Bronze pipeline from the command line.
    This function handles logging setup and argument parsing.
    """
    # --- SETUP LOGGING ---
    # load logging configuration
    setup_logging_from_yaml(
        log_path=config_logging.BRONZE_PIPELINE_LOGS_PATH,
        default_level=logging.DEBUG,
        default_yaml_path=config_logging.LOGGING_YAML,
    )
    # --- PARSE COMMAND-LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(
        description="Run the Bronze Data Validation Pipeline."
    )
    parser.add_argument(
        "file_name",
        type=str,
        help="The name of the raw data file in the 'raw' directory (e.g., 'train.csv').",
    )
    args = parser.parse_args()

    logger.info(f"--- Starting Bronze Validation Pipeline for: {args.file_name} ---")

    file_name = args.file_name

    # --- RUN THE PIPELINE ---
    pipeline_success = run_bronze_pipeline(file_name=file_name)

    # Exit with a status code that an orchestrator like Airflow can interpret
    if not pipeline_success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
