# src/data_validation/pipelines/bronze_pipeline.py
import logging
import argparse
from pathlib import Path
import sys

from config import (
    GE_ROOT_DIR,
    RAW_DATA_SOURCE,
    BRONZE_CHECKPOINT_NAME,
    RAW_ASSET_NAME,
    BRONZE_SUITE_NAME,
    RAW_DATA_SOURCE_NAME,
    RAW_PENDING_DIR,
    BRONZE_PIPELINE_LOGS_PATH,
    BRONZE_BATCH_DEFINITION_NAME,
    BRONZE_VALIDATION_DEFINITION_NAME,
    RAW_BATCH_PATH,
)
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
from shared.utils import setup_logger

# Create a logger object for this module
logger = logging.getLogger(__name__)


def run_bronze_pipeline(file_path: Path) -> bool:
    """
    Runs the Bronze data validation pipeline on a specified raw data file using
    a static, definition-based workflow.

    Args:
        file_path: The absolute path to the raw data file to validate.

    Returns:
        True if the validation succeeds, False otherwise.
    """

    abs_batch_path = RAW_PENDING_DIR / file_path
    if not abs_batch_path.exists():
        logger.error(f"File not found at {file_path}. Aborting Bronze pipeline.")
        return False

    # --- 1. Initialize GE Context and Datasource ---
    context = get_ge_context(project_root_dir=GE_ROOT_DIR)
    datasource = get_or_create_datasource(
        context=context, source_name=RAW_DATA_SOURCE_NAME, data_dir=RAW_DATA_SOURCE
    )
    asset = get_or_create_csv_asset(datasource=datasource, asset_name=RAW_ASSET_NAME)

    # --- 3. Create Definitions to Link Data and Rules ---
    batch_definition = get_or_create_batch_definition(
        asset=asset,
        batch_definition_name=BRONZE_BATCH_DEFINITION_NAME,
        file_path=file_path,
    )

    # --- 2. Build Expectation Suite ---
    suite = get_or_create_expectation_suite(
        context=context, suite_name=BRONZE_SUITE_NAME
    )
    bronze_expectations = build_bronze_expectations()
    add_expectations_to_suite(suite=suite, expectation_list=bronze_expectations)
    suite.save()

    logger.info(
        f"Bronze expectation suite '{BRONZE_SUITE_NAME}' is built successfully."
    )

    validation_definition = get_or_create_validation_definition(
        context=context,
        definition_name=BRONZE_VALIDATION_DEFINITION_NAME,
        batch_definition=batch_definition,
        suite=suite,
    )
    action_list = get_action_list()

    # --- 4. Run the Checkpoint ---
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=BRONZE_CHECKPOINT_NAME,
        validation_definition_list=[validation_definition],
        action_list=action_list,
    )
    result = run_checkpoint(checkpoint=checkpoint)

    # --- 5. Return Final Status ---
    if not result.success:
        logger.warning(f"--- Bronze Validation Pipeline: FAILED for {file_path} ---")
    else:
        logger.info(f"--- Bronze Validation Pipeline: PASSED for {file_path} ---")

    return result.success


def main():
    """
    Main entry point for running the Bronze pipeline from the command line.
    This function handles logging setup and argument parsing.
    """
    # ----For Testing purposes only-----
    file_path = RAW_BATCH_PATH
    # --- SETUP LOGGING ---
    setup_logger(verbose=True, log_file=BRONZE_PIPELINE_LOGS_PATH, mode="w")
    logger.info(f"--- Starting Bronze Validation Pipeline for: {file_path} ---")

    # parser = argparse.ArgumentParser(
    #     description="Run the Bronze Data Validation Pipeline."
    # )
    # parser.add_argument(
    #     "file_path",
    #     type=str,
    #     help="The relative path to the raw data file to validate.",
    # )
    # args = parser.parse_args()

    # full_file_path = PROJECT_ROOT / args.file_path

    # Run the pipeline and get the success status
    # Run the pipeline and get the success status
    pipeline_success = run_bronze_pipeline(file_path)

    # Exit with a status code that an orchestrator like Airflow can interpret
    if not pipeline_success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    # This allows the script to be run directly using 'python <script_name> <args>'
    main()
