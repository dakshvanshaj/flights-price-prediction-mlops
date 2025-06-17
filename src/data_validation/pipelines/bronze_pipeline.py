# src/data_validation/pipelines/bronze_pipeline.py
import logging
import argparse
from pathlib import Path
import sys

from config import (
    GE_ROOT_DIR,
    DATA_DIR,
    BRONZE_CHECKPOINT_NAME,
    BRONZE_RAW_ASSET_NAME,
    BRONZE_SUITE_NAME,
    RAW_DATA_SOURCE_NAME,
    BRONZE_PIPELINE_LOGS_PATH,
    BRONZE_BATCH_DEFINITION_NAME,
    BRONZE_VALIDATION_DEFINITION_NAME,
)
from data_validation.expectations.bronze_expectations import build_bronze_expectations
from data_validation.ge_components import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    get_or_create_batch_definition,
    get_or_create_expectation_suite,
    upsert_expectation,
    get_or_create_validation_definition,
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
    # --- SETUP LOGGING ---
    setup_logger(verbose=True, log_file=BRONZE_PIPELINE_LOGS_PATH, mode="w")
    logger.info(f"--- Starting Bronze Validation Pipeline for: {file_path.name} ---")

    if not file_path.exists():
        logger.error(f"File not found at {file_path}. Aborting Bronze pipeline.")
        return False

    # --- 1. Initialize GE Context and Datasource ---
    context = get_ge_context(project_root_dir=GE_ROOT_DIR)
    datasource = get_or_create_datasource(
        context=context, source_name=RAW_DATA_SOURCE_NAME, data_dir=DATA_DIR
    )
    asset = get_or_create_csv_asset(
        datasource=datasource, asset_name=BRONZE_RAW_ASSET_NAME
    )

    # --- 2. Build Expectation Suite ---
    suite = get_or_create_expectation_suite(
        context=context, suite_name=BRONZE_SUITE_NAME
    )
    bronze_expectations = build_bronze_expectations()
    upsert_expectation(suite=suite, expectation=bronze_expectations)
    # context.suites.save(suite=suite)
    logger.info(f"Bronze expectation suite '{BRONZE_SUITE_NAME}' is built and saved.")

    # --- 3. Create Definitions to Link Data and Rules ---
    batch_definition = get_or_create_batch_definition(
        asset=asset,
        batch_definition_name=BRONZE_BATCH_DEFINITION_NAME,
        file_path=file_path,
    )
    validation_definition = get_or_create_validation_definition(
        context=context,
        definition_name=BRONZE_VALIDATION_DEFINITION_NAME,
        batch_definition=batch_definition,
        suite=suite,
    )

    # --- 4. Run the Checkpoint ---
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=BRONZE_CHECKPOINT_NAME,
        validation_definition_names=[validation_definition.name],
    )
    result = run_checkpoint(checkpoint=checkpoint)

    # --- 5. Return Final Status ---
    if not result.success:
        logger.warning(
            f"--- Bronze Validation Pipeline: FAILED for {file_path.name} ---"
        )
    else:
        logger.info(f"--- Bronze Validation Pipeline: PASSED for {file_path.name} ---")

    return result.success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Bronze Data Validation Pipeline."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The relative path to the raw data file to validate.",
    )
    args = parser.parse_args()

    # We need to import PROJECT_ROOT here, inside this block, to avoid circular
    # import issues if other modules were to import this script.
    from config import PROJECT_ROOT

    full_file_path = PROJECT_ROOT / args.file_path

    # Run the pipeline and get the success status
    pipeline_success = run_bronze_pipeline(file_path=full_file_path)

    # Exit with a status code that an orchestrator like Airflow can interpret
    if not pipeline_success:
        sys.exit(1)
    sys.exit(0)
