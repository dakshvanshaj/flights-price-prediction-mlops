# src/data_validation/pipelines/bronze_pipeline.py
import logging
import argparse
from pathlib import Path
import sys
import shutil

from shared.config import (
    GE_ROOT_DIR,
    RAW_DATA_SOURCE,
    RAW_PROCESSED_DIR,
    RAW_QUARANTINE_DIR,
    BRONZE_CHECKPOINT_NAME,
    RAW_ASSET_NAME,
    BRONZE_SUITE_NAME,
    RAW_DATA_SOURCE_NAME,
    BRONZE_PIPELINE_LOGS_PATH,
    BRONZE_BATCH_DEFINITION_NAME,
    BRONZE_VALIDATION_DEFINITION_NAME,
    LOGGING_YAML
    
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
from shared.utils import setup_logging_from_yaml

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
    file_path = RAW_DATA_SOURCE / file_path
    logger.info(f"Validating file: {file_path}")
    if not file_path.exists():
        logger.error(f"File not found at {file_path}. Aborting Bronze pipeline.")
        return False

    # --- 1. Initialize GE Context and Datasource ---
    context = get_ge_context(project_root_dir=GE_ROOT_DIR)
    datasource = get_or_create_datasource(
        context=context, source_name=RAW_DATA_SOURCE_NAME, data_dir=RAW_DATA_SOURCE
    )
    asset = get_or_create_csv_asset(datasource=datasource, asset_name=RAW_ASSET_NAME)

    # --- 2. Create Definitions to Link Data and Rules ---
    # NOTE: This assumes the `file_name` provided to this function is what's
    # needed by `get_or_create_batch_definition`.
    batch_definition = get_or_create_batch_definition(
        asset=asset,
        batch_definition_name=BRONZE_BATCH_DEFINITION_NAME,
        file_name=file_name,
    )

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

    # --- 3. Run the Checkpoint ---
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=BRONZE_CHECKPOINT_NAME,
        validation_definition_list=[validation_definition],
        action_list=action_list,
    )
    result = run_checkpoint(checkpoint=checkpoint)

    # --- 4. Move File Based on Result ---
    # Ensure the destination directories exist
    RAW_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    if result.success:
        logger.info(f"--- Bronze Validation Pipeline: PASSED for {file_name} ---")
        destination_path = RAW_PROCESSED_DIR / file_path.name
        logger.info(f"Moving validated file to: {destination_path}")
        shutil.move(src=file_path, dst=destination_path)
    else:
        logger.warning(f"--- Bronze Validation Pipeline: FAILED for {file_name} ---")
        destination_path = RAW_QUARANTINE_DIR / file_path.name
        logger.warning(f"Moving failed file to quarantine: {destination_path}")
        shutil.move(src=file_path, dst=destination_path)

    return result.success


def main():
    """
    Main entry point for running the Bronze pipeline from the command line.
    This function handles logging setup and argument parsing.
    """
    # --- SETUP LOGGING ---
    # load logging configuration
    setup_logging_from_yaml(
        log_path=BRONZE_PIPELINE_LOGS_PATH,
        default_level=logging.DEBUG,
        default_yaml_path=LOGGING_YAML,
    )
    # --- PARSE COMMAND-LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(
        description="Run the Bronze Data Validation Pipeline."
    )
    parser.add_argument(
        "file_name",
        type=str,
        help="The name of the raw data file in the 'pending' directory (e.g., 'flights_2022-02.csv').",
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


# -----------------------------------------OVERVIEW-----------------------------------------

# flow or steps in the Bronze pipeline:

# ge context created at GE_ROOT_DIR
# datasource created at RAW_DATA_SOURCE   -> Points to a folder with raw data files to validate
# csv asset created at RAW_ASSET_NAME -> Points to csv files in the datasource
# batch definition created for file_name  -> Points to a sepecific raw data file,
# throws a regex error if absolute or any path is specified for the raw data file
# expectation suite with name BRONZE_SUITE_NAME
# expectations added to the suite from build_bronze_expectations()
# validation definition created at BRONZE_VALIDATION_DEFINITION_NAME
# checkpoint created at BRONZE_CHECKPOINT_NAME
