from pathlib import Path
import logging
import argparse
import sys


# --- Local Application Imports ---
from data_validation.expectations.silver_expectations import build_silver_expectations
from data_validation.ge_components import run_checkpoint_on_dataframe
from data_ingestion.data_loader import load_data
from silver_data_preprocessing.silver_preprocessing import (
    rename_specific_columns,
    standardize_column_format,
    optimize_data_types,
    create_date_features,
    handle_erroneous_duplicates,
    sort_data_by_date,
    enforce_column_order,
)

# CORRECTED IMPORT PATTERN FOR TESTABILITY
from shared import config
from shared.utils import setup_logging_from_yaml, save_dataframe_based_on_validation

logger = logging.getLogger(__name__)


def run_silver_pipeline(
    input_filepath: Path,
) -> bool:
    """
    Orchestrates the full Silver layer data processing and validation pipeline.
    """
    logger.info(f"Validating file path: {input_filepath}")
    if not input_filepath.exists():
        logger.error(f"File not found at {input_filepath}. Aborting pipeline.")
        return False

    file_name = Path(input_filepath).name
    logger.info(f"--- Starting Silver Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/5: DATA INGESTION " + "=" * 25)
    df = load_data(file_path=input_filepath)
    if df is None:
        # Added an explicit error log for clarity
        logger.error(f"Failed to load data from {input_filepath}. Aborting pipeline.")
        return False
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: PREPROCESSING & CLEANING ===
    logger.info("=" * 25 + " STAGE 2/5: PREPROCESSING & CLEANING " + "=" * 25)
    df = rename_specific_columns(df, rename_mapping=config.COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    df = optimize_data_types(df, date_cols=["date"])
    df = sort_data_by_date(df, date_column="date")
    df = handle_erroneous_duplicates(df=df, subset_cols=config.ERRONEOUS_DUPE_SUBSET)
    logger.info("Standardization, cleaning, and sorting complete.")

    # === STAGE 3: FEATURE ENGINEERING ===
    logger.info("=" * 25 + " STAGE 3/5: FEATURE ENGINEERING " + "=" * 25)
    df = create_date_features(df, date_column="date")
    logger.info("Date part extraction complete.")

    # === STAGE 4: ENFORCE SCHEMA ORDER ===
    logger.info("=" * 25 + " STAGE 4/5: ENFORCE SCHEMA ORDER " + "=" * 25)
    df = enforce_column_order(df, column_order=config.SILVER_EXPECTED_COLS_ORDER)

    # === STAGE 5: DATA VALIDATION (QUALITY GATE) ===
    logger.info("=" * 25 + " STAGE 5/5: FINAL VALIDATION " + "=" * 25)
    silver_expectations = build_silver_expectations(
        expected_cols_ordered=config.SILVER_EXPECTED_COLS_ORDER,
        expected_col_types=config.SILVER_EXPECTED_COLUMN_TYPES,
        non_null_cols=config.SILVER_REQUIRED_NON_NULL_COLS,
        unique_record_cols=config.ERRONEOUS_DUPE_SUBSET,
    )
    result = run_checkpoint_on_dataframe(
        project_root_dir=config.GE_ROOT_DIR,
        datasource_name=config.SILVER_DATA_SOURCE_NAME,
        asset_name=config.SILVER_ASSET_NAME,
        batch_definition_name=config.SILVER_BATCH_DEFINITION_NAME,
        suite_name=config.SILVER_SUITE_NAME,
        validation_definition_name=config.SILVER_VALIDATION_DEFINITION_NAME,
        checkpoint_name=config.SILVER_CHECKPOINT_NAME,
        dataframe_to_validate=df,
        expectation_list=silver_expectations,
    )

    # === STAGE 6: SAVE DATAFRAME BASED ON RESULT ===
    save_successful = save_dataframe_based_on_validation(
        result=result,
        df=df,
        file_name=Path(file_name).stem,
        success_dir=config.SILVER_PROCESSED_DIR,
        failure_dir=config.SILVER_QUARANTINE_DIR,
    )

    # The pipeline's true success depends on BOTH validation AND the save operation
    pipeline_successful = result.success and save_successful

    # === STAGE 7: LOG FINAL STATUS ===
    if pipeline_successful:
        logger.info(
            f"--- Silver Preprocessing & Validation: PASSED for {file_name} ---"
        )
    else:
        logger.warning(
            f"--- Silver Preprocessing & Validation: FAILED for {file_name} ---"
        )

    return pipeline_successful


def main():
    """
    Main entry point for running the Silver pipeline from the command line.
    """
    # --- SETUP LOGGING ---
    # load logging configuration
    setup_logging_from_yaml(
        log_path=config.SILVER_PIPELINE_LOGS_PATH,
        default_level=logging.DEBUG,
        default_yaml_path=config.LOGGING_YAML,
    )
    parser = argparse.ArgumentParser(
        description="Run the Silver Data Processing Pipeline."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The name of the file in the 'processed' directory to clean (e.g., 'train.csv').",
    )

    args = parser.parse_args()

    input_filepath = config.BRONZE_PROCESSED_DIR / args.input_file

    pipeline_success = run_silver_pipeline(
        input_filepath=Path(input_filepath),
    )

    # Exit with a status code that an orchestrator like Airflow can interpret
    if not pipeline_success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
