from data_ingestion.data_loader import load_data
from shared.config import config_gold, config_logging, config_silver
from shared.utils import setup_logging_from_yaml, save_dataframe_based_on_validation
from pathlib import Path
import argparse
import sys
import logging
from gold_data_preprocessing.data_cleaning import (
    drop_columns,
    drop_duplicates,
    drop_missing_target_rows,
)

logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: Path,
) -> bool:
    logger.info(f"Validating file path: {input_filepath}")
    if not input_filepath.exists():
        logger.error(f"File not found at {input_filepath}. Aborting pipeline.")
        return False

    file_name = Path(input_filepath).name
    logger.info(f"--- Starting Gold Engineering Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/?: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        logger.error(f"Failed to load data from {input_filepath}. Aborting pipeline.")
        return False
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: Dropping Unique Id Columns ==
    logger.info(
        "=" * 25 + " STAGE 2/?: DROPPING COLUMNS AND HANDLING DUPLICATES " + "=" * 25
    )
    df = drop_columns(df, columns_to_drop=config_gold.GOLD_DROP_COLS)
    df = drop_duplicates(df, keep="first")
    df = drop_missing_target_rows(df, config_gold.TARGET_COLUMN)

    # for testing before implementing valdiation lets say result was success
    from types import SimpleNamespace

    result = SimpleNamespace(success=True)
    # === STAGE ?: SAVE DATAFRAME BASED ON RESULT ===
    save_successful = save_dataframe_based_on_validation(
        result=result,
        df=df,
        file_name="test_gold",  # at the end will replace it with Path(file_name).stem
        success_dir=config_gold.GOLD_PROCESSED_DIR,
        failure_dir=config_gold.GOLD_QUARANTINE_DIR,
    )

    # The pipeline's true success depends on BOTH validation AND the save operation
    pipeline_successful = result.success and save_successful
    # === STAGE ?: LOG FINAL STATUS ===
    if pipeline_successful:
        logger.info(
            f"--- GOLD Preprocessing, Feature Engineering & Validation: PASSED for {file_name} ---"
        )
    else:
        logger.warning(
            f"--- GOLD Preprocessing, Feature Engineering & Validation: FAILED for {file_name} ---"
        )

    return pipeline_successful


def main():
    setup_logging_from_yaml(
        log_path=config_logging.GOLD_PIPELINE_LOGS_PATH,
        default_level=logging.DEBUG,
        default_yaml_path=config_logging.LOGGING_YAML,
    )

    parser = argparse.ArgumentParser(description="Run The Gold Engineering Pipeline.")
    parser.add_argument(
        "file_name",
        type=str,
        help="Enter the file name to load from silver processed folder",
    )

    args = parser.parse_args()
    file_name = args.file_name
    input_filepath = Path(config_silver.SILVER_PROCESSED_DIR / file_name)

    pipeline_success = gold_engineering_pipeline(input_filepath=input_filepath)

    # Exit with a status code that an orchestrator like Airflow can interpret
    if not pipeline_success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
