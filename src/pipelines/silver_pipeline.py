import pandas as pd
from pathlib import Path
import logging
import argparse
import sys
from typing import Dict, Optional, Any, List

# --- Local Application Imports ---
from data_validation.expectations.silver_expectations import build_silver_expectations
from data_validation.ge_components import run_checkpoint_on_dataframe
from data_ingestion.data_loader import load_data
from data_preprocessing.silver_preprocessing import (
    rename_specific_columns,
    standardize_column_format,
    optimize_data_types,
    create_date_features,
    handle_erroneous_duplicates,
    MissingValueHandler,
    sort_data_by_date,
    enforce_column_order,  # Import the new function
    ImputerNotFittedError,
    ImputerLoadError,
)
from shared.config import (
    COLUMN_RENAME_MAPPING,
    ERRONEOUS_DUPE_SUBSET,
    SAVED_MV_IMPUTER_PATH,
    COLUMN_IMPUTATION_RULES,
    ID_COLS_TO_EXCLUDE_FROM_IMPUTATION,
    GE_ROOT_DIR,
    SILVER_EXPECTED_COLS_ORDER,
    SILVER_EXPECTED_COLUMN_TYPES,
    SILVER_REQUIRED_NON_NULL_COLS,
    SILVER_PIPELINE_LOGS_PATH,
    RAW_PROCESSED_DIR,
    SILVER_PROCESSED_DIR,
    SILVER_QUARANTINE_DIR,
    SILVER_DATA_SOURCE_NAME,
    SILVER_ASSET_NAME,
    SILVER_BATCH_DEFINITION_NAME,
    SILVER_SUITE_NAME,
    SILVER_VALIDATION_DEFINITION_NAME,
    SILVER_CHECKPOINT_NAME,
)
from shared.utils import setup_logging_from_yaml

logger = logging.getLogger(__name__)


def run_silver_pipeline(
    input_filepath: str,
    imputer_path: str,
    train_mode: bool = False,
    column_strategies: Optional[Dict[str, Any]] = None,
    exclude_cols_imputation: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Orchestrates the full Silver layer data processing and validation pipeline.
    """
    file_name = Path(input_filepath).name
    logger.info(
        f"--- Starting Silver Pipeline for: {file_name} (Train Mode: {train_mode}) ---"
    )

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/6: DATA INGESTION " + "=" * 25)
    df = load_data(file_path=input_filepath)
    if df is None:
        return None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: PREPROCESSING & CLEANING ===
    logger.info("=" * 25 + " STAGE 2/6: PREPROCESSING & CLEANING " + "=" * 25)
    df = rename_specific_columns(df, rename_mapping=COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    df = optimize_data_types(df, date_cols=["date"])
    df = sort_data_by_date(df, date_column="date")
    df = handle_erroneous_duplicates(df=df, subset_cols=ERRONEOUS_DUPE_SUBSET)
    logger.info("Standardization, cleaning, and sorting complete.")

    # === STAGE 3: FEATURE ENGINEERING ===
    logger.info("=" * 25 + " STAGE 3/6: FEATURE ENGINEERING " + "=" * 25)
    df = create_date_features(df, date_column="date")
    logger.info("Date part extraction complete.")

    # === STAGE 4: MISSING VALUE IMPUTATION ===
    logger.info("=" * 25 + " STAGE 4/6: MISSING VALUE IMPUTATION " + "=" * 25)
    try:
        if train_mode:
            logger.info("Training mode: Creating and fitting a new imputer...")
            handler = MissingValueHandler(
                column_strategies=column_strategies,
                exclude_columns=exclude_cols_imputation,
            )
            handler.fit(df)
            handler.save(imputer_path)
        else:
            logger.info(f"Inference mode: Loading imputer from {imputer_path}...")
            handler = MissingValueHandler.load(imputer_path)

        df = handler.transform(df)
        logger.info("Missing value imputation complete.")
    except (ImputerNotFittedError, ImputerLoadError, FileNotFoundError) as e:
        logger.error(
            f"A critical error occurred during missing value handling: {e}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during imputation: {e}", exc_info=True
        )
        return None

    # === STAGE 5/6: ENFORCE SCHEMA ORDER ===
    logger.info("=" * 25 + " STAGE 5/6: ENFORCE SCHEMA ORDER " + "=" * 25)
    df = enforce_column_order(df, column_order=SILVER_EXPECTED_COLS_ORDER)

    # === FINAL STAGE 6/6: DATA VALIDATION (QUALITY GATE) ===
    logger.info("=" * 25 + " STAGE 6/6: FINAL VALIDATION " + "=" * 25)
    silver_expectations = build_silver_expectations(
        expected_cols_ordered=SILVER_EXPECTED_COLS_ORDER,
        expected_col_types=SILVER_EXPECTED_COLUMN_TYPES,
        non_null_cols=SILVER_REQUIRED_NON_NULL_COLS,
        unique_record_cols=ERRONEOUS_DUPE_SUBSET,
    )
    validation_result = run_checkpoint_on_dataframe(
        project_root_dir=GE_ROOT_DIR,
        datasource_name=SILVER_DATA_SOURCE_NAME,
        asset_name=SILVER_ASSET_NAME,
        batch_definition_name=SILVER_BATCH_DEFINITION_NAME,
        suite_name=SILVER_SUITE_NAME,
        validation_definition_name=SILVER_VALIDATION_DEFINITION_NAME,
        checkpoint_name=SILVER_CHECKPOINT_NAME,
        dataframe_to_validate=df,
        expectation_list=silver_expectations,
    )

    if validation_result is None:
        logger.critical("--- Silver Data Validation: FAILED TO RUN ---")
        logger.critical("The validation checkpoint itself failed to execute. Aborting.")
        return None

    if not validation_result.success:
        logger.critical("--- Silver Data Validation: FAILED ---")
        logger.critical("The cleaned data does not meet quality standards.")
        quarantine_path = SILVER_QUARANTINE_DIR / file_name
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(quarantine_path, index=False)
        logger.warning(f"Failed data saved to quarantine: {quarantine_path}")
        return None

    logger.info("--- Silver Data Validation: PASSED ---")
    logger.info("=" * 20 + " Silver Pipeline Completed Successfully " + "=" * 20)
    return df


def main():
    """
    Main entry point for running the Silver pipeline from the command line.
    """
    setup_logging_from_yaml(log_path=SILVER_PIPELINE_LOGS_PATH)

    parser = argparse.ArgumentParser(
        description="Run the Silver Data Processing Pipeline."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The name of the file in the 'processed' directory to clean (e.g., 'flights_2022-02.csv').",
    )
    parser.add_argument(
        "--train-mode",
        action="store_true",
        help="Run in training mode to create and save a new imputer. If not set, runs in inference mode.",
    )
    args = parser.parse_args()

    input_filepath = RAW_PROCESSED_DIR / args.input_file
    logger.info(
        f"Running pipeline for '{args.input_file}' | Train Mode: {args.train_mode}"
    )

    cleaned_df = run_silver_pipeline(
        input_filepath=str(input_filepath),
        imputer_path=str(SAVED_MV_IMPUTER_PATH),
        train_mode=args.train_mode,
        column_strategies=COLUMN_IMPUTATION_RULES,
        exclude_cols_imputation=ID_COLS_TO_EXCLUDE_FROM_IMPUTATION,
    )

    if cleaned_df is not None:
        logger.info("Pipeline executed successfully.")
        output_path = SILVER_PROCESSED_DIR / args.input_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to '{output_path}'")
        sys.exit(0)
    else:
        logger.error(
            f"Pipeline execution failed. Check logs at '{SILVER_PIPELINE_LOGS_PATH}'."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
