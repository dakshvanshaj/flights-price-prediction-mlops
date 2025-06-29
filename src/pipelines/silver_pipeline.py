import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, Any, List

# --- Local Application Imports ---
# Import the new silver expectations builder and the dataframe checkpoint runner
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
)

# Import all necessary configuration variables
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
    PROJECT_ROOT,
    SILVER_DATA_SOURCE_NAME,
    SILVER_ASSET_NAME,
    SILVER_BATCH_DEFINITION_NAME,
    SILVER_SUITE_NAME,
    SILVER_VALIDATION_DEFINITION_NAME,
    SILVER_CHECKPOINT_NAME,
)
from shared.utils import setup_logging_from_yaml

# Initialize a logger for this module
logger = logging.getLogger(__name__)


def run_silver_pipeline(
    input_filepath: str,
    imputer_path: str,
    train_mode: bool = False,
    column_strategies: Optional[Dict[str, Any]] = None,
    exclude_cols_imputation: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Orchestrates the Silver layer data processing pipeline, including a final
    validation step on the cleaned data.
    """
    file_name = Path(input_filepath).name
    logger.info(
        f"--- Starting Silver Pipeline for: {file_name} (Train Mode: {train_mode}) ---"
    )

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/5: DATA INGESTION " + "=" * 25)
    df = load_data(file_path=input_filepath)
    if df is None:
        return None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: PREPROCESSING & CLEANING ===
    logger.info("=" * 25 + " STAGE 2/5: PREPROCESSING & CLEANING " + "=" * 25)
    df = rename_specific_columns(df, rename_mapping=COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    df = optimize_data_types(df, date_cols=["date"])
    df = sort_data_by_date(df, date_column="date")
    df = handle_erroneous_duplicates(df=df, subset_cols=ERRONEOUS_DUPE_SUBSET)
    logger.info("Standardization, cleaning, and sorting complete.")

    # === STAGE 3: FEATURE ENGINEERING ===
    logger.info("=" * 25 + " STAGE 3/5: FEATURE ENGINEERING " + "=" * 25)
    df = create_date_features(df, date_column="date")
    logger.info("Date part extraction complete.")

    # === STAGE 4: MISSING VALUE IMPUTATION ===
    logger.info("=" * 25 + " STAGE 4/5: MISSING VALUE IMPUTATION " + "=" * 25)
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
            if not Path(imputer_path).exists():
                logger.error(f"Imputer file not found at '{imputer_path}'. Aborting.")
                return None
            handler = MissingValueHandler.load(imputer_path)

        df = handler.transform(df)
        logger.info("Missing value imputation complete.")
    except Exception as e:
        logger.error(
            f"An error occurred during missing value handling: {e}", exc_info=True
        )
        return None

    # === FINAL STAGE 5/5: DATA VALIDATION (QUALITY GATE) ===
    logger.info("=" * 25 + " STAGE 5/5: FINAL VALIDATION " + "=" * 25)

    # 1. Build the list of expectations using our function and central config
    silver_expectations = build_silver_expectations(
        expected_cols_ordered=SILVER_EXPECTED_COLS_ORDER,
        expected_col_types=SILVER_EXPECTED_COLUMN_TYPES,
        non_null_cols=SILVER_REQUIRED_NON_NULL_COLS,
        unique_record_cols=ERRONEOUS_DUPE_SUBSET,
    )

    # 2. Run the checkpoint on our cleaned DataFrame
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

    # 3. Act on the validation result
    if not validation_result.success:
        logger.critical("--- Silver Data Validation: FAILED ---")
        logger.critical(
            "The cleaned data does not meet quality standards. Aborting pipeline."
        )
        # In a real scenario, you might want to save the failed `df` to a quarantine area.
        return None

    logger.info("--- Silver Data Validation: PASSED ---")
    logger.info("=" * 20 + " Silver Pipeline Completed Successfully " + "=" * 20)

    return df


if __name__ == "__main__":
    # Setup logging from the YAML configuration file
    setup_logging_from_yaml(log_path=SILVER_PIPELINE_LOGS_PATH)

    # Define a test file to run the pipeline on
    test_file = (
        Path(PROJECT_ROOT) / "data" / "raw" / "processed" / "flights_2022-02.csv"
    )

    logger.info(f"Running pipeline in TRAIN mode on test file: {test_file}")

    # Execute the main pipeline function
    cleaned_df = run_silver_pipeline(
        input_filepath=str(test_file),
        imputer_path=str(SAVED_MV_IMPUTER_PATH),
        train_mode=True,
        column_strategies=COLUMN_IMPUTATION_RULES,
        exclude_cols_imputation=ID_COLS_TO_EXCLUDE_FROM_IMPUTATION,
    )

    if cleaned_df is not None:
        logger.info("Pipeline execution finished. Cleaned DataFrame head:")
        print(cleaned_df.head())
        logger.info(f"Imputer state saved to '{SAVED_MV_IMPUTER_PATH}'")
    else:
        logger.error(
            f"Pipeline execution failed. Check logs at '{SILVER_PIPELINE_LOGS_PATH}'."
        )
