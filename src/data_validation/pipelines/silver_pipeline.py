import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, Any

# Local application imports
from data_ingestion.data_loader import load_data
from data_preprocessing.silver_preprocessing import (
    rename_specific_columns,
    standardize_column_format,
    optimize_data_types,
    handle_erroneous_duplicates,
    MissingValueHandler,
)
from config import (
    COLUMN_RENAME_MAPPING,
    PROJECT_ROOT,
    SILVER_PIPELINE_LOGS_PATH,
    ERRONEOUS_DUPE_SUBSET,
    SAVED_MV_IMPUTER_PATH,
    COLUMN_IMPUTATION_RULES,
)
from shared.utils import setup_logger

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def run_silver_pipeline(
    input_filepath: str,
    imputer_path: str,
    train_mode: bool = False,
    column_strategies: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Orchestrates the Silver layer data processing pipeline.

    This function can run in two modes:
    1.  train_mode=True: It will learn imputation values from the data, apply them,
        and save the learned imputer to `imputer_path`.
    2.  train_mode=False: It will load a pre-existing imputer from `imputer_path`
        and apply it to the new data.

    Args:
        input_filepath (str): Path to the processed bronze data file.
        imputer_path (str): Path to save or load the imputer JSON file.
        train_mode (bool): If True, fits and saves a new imputer.
        column_strategies (dict, optional): Column-specific imputation strategies.

    Returns:
        Optional[pd.DataFrame]: The cleaned DataFrame, or None if an error occurs.
    """
    file_name = Path(input_filepath).name
    logger.info(
        f"--- Starting Silver Pipeline for: {file_name} (Train Mode: {train_mode}) ---"
    )

    # --- Stage 1: Data Ingestion ---
    logger.info("[Stage 1/4] Ingesting data...")
    df = load_data(file_path=input_filepath)
    if df is None:
        logger.error("Data loading failed. Aborting pipeline.")
        return None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # --- Stage 2: Standard Cleaning ---
    logger.info("[Stage 2/4] Applying standard cleaning and formatting...")
    df = rename_specific_columns(df, rename_mapping=COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    df = optimize_data_types(df, date_cols=["date"])
    df = handle_erroneous_duplicates(df=df, subset_cols=ERRONEOUS_DUPE_SUBSET)
    logger.info("Standard cleaning complete.")

    # --- Stage 3: Handle Missing Values ---
    logger.info("[Stage 3/4] Handling missing values...")
    try:
        if train_mode:
            # In training mode, create, fit, and save a new imputer
            logger.info("Training mode: Creating and fitting a new imputer...")
            handler = MissingValueHandler(column_strategies=column_strategies)
            handler.fit(df)
            handler.save(imputer_path)
        else:
            # In inference mode, load the pre-trained imputer
            logger.info(f"Inference mode: Loading imputer from {imputer_path}...")
            if not Path(imputer_path).exists():
                logger.error(
                    f"Imputer file not found at '{imputer_path}'. Cannot proceed."
                )
                return None
            handler = MissingValueHandler.load(imputer_path)

        # Apply the imputation to the dataframe
        df = handler.transform(df)
        logger.info("Missing value handling complete.")
    except Exception as e:
        logger.error(
            f"An error occurred during missing value handling: {e}", exc_info=True
        )
        return None

    # --- Stage 4: Data Validation (Placeholder) ---
    logger.info("[Stage 4/4] Validating processed data...")
    # TODO: Add call to Silver Great Expectations checkpoint here.
    logger.info("Data validation complete (placeholder).")

    logger.info(f"--- Silver Pipeline for: {file_name} completed successfully ---")

    return df


if __name__ == "__main__":
    # This block is for directly running and testing the pipeline.
    # It runs in 'train_mode' to generate the imputer file from a test file.
    setup_logger(verbose=True, log_file=SILVER_PIPELINE_LOGS_PATH, mode="w")

    # Define the path to a sample input file to train the imputer
    test_file_path = PROJECT_ROOT / "data" / "raw" / "processed" / "flights_2022-02.csv"

    # Define custom strategies for imputation (optional)

    # Execute the pipeline in training mode
    logger.info(f"Running pipeline in TRAIN mode on test file: {test_file_path}")
    cleaned_df = run_silver_pipeline(
        input_filepath=str(test_file_path),
        imputer_path=str(SAVED_MV_IMPUTER_PATH),
        train_mode=True,
        # column_strategies=COLUMN_IMPUTATION_RULES,
    )

    if cleaned_df is not None:
        logger.info("Pipeline executed successfully. Cleaned DataFrame head:")
        print(cleaned_df.head())
        logger.info(f"Imputer state saved to '{SAVED_MV_IMPUTER_PATH}'")

    else:
        logger.error(
            f"Pipeline execution failed. Check logs at '{SILVER_PIPELINE_LOGS_PATH}'."
        )
