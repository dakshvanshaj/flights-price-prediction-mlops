import pandas as pd
from pathlib import Path
import logging
from data_ingestion.data_loader import load_data
from data_preprocessing.silver_preprocessing import (
    rename_specific_columns,
    standardize_column_format,
    optimize_data_types,
    handle_erroneous_duplicates,
)

from config import (
    COLUMN_RENAME_MAPPING,
    PROJECT_ROOT,
    SILVER_PIPELINE_LOGS_PATH,
    ERRONEOUS_DUPE_SUBSET,
)
from shared.utils import setup_logger

logger = logging.getLogger(__name__)


def run_silver_pipeline(input_filepath: str) -> pd.DataFrame:
    """
    Orchestrates the Silver layer data processing pipeline.
    """
    file_name = Path(input_filepath).name
    logger.info(f"--- Starting Silver Pipeline for: {file_name} ---")

    # --- Stage 1: Data Ingestion ---
    logger.info("[Stage 1/3] Ingesting data...")
    df = load_data(file_path=input_filepath)
    if df is None:
        logger.error("Data loading failed. Aborting pipeline.")
        return None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # --- Stage 2: Data Preprocessing ---
    # This stage now includes all cleaning steps.
    logger.info("[Stage 2/3] Preprocessing and cleaning data...")
    df = rename_specific_columns(df, rename_mapping=COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    df = optimize_data_types(df, date_cols=["date"])
    df = handle_erroneous_duplicates(df=df, subset_cols=ERRONEOUS_DUPE_SUBSET)
    logger.info("Data preprocessing complete.")

    # --- Stage 3: Data Validation (Placeholder) ---
    # This stage will use Great Expectations to validate the final DataFrame.
    logger.info("[Stage 3/3] Validating processed data...")
    # TODO: Add call to Silver Great Expectations checkpoint here.
    logger.info("Data validation complete.")

    logger.info(f"--- Silver Pipeline for: {file_name} completed successfully ---")

    return df


if __name__ == "__main__":
    # Configure logging for the script run
    setup_logger(verbose=False, log_file=SILVER_PIPELINE_LOGS_PATH, mode="w")

    # Define the path to the input file for testing
    path = PROJECT_ROOT / r"data\raw\processed\flights_2022-02.csv"

    # Execute the pipeline
    run_silver_pipeline(str(path))
