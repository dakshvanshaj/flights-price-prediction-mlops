# src/data_split/split_data.py
import pandas as pd
import logging

# Import project-specific modules
from shared import config
from shared.utils import setup_logging_from_yaml
from data_ingestion.data_loader import load_data

# Create a logger object for this module
logger = logging.getLogger(__name__)


def split_data_chronologically():
    """
    Reads the raw flight data, sorts it by date, and splits it chronologically
    into training, validation, and test sets.
    """
    logger.info("--- Starting Data Splitting Process ---")

    # --- 1. Ensure Output Directory Exists ---
    logger.info(f"Ensuring output directory exists at '{config.SPLIT_DATA_DIR}'...")
    config.SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory is ready.")

    # --- 2. Load and Sort the Data ---
    logger.info(f"Loading data from '{config.RAW_CSV_PATH}'...")
    try:
        df = load_data(config.RAW_CSV_PATH)
        if df is None:
            raise FileNotFoundError  # load_data returns None on error
    except FileNotFoundError:
        logger.error(
            f"Failed to load data from '{config.RAW_CSV_PATH}'. Aborting process."
        )
        return

    logger.info(f"Loaded {len(df)} rows successfully.")

    # Convert 'date' column to datetime, sort, and handle any conversion errors
    logger.info("Converting 'date' column to datetime and sorting chronologically...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.dropna(subset=["date"], inplace=True)  # Drop rows where date conversion failed
    df.sort_values("date", inplace=True)

    # resets index to start from 0 instead of jumbled indexes(if there)
    df.reset_index(drop=True, inplace=True)
    logger.info("Data sorted and cleaned.")

    # --- 3. Perform Chronological Splits ---
    total_rows = len(df)
    train_end_index = int(total_rows * config.TRAIN_SET_SIZE)
    validation_end_index = train_end_index + int(total_rows * config.VAL_SET_SIZE)

    train_df = df.iloc[:train_end_index]
    validation_df = df.iloc[train_end_index:validation_end_index]
    test_df = df.iloc[validation_end_index:]

    logger.info("Data split into sets:")
    logger.info(f"  - Training Set:   {len(train_df)} rows")
    logger.info(f"  - Validation Set: {len(validation_df)} rows")
    logger.info(f"  - Test Set:       {len(test_df)} rows")

    # --- 4. Save the Datasets ---
    train_path = config.SPLIT_DATA_DIR / "train.csv"
    validation_path = config.SPLIT_DATA_DIR / "validation.csv"
    test_path = config.SPLIT_DATA_DIR / "test.csv"

    logger.info(f"Saving training set to '{train_path}'...")
    train_df.to_csv(train_path, index=False)

    logger.info(f"Saving validation set to '{validation_path}'...")
    validation_df.to_csv(validation_path, index=False)

    logger.info(f"Saving test set to '{test_path}'...")
    test_df.to_csv(test_path, index=False)

    logger.info("--- Data Splitting Process Completed Successfully! ---")


if __name__ == "__main__":
    # Setup logging from the central YAML file
    setup_logging_from_yaml(log_path=config.SPLIT_DATA_LOGS_PATH)
    split_data_chronologically()
