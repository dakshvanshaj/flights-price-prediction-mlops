# src/data_preparation/split_data.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from config import (
    INPUT_CSV_PATH,
    PREPARED_DATA_DIR,
    DRIFT_SIMULATION_DIR,
    DEV_SET_SIZE,
    EVAL_SET_SIZE,
    DRIFT_SIMULATION_FILES,
    SPLIT_DATA_LOGS_PATH,  # Assuming you add this to your config
)
from shared.utils import setup_logger

# Create a logger object for this module
logger = logging.getLogger(__name__)


def split_data():
    """
    Reads the raw flight data, sorts it by date, and splits it into three sets:
    1.  development_data.csv: For initial training and validation.
    2.  evaluation_holdout.csv: For final evaluation of the initial model.
    3.  A set of smaller files for simulating future data arrivals and drift.
    """
    logger.info("--- Starting Data Splitting Process ---")

    # --- 1. Create Output Directories ---
    logger.info(f"Ensuring output directories exist at '{PREPARED_DATA_DIR}'...")
    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_SIMULATION_DIR.mkdir(exist_ok=True)
    logger.info("Directories are ready.")

    # --- 2. Load and Sort the Data ---
    logger.info(f"Loading data from '{INPUT_CSV_PATH}'...")
    if not INPUT_CSV_PATH.exists():
        logger.error(f"Input file not found at '{INPUT_CSV_PATH}'. Aborting process.")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    logger.info(f"Loaded {len(df)} rows successfully.")

    logger.info("Converting 'date' column to datetime and sorting chronologically...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.dropna(subset=["date"], inplace=True)  # Drop rows where date conversion failed
    logger.info("Data sorted.")

    # --- 3. Perform Chronological Splits ---
    total_rows = len(df)
    dev_end_index = int(total_rows * DEV_SET_SIZE)
    eval_end_index = dev_end_index + int(total_rows * EVAL_SET_SIZE)

    # Split the DataFrame
    dev_df = df.iloc[:dev_end_index]
    eval_df = df.iloc[dev_end_index:eval_end_index]
    drift_df = df.iloc[eval_end_index:]

    logger.info("Data split into sets:")
    logger.info(f"  - Development Set: {len(dev_df)} rows")
    logger.info(f"  - Evaluation Hold-out Set: {len(eval_df)} rows")
    logger.info(f"  - Drift Simulation Set: {len(drift_df)} rows")

    # --- 4. Save the Main Datasets ---
    dev_path = PREPARED_DATA_DIR / "development_data.csv"
    eval_path = PREPARED_DATA_DIR / "evaluation_holdout.csv"

    logger.info(f"Saving development set to '{dev_path}'...")
    dev_df.to_csv(dev_path, index=False)
    logger.info(f"Saving evaluation set to '{eval_path}'...")
    eval_df.to_csv(eval_path, index=False)
    logger.info("Main datasets saved.")

    # --- 5. Create and Save Drift Simulation Files ---
    logger.info(
        f"Splitting drift simulation data into {DRIFT_SIMULATION_FILES} files..."
    )

    # Use numpy.array_split for a clean and readable split.
    drift_chunks = np.array_split(drift_df, DRIFT_SIMULATION_FILES)

    # Enumerate through the chunks to save them with a 1-based index.
    for i, chunk_df in enumerate(drift_chunks, 1):
        chunk_path = DRIFT_SIMULATION_DIR / f"future_data_chunk_{i}.csv"
        logger.info(f"  - Saving chunk {i} ({len(chunk_df)} rows) to '{chunk_path}'...")
        chunk_df.to_csv(chunk_path, index=False)

    logger.info("--- Data Splitting Process Completed Successfully! ---")


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It sets up the logger for this specific run.
    setup_logger(verbose=True, log_file=SPLIT_DATA_LOGS_PATH, mode="w")
    split_data()
