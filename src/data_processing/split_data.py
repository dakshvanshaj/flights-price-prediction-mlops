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
    SPLIT_DATA_LOGS_PATH,
)
from shared.utils import setup_logger

# Create a logger object for this module
logger = logging.getLogger(__name__)


def split_data():
    """
    Reads the raw flight data, sorts it chronologically, and splits it into:
    1. A development set for initial training and validation.
    2. A hold-out set for final evaluation of the initial model.
    3. A set of monthly files to simulate future data arrivals for drift detection
       and retraining.
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

    # --- 5. Create and Save Monthly Drift Simulation Files ---
    logger.info("Splitting drift simulation data into monthly files...")

    if drift_df.empty:
        logger.warning(
            "Drift simulation set is empty. No monthly files will be created."
        )
        logger.info("--- Data Splitting Process Completed Successfully! ---")
        return

    # Group the drift dataframe by year and month
    drift_df["year_month"] = drift_df["date"].dt.to_period("M")
    monthly_groups = drift_df.groupby("year_month")

    for period, group_df in monthly_groups:
        # Create a descriptive filename like 'flights_2022-10.csv'
        file_name = f"flights_{period}.csv"
        chunk_path = DRIFT_SIMULATION_DIR / file_name

        # Make a copy to avoid SettingWithCopyWarning when dropping the column
        data_to_save = group_df.copy()
        data_to_save.drop(columns=["year_month"], inplace=True)

        logger.info(
            f"  - Saving month {period} ({len(data_to_save)} rows) to '{chunk_path}'..."
        )
        data_to_save.to_csv(chunk_path, index=False)

    logger.info("--- Data Splitting Process Completed Successfully! ---")


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    setup_logger(verbose=True, log_file=SPLIT_DATA_LOGS_PATH, mode="w")
    split_data()
