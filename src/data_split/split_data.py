# src/data_split/split_data.py
import pandas as pd
import logging
from pathlib import Path

# --- FIXED: Import the config module itself, not the variables from it ---
# This is the key to making monkeypatch work reliably.
from shared import config
from shared.utils import setup_logging_from_yaml
from data_ingestion.data_loader import load_data

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
    # --- FIXED: Access variables through the config module ---
    logger.info(
        f"Ensuring output directories exist at '{config.INITIAL_DATA_SPLITS}'..."
    )
    config.INITIAL_DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    config.DRIFT_SIMULATION_DIR.mkdir(exist_ok=True)
    logger.info("Directories are ready.")

    # --- 2. Load and Sort the Data ---
    logger.info(f"Loading data from '{config.INPUT_CSV_PATH}'...")
    df = load_data(str(config.INPUT_CSV_PATH))

    if df is None:
        logger.error(
            f"Failed to load data from '{config.INPUT_CSV_PATH}'. Aborting process."
        )
        return

    logger.info(f"Loaded {len(df)} rows successfully.")

    logger.info("Converting 'date' column to datetime and sorting chronologically...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.dropna(subset=["date"], inplace=True)
    logger.info("Data sorted.")

    # --- 3. Perform Chronological Splits ---
    total_rows = len(df)
    dev_end_index = int(total_rows * config.DEV_SET_SIZE)
    eval_end_index = dev_end_index + int(total_rows * config.EVAL_SET_SIZE)

    dev_df = df.iloc[:dev_end_index]
    eval_df = df.iloc[dev_end_index:eval_end_index]
    drift_df = df.iloc[eval_end_index:].copy()  # Using .copy() to avoid warnings

    logger.info("Data split into sets:")
    logger.info(f"  - Development Set: {len(dev_df)} rows")
    logger.info(f"  - Evaluation Hold-out Set: {len(eval_df)} rows")
    logger.info(f"  - Drift Simulation Set: {len(drift_df)} rows")

    # --- 4. Save the Main Datasets ---
    dev_path = config.INITIAL_DATA_SPLITS / "development_data.csv"
    eval_path = config.INITIAL_DATA_SPLITS / "evaluation_holdout.csv"

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

    drift_df["year_month"] = drift_df["date"].dt.to_period("M")
    monthly_groups = drift_df.groupby("year_month")

    for period, group_df in monthly_groups:
        file_name = f"flights_{period}.csv"
        chunk_path = config.DRIFT_SIMULATION_DIR / file_name

        data_to_save = group_df.copy()
        data_to_save.drop(columns=["year_month"], inplace=True)

        logger.info(
            f"  - Saving month {period} ({len(data_to_save)} rows) to '{chunk_path}'..."
        )
        data_to_save.to_csv(chunk_path, index=False)

    logger.info("--- Data Splitting Process Completed Successfully! ---")


if __name__ == "__main__":
    setup_logging_from_yaml(
        log_path=config.SPLIT_DATA_LOGS_PATH,
        default_level="DEBUG",
        default_yaml_path="logging.yaml",
    )
    split_data()
