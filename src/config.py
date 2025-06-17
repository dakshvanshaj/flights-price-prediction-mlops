# src/config.py
from pathlib import Path

# --- Absolute Path Configuration ---
# Defines the absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Path Definitions ---
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Location of the original, full dataset used by the splitting script.
INPUT_CSV_PATH = DATA_DIR / "flights.csv"

# Raw Data Flow Directories
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PENDING_DIR = RAW_DATA_DIR / "pending"
RAW_PROCESSED_DIR = RAW_DATA_DIR / "processed"
RAW_QUARANTINE_DIR = RAW_DATA_DIR / "quarantine"

# Prepared and Cleaned Data Directories
PREPARED_DATA_DIR = DATA_DIR / "prepared_data"
CLEANED_DATA_DIR = PREPARED_DATA_DIR / "cleaned"


# --- Log File Paths ---
SPLIT_DATA_LOGS_PATH = LOGS_DIR / "split_data.log"
BRONZE_PIPELINE_LOGS_PATH = LOGS_DIR / "bronze_pipeline.log"
SILVER_PIPELINE_LOGS_PATH = LOGS_DIR / "silver_pipeline.log"


# --- Data Splitting Configuration ---
DEV_SET_SIZE = 0.70
EVAL_SET_SIZE = 0.15


# --- Great Expectations Configuration ---
GE_ROOT_DIR = PROJECT_ROOT / "src" / "data_validation" / "great_expectations"

# --- Bronze Gate Configuration ---
# Specific names for the Bronze pipeline components
RAW_DATA_SOURCE_NAME = "raw_flight_data_source"
BRONZE_RAW_ASSET_NAME = "raw_flights_asset"
BRONZE_SUITE_NAME = "bronze_raw_flights_suite"
BRONZE_BATCH_DEFINITION_NAME = "bronze_batch_definition"
BRONZE_VALIDATION_DEFINITION_NAME = "bronze_validation_definition"
BRONZE_CHECKPOINT_NAME = "bronze_checkpoint"


# --- Silver Gate Configuration (Placeholders for later) ---
# We will uncomment and use these when we build the Silver pipeline
# CLEANED_DATA_SOURCE_NAME = "cleaned_flight_data_source"
# SILVER_CLEANED_ASSET_NAME = "silver_cleaned_flights_asset"
# SILVER_SUITE_NAME = "silver_cleaned_flights_suite"
# SILVER_CHECKPOINT_NAME = "silver_checkpoint"
