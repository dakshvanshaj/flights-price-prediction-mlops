# src/config.py
from pathlib import Path

# --- Absolute Path Configuration ---
# Defines the absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Path Definitions ---
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Raw Data Flow Directories
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PENDING_DIR = RAW_DATA_DIR / "pending"
RAW_PROCESSED_DIR = RAW_DATA_DIR / "processed"
RAW_QUARANTINE_DIR = RAW_DATA_DIR / "quarantine"

# Prepared and Cleaned Data Directories
PREPARED_DATA_DIR = DATA_DIR / "prepared_data"
CLEANED_DATA_DIR = PREPARED_DATA_DIR / "cleaned"

# ---------------------SPLIT_DATA------------------------
INPUT_CSV_PATH = DATA_DIR / "flights.csv"
DEV_SET_SIZE = 0.70
EVAL_SET_SIZE = 0.15


# --- Great Expectations Configuration ---
GE_ROOT_DIR = PROJECT_ROOT / "src" / "data_validation" / "great_expectations"

# gx Datasource related configurations
# ---------------------BRONZE PIPELINE-------------------
# RAW_PENDING_DIR = RAW_DATA_DIR / "pending"
RAW_DATA_SOURCE_NAME = "raw_flight_data_source"
RAW_DATA_SOURCE = RAW_PENDING_DIR

# gx asset realted config
# ---------------------BRONZE PIPELINE-------------------
RAW_ASSET_NAME = "raw_flights_csv_files"

# batch definition related config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_BATCH_DEFINITION_NAME = "bronze_batch_definition"


# gx suite related config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_SUITE_NAME = "bronze_raw_flights_suite"

# gx validation definition config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_VALIDATION_DEFINITION_NAME = "bronze_validation_definition"

# gx checkpoint definition config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_CHECKPOINT_NAME = "bronze_checkpoint"


# ---------------------LOG FILES PATHS-------------------
SPLIT_DATA_LOGS_PATH = LOGS_DIR / "split_data.log"
BRONZE_PIPELINE_LOGS_PATH = LOGS_DIR / "bronze_pipeline.log"
SILVER_PIPELINE_LOGS_PATH = LOGS_DIR / "silver_pipeline.log"


# ---------------------- Silver Layer Preprocessing Configurations-------------------

# column rename mapping
COLUMN_RENAME_MAPPING = {"from": "from_location", "to": "to_location"}
ERRONEOUS_DUPE_SUBSET = [
    "user_code",
    "from_location",
    "to_location",
    "date",
    "time",
    "agency",
]
