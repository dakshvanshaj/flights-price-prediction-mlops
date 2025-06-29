# src/config.py
from pathlib import Path

# --- Absolute Path Configuration ---
# Defines the absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
# ----------------------SILVER PIPELINE-------------------
SILVER_DATA_SOURCE_NAME = "pandas_temp_datasource"

# gx asset realted config
# ---------------------BRONZE PIPELINE-------------------
RAW_ASSET_NAME = "raw_flights_csv_files"
# ----------------------SILVER PIPELINE-------------------
SILVER_ASSET_NAME = "silver_in_memory_asset"


# batch definition related config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_BATCH_DEFINITION_NAME = "bronze_batch_definition"
# ----------------------SILVER PIPELINE-------------------
SILVER_BATCH_DEFINITION_NAME = "silver_batch_definition"

# gx suite related config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_SUITE_NAME = "bronze_raw_flights_suite"
# ----------------------SILVER PIPELINE-------------------
SILVER_SUITE_NAME = "silver_df_flights_suite"

# gx validation definition config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_VALIDATION_DEFINITION_NAME = "bronze_validation_definition"
# ----------------------SILVER PIPELINE-------------------
SILVER_VALIDATION_DEFINITION_NAME = "silver_validation_definition"

# gx checkpoint definition config
# ---------------------BRONZE PIPELINE-------------------
BRONZE_CHECKPOINT_NAME = "bronze_checkpoint"
SILVER_CHECKPOINT_NAME = "silver_checkpoint"

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

# missing value imputer path
SAVED_MV_IMPUTER_PATH = PROJECT_ROOT / "models" / "missing_value_imputer.json"
COLUMN_IMPUTATION_RULES = {
    "from_location": "most_frequent",
    "to_location": "most_frequent",
    "flight_type": "most_frequent",
    "price": "median",
    "time": "median",
    "distance": "median",
    "agency": "most_frequent",
}

ID_COLS_TO_EXCLUDE_FROM_IMPUTATION = [
    "travel_code",
    "user_code",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
]

# ---------------------- Silver Layer Expectations Configurations-------------------
# The exact column order and names we expect after all Silver processing.
SILVER_EXPECTED_COLS_ORDER = [
    "travel_code",
    "user_code",
    "from_location",
    "to_location",
    "flight_type",
    "price",
    "time",
    "distance",
    "agency",
    "date",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
]

# The exact data types we expect after optimization.
SILVER_EXPECTED_COLUMN_TYPES = {
    "travel_code": "int32",
    "user_code": "int16",
    "from_location": "category",
    "to_location": "category",
    "flight_type": "category",
    "price": "float32",
    "time": "float32",
    "distance": "float32",
    "agency": "category",
    "date": "datetime64[ns]",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "day_of_week": "int8",
    "day_of_year": "int16",
    "week_of_year": "int32",
}

# The columns that should have no missing values after imputation.
SILVER_REQUIRED_NON_NULL_COLS = [
    "from_location",
    "to_location",
    "flight_type",
    "price",
    "time",
    "distance",
    "agency",
    "date",
]
