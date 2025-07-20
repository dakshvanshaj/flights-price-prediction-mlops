# src/shared/config.py
from pathlib import Path

# ==============================================================================
# --- 1. CORE PROJECT PATHS ---
# ==============================================================================
# Defines the absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Top-level directories derived from the project root.
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
GE_ROOT_DIR = PROJECT_ROOT / "src" / "data_validation" / "great_expectations"


# ==============================================================================
# --- 2. DATA FLOW & LOGGING PATHS ---
# ==============================================================================

# Raw Data  Directory
RAW_DATA_DIR = DATA_DIR / "raw"
# Split data directory
SPLIT_DATA_DIR = RAW_DATA_DIR / "train_validation_test"

# Bronze Preprocessed Data Flow Directories(BRONZE_PIPELINE)
RAW_DATA_SOURCE = SPLIT_DATA_DIR
BRONZE_DATA_DIR = DATA_DIR / "bronze_data"
BRONZE_PROCESSED_DIR = BRONZE_DATA_DIR / "processed"
BRONZE_QUARANTINE_DIR = BRONZE_DATA_DIR / "quarantine"

# Silver Preprocessed Data Flow Directories(SILVER_PIPELINE)
SILVER_DATA_DIR = DATA_DIR / "silver_data"
SILVER_PROCESSED_DIR = SILVER_DATA_DIR / "processed"
SILVER_QUARANTINE_DIR = SILVER_DATA_DIR / "quarantine"

# Logger config path
LOGGING_YAML = PROJECT_ROOT / "logging.yaml"


# Individual Log File Paths
BRONZE_PIPELINE_LOGS_PATH = LOGS_DIR / "bronze_pipeline.log"
SILVER_PIPELINE_LOGS_PATH = LOGS_DIR / "silver_pipeline.log"
SPLIT_DATA_LOGS_PATH = LOGS_DIR / "split_data.log"


# ==============================================================================
# --- 3. GREAT EXPECTATIONS CONFIGURATIONS ---
# ==============================================================================

# --- Bronze Pipeline GE Names ---
RAW_DATA_SOURCE_NAME = "raw_flight_data_source"
RAW_ASSET_NAME = "raw_flights_csv_files"
BRONZE_BATCH_DEFINITION_NAME = "bronze_batch_definition"
BRONZE_SUITE_NAME = "bronze_raw_flights_suite"
BRONZE_VALIDATION_DEFINITION_NAME = "bronze_validation_definition"
BRONZE_CHECKPOINT_NAME = "bronze_checkpoint"

# --- Silver Pipeline GE Names ---
SILVER_DATA_SOURCE_NAME = "pandas_temp_datasource"
SILVER_ASSET_NAME = "silver_in_memory_asset"
SILVER_BATCH_DEFINITION_NAME = "silver_batch_definition"
SILVER_SUITE_NAME = "silver_df_flights_suite"
SILVER_VALIDATION_DEFINITION_NAME = "silver_validation_definition"
SILVER_CHECKPOINT_NAME = "silver_checkpoint"


# ==============================================================================
# --- 4. DATA PREPARATION & PIPELINE CONFIGURATIONS ---
# ==============================================================================

# --- Data Splitting Config ---
RAW_CSV_PATH = RAW_DATA_DIR / "flights.csv"
TRAIN_SET_SIZE = 0.70
VAL_SET_SIZE = 0.15
# remaining is validation set size

# --- Silver Preprocessing Config ---
COLUMN_RENAME_MAPPING = {"from": "from_location", "to": "to_location"}
ERRONEOUS_DUPE_SUBSET = [
    "user_code",
    "from_location",
    "to_location",
    "date",
    "time",
    "agency",
]
SAVED_MV_IMPUTER_PATH = MODELS_DIR / "missing_value_imputer.json"
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

# --- Silver Expectations Config (Data Contracts) ---
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

SILVER_EXPECTED_COLUMN_TYPES = {
    "travel_code": "int32",
    "user_code": "int16",
    # Use the exact type string reported by Great Expectations for categorical columns.
    "from_location": "CategoricalDtypeType",
    "to_location": "CategoricalDtypeType",
    "flight_type": "CategoricalDtypeType",
    "agency": "CategoricalDtypeType",
    "price": "float32",
    "time": "float32",
    "distance": "float32",
    "date": "datetime64[ns]",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "day_of_week": "int8",
    "day_of_year": "int16",
    "week_of_year": "int32",
}

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
