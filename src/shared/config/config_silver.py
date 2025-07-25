from shared.config.core_paths import SILVER_DATA_DIR, LOGS_DIR, MODELS_DIR

# --- Silver Data Flow ---
SILVER_PROCESSED_DIR = SILVER_DATA_DIR / "processed"
SILVER_QUARANTINE_DIR = SILVER_DATA_DIR / "quarantine"

# --- Silver Logging ---
SILVER_PIPELINE_LOGS_PATH = LOGS_DIR / "silver_pipeline.log"

# --- Silver Pipeline GE Names ---
SILVER_DATA_SOURCE_NAME = "pandas_temp_datasource"
SILVER_ASSET_NAME = "silver_in_memory_asset"
SILVER_BATCH_DEFINITION_NAME = "silver_batch_definition"
SILVER_SUITE_NAME = "silver_df_flights_suite"
SILVER_VALIDATION_DEFINITION_NAME = "silver_validation_definition"
SILVER_CHECKPOINT_NAME = "silver_checkpoint"

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
