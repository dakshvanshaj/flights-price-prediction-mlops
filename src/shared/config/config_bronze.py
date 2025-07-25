from shared.config.core_paths import BRONZE_DATA_DIR, SPLIT_DATA_DIR, LOGS_DIR

# --- Bronze Data Flow ---
RAW_DATA_SOURCE = SPLIT_DATA_DIR
BRONZE_PROCESSED_DIR = BRONZE_DATA_DIR / "processed"
BRONZE_QUARANTINE_DIR = BRONZE_DATA_DIR / "quarantine"

# --- Bronze Logging ---
BRONZE_PIPELINE_LOGS_PATH = LOGS_DIR / "bronze_pipeline.log"

# --- Bronze Pipeline GE Names ---
RAW_DATA_SOURCE_NAME = "raw_flight_data_source"
RAW_ASSET_NAME = "raw_flights_csv_files"
BRONZE_BATCH_DEFINITION_NAME = "bronze_batch_definition"
BRONZE_SUITE_NAME = "bronze_raw_flights_suite"
BRONZE_VALIDATION_DEFINITION_NAME = "bronze_validation_definition"
BRONZE_CHECKPOINT_NAME = "bronze_checkpoint"
