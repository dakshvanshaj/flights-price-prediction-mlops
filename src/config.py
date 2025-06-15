from pathlib import Path

# --- Absolute Path Configuration ---
# Defines the absolute path to the project root directory, making all other
# paths independent of where the script is run from.
# Assumes this config file is located at root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
# --- Path Definitions ---
# Centralized directory for all log files.
LOGS_DIR = PROJECT_ROOT / "logs"

# Data-related paths.
DATA_DIR = PROJECT_ROOT / "data"

INPUT_CSV_PATH = DATA_DIR / "flights.csv"

PREPARED_DATA_DIR = DATA_DIR / "prepared_data"
DRIFT_SIMULATION_DIR = PREPARED_DATA_DIR / "drift_simulation_data"

# --- Log File Paths ---
# Specific log file paths, all pointing to the central LOGS_DIR.
SPLIT_DATA_LOGS_PATH = LOGS_DIR / "split_data.log"
VALIDATION_PIPELINE_LOGS_PATH = LOGS_DIR / "data_validation_pipeline.log"


# --- Data Splitting Configuration ---
DEV_SET_SIZE = 0.70  # 70% for initial model development
EVAL_SET_SIZE = 0.15  # 15% for the initial hold-out evaluation
DRIFT_SIMULATION_FILES = 4  # Number of files for drift simulation


# --- Great Expectations Configuration ---
# GE project root directory.
GE_ROOT_DIR = PROJECT_ROOT / "src" / "data_validation" / "great_expectations"
GE_ROOT_DIR
# Data source, asset, and batch names for GE.
SOURCE_NAME = "flights"
ASSET_NAME = "flights_data"
BATCH_NAME = "flights_main"

# Expectation suite and checkpoint names.
SUITE_NAME = "flights_expectations_suite"
CHECKPOINT_NAME = "data_validation_checkpoint"
VALIDATION_DEFINITION_NAME = "flights_validation_definition"
