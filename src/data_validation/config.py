import logging
from pathlib import Path
from utils import setup_logger

# Resolve the absolute path of this config file
current_abs_path = Path(__file__).resolve()

# Project root is three levels up from this file
PROJECT_ROOT = current_abs_path.parent.parent.parent

# Setup logger for this module
logger = logging.getLogger(__name__)
CONFIG_LOGS = PROJECT_ROOT / "src" / "data_validation" / "logs" / "Config_paths.log"
setup_logger(verbose=True, log_file=CONFIG_LOGS, mode="w")

# You can configure logging handlers here or configure it in your main script
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# Debug logs (replace print statements)
logger.debug(f"Current file absolute path: {current_abs_path}")
logger.debug(f"Project root directory: {PROJECT_ROOT}")

# Configuration for variable parameters

# Great Expectations project root directory
GE_ROOT_DIR = PROJECT_ROOT / "src" / "data_validation" / "great_expectations"
logger.debug(f"Great Expectations root directory: {GE_ROOT_DIR}")

# Data source and asset names
SOURCE_NAME = "flights"
ASSET_NAME = "flights_data"

# Data directory and batch info
DATA_BASE_DIR = PROJECT_ROOT / "data"
BATCH_NAME = "flights_main"
BATCH_PATH = "flights.csv"

# Expectation suite and validation definition names
SUITE_NAME = "flights_expectations_suite"
VALIDATION_DEFINITION_NAME = "flights_validation_definition"

# Batch parameters for validation run
BATCH_PARAMETERS = {"path": BATCH_PATH}

# Location to store logs
SUITE_LOGS = (
    PROJECT_ROOT / "src" / "data_validation" / "logs" / "run_expectations_suite.log"
)
VAL_DEF_LOGS = PROJECT_ROOT / "src" / "data_validation" / "logs" / "val_definition.log"
CHECKPOINTS_LOGS = (
    PROJECT_ROOT / "src" / "data_validation" / "logs" / "run_validation_checkpoints.log"
)

# Names for checkpoints
CHECKPOINT_NAME = "data_validation_checkpoint"
