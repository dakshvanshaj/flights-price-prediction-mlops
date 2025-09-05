from pathlib import Path

# Defines the absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Top-level directories derived from the project root.
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
GE_ROOT_DIR = SRC_DIR / "data_validation" / "great_expectations"
REPORTS_DIR = PROJECT_ROOT / "reports"


# Raw Data Directory
RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = RAW_DATA_DIR / "train_validation_test"

# Bronze, Silver, and Gold Data Directories
BRONZE_DATA_DIR = DATA_DIR / "bronze_data"
SILVER_DATA_DIR = DATA_DIR / "silver_data"
GOLD_DATA_DIR = DATA_DIR / "gold_engineered_data"
