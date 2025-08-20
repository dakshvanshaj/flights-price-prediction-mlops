from shared.config.core_paths import LOGS_DIR, PROJECT_ROOT

LOGGING_YAML = PROJECT_ROOT / "logging.yaml"

BRONZE_PIPELINE_LOGS_PATH = LOGS_DIR / "bronze_pipeline.log"
SILVER_PIPELINE_LOGS_PATH = LOGS_DIR / "silver_pipeline.log"
SPLIT_DATA_LOGS_PATH = LOGS_DIR / "split_data.log"
GOLD_PIPELINE_LOGS_PATH = LOGS_DIR / "gold_pipeline.log"
TRAINING_PIPELINE_LOGS_PATH = LOGS_DIR / "training_pipeline.log"
