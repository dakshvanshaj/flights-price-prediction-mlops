from data_ingestion.data_loader import load_data
from shared import config
from shared.utils import setup_logging_from_yaml
from pathlib import Path
import argparse
import sys
import logging

logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: str,
) -> bool:
    logger.info(f"Validating file path: {input_filepath}")
    if not input_filepath.exists():
        logger.error(f"File not found at {input_filepath}. Aborting pipeline.")
        return False

    file_name = Path(input_filepath).name
    logger.info(f"--- Starting Gold Engineering Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/5: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        logger.error(f"Failed to load data from {input_filepath}. Aborting pipeline.")
        return False
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: OUTLIER DETECTION AND HANDLING ==
    logger.info("=" * 25 + " STAGE 2/5: OUTLIER DETECTION AND HANDLING " + "=" * 25)


if __name__ == "__main__":
    setup_logging_from_yaml(
        log_path=config.GOLD_PIPELINE_LOGS_PATH,
        default_level=logging.DEBUG,
        default_yaml_path=config.LOGGING_YAML,
    )

    parser = argparse.ArgumentParser(description="Run The Gold Engineering Pipeline.")
    parser.add_argument(
        "file_name",
        type=str,
        help="Enter the file name to load from silver processed folder",
    )

    args = parser.parse_args()
    file_name = args.file_name
    input_filepath = Path(config.SILVER_PROCESSED_DIR / file_name)

    pipeline_success = gold_engineering_pipeline(input_filepath=str(input_filepath))

    # Exit with a status code that an orchestrator like Airflow can interpret
    if not pipeline_success:
        sys.exit(1)
    sys.exit(0)
