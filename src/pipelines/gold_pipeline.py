from data_ingestion.data_loader import load_data
from shared.config import config_gold, config_logging, config_silver
from shared.utils import setup_logging_from_yaml, save_dataframe_based_on_validation
from pathlib import Path
import sys
import logging
from gold_data_preprocessing.data_cleaning import (
    drop_columns,
    drop_duplicates,
    drop_missing_target_rows,
)
from gold_data_preprocessing.missing_value_imputer import SimpleImputer
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: Path, imputer_to_apply: Optional[SimpleImputer] = None
) -> Tuple[bool, Optional[SimpleImputer]]:
    file_name = input_filepath.name
    logger.info(f"--- Starting Gold Engineering Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/?: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        return False, None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: DATA CLEANING ===
    logger.info("=" * 25 + " STAGE 2/?: DATA CLEANING " + "=" * 25)
    df = drop_columns(df, columns_to_drop=config_gold.GOLD_DROP_COLS)
    df = drop_duplicates(df, keep="first")
    df = drop_missing_target_rows(df, config_gold.TARGET_COLUMN)

    # === STAGE 3: IMPUTATION (TRAINING OR INFERENCE) ===
    logger.info("=" * 25 + " STAGE 3/?: IMPUTATION " + "=" * 25)
    fitted_imputer = None
    if imputer_to_apply:
        # --- INFERENCE MODE ---
        logger.info("Applying existing imputer...")
        df = imputer_to_apply.transform(df)
    else:
        # --- TRAINING MODE ---
        logger.info("Fitting new imputer...")
        imputer = SimpleImputer(strategy_dict=config_gold.IMPUTER_STRATEGY)
        df = imputer.fit_transform(df)
        fitted_imputer = imputer  # This is the new imputer we just trained

    # === STAGE 4: SAVE DATAFRAME ===
    # This part can be expanded with a real validation result later
    from types import SimpleNamespace

    result = SimpleNamespace(success=True)

    save_successful = save_dataframe_based_on_validation(
        result=result,
        df=df,
        file_name=input_filepath.stem,
        success_dir=config_gold.GOLD_PROCESSED_DIR,
        failure_dir=config_gold.GOLD_QUARANTINE_DIR,
    )

    pipeline_successful = result.success and save_successful
    if pipeline_successful:
        logger.info(f"--- Gold Pipeline: PASSED for {file_name} ---")
    else:
        logger.warning(f"--- Gold Pipeline: FAILED for {file_name} ---")

    return pipeline_successful, fitted_imputer


def main():
    """
    Orchestrates the entire Gold data processing workflow for train,
    validation, and test sets.
    """
    setup_logging_from_yaml(log_path=config_logging.GOLD_PIPELINE_LOGS_PATH)

    # --- ORCHESTRATION LOGIC ---
    imputer_path = config_gold.SIMPLE_IMPUTER_PATH

    # 1. Process Training Data
    logger.info(">>> ORCHESTRATOR: Processing training data...")
    train_path = config_silver.SILVER_PROCESSED_DIR / "train.parquet"
    train_success, fitted_imputer = gold_engineering_pipeline(input_filepath=train_path)

    if not train_success or not fitted_imputer:
        logger.critical("Training data processing failed. Aborting.")
        sys.exit(1)

    # Save the fitted imputer
    fitted_imputer.save(imputer_path)

    # 2. Process Validation Data
    logger.info(">>> ORCHESTRATOR: Processing validation data...")
    validation_path = config_silver.SILVER_PROCESSED_DIR / "validation.parquet"
    # Load the imputer that was fitted on the training data
    loaded_imputer = SimpleImputer.load(imputer_path)
    validation_success, _ = gold_engineering_pipeline(
        input_filepath=validation_path, imputer_to_apply=loaded_imputer
    )
    if not validation_success:
        logger.error("Validation data processing failed.")
        # In a real scenario, you might want to exit or send an alert here

    # 3. Process Test Data
    logger.info(">>> ORCHESTRATOR: Processing test data...")
    test_path = config_silver.SILVER_PROCESSED_DIR / "test.parquet"
    # Re-load for safety, though the object is the same
    loaded_imputer = SimpleImputer.load(imputer_path)
    test_success, _ = gold_engineering_pipeline(
        input_filepath=test_path, imputer_to_apply=loaded_imputer
    )
    if not test_success:
        logger.error("Test data processing failed.")

    logger.info(">>> ORCHESTRATOR: All data processing complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
