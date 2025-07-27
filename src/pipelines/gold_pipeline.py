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

# Import the new encoder
from gold_data_preprocessing.categorical_encoder import CategoricalEncoder
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: Path,
    imputer_to_apply: Optional[SimpleImputer] = None,
    encoder_to_apply: Optional[CategoricalEncoder] = None,
) -> Tuple[bool, Optional[SimpleImputer], Optional[CategoricalEncoder]]:
    """
    Executes the full gold layer pipeline: cleaning, imputation, and encoding.
    Manages both training (fitting objects) and inference (applying them).
    """
    file_name = input_filepath.name
    logger.info(f"--- Starting Gold Engineering Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/5: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        return False, None, None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: DATA CLEANING ===
    logger.info("=" * 25 + " STAGE 2/5: DATA CLEANING " + "=" * 25)
    df = drop_columns(df, columns_to_drop=config_gold.GOLD_DROP_COLS)
    df = drop_duplicates(df, keep="first")
    df = drop_missing_target_rows(df, config_gold.TARGET_COLUMN)

    # === STAGE 3: IMPUTATION (TRAINING OR INFERENCE) ===
    logger.info("=" * 25 + " STAGE 3/5: IMPUTATION " + "=" * 25)
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

    # === STAGE 4: ENCODE CATEGORICAL COLUMNS (TRAINING OR INFERENCE) ===
    logger.info("=" * 25 + " STAGE 4/5: ENCODING " + "=" * 25)
    fitted_encoder = None
    if encoder_to_apply:
        # --- INFERENCE MODE ---
        logger.info("Applying existing encoder...")
        df = encoder_to_apply.transform(df)
    else:
        # --- TRAINING MODE ---
        logger.info("Fitting new encoder...")
        encoder = CategoricalEncoder(encoding_config=config_gold.ENCODING_CONFIG)
        df = encoder.fit_transform(df)
        fitted_encoder = encoder  # This is the new encoder we just trained

    # === STAGE 5: SAVE DATAFRAME ===
    logger.info("=" * 25 + " STAGE 5/5: SAVE DATAFRAME " + "=" * 25)
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

    return pipeline_successful, fitted_imputer, fitted_encoder


def main():
    """
    Orchestrates the entire Gold data processing workflow for train,
    validation, and test sets, including imputer and encoder lifecycle.
    """
    setup_logging_from_yaml(log_path=config_logging.GOLD_PIPELINE_LOGS_PATH)

    # --- ORCHESTRATION LOGIC ---
    imputer_path = config_gold.SIMPLE_IMPUTER_PATH
    encoder_path = config_gold.CATEGORICAL_ENCODER_PATH

    # 1. Process Training Data
    logger.info(">>> ORCHESTRATOR: Processing training data...")
    train_path = config_silver.SILVER_PROCESSED_DIR / "train.parquet"
    train_success, fitted_imputer, fitted_encoder = gold_engineering_pipeline(
        input_filepath=train_path
    )

    if not train_success or not fitted_imputer or not fitted_encoder:
        logger.critical("Training data processing failed. Aborting.")
        sys.exit(1)

    # Save the fitted preprocessing objects
    fitted_imputer.save(imputer_path)
    fitted_encoder.save(encoder_path)
    logger.info("Successfully fitted and saved imputer and encoder.")

    # 2. Process Validation Data
    logger.info(">>> ORCHESTRATOR: Processing validation data...")
    validation_path = config_silver.SILVER_PROCESSED_DIR / "validation.parquet"
    # Load the objects that were fitted on the training data
    loaded_imputer = SimpleImputer.load(imputer_path)
    loaded_encoder = CategoricalEncoder.load(encoder_path)
    validation_success, _, _ = gold_engineering_pipeline(
        input_filepath=validation_path,
        imputer_to_apply=loaded_imputer,
        encoder_to_apply=loaded_encoder,
    )
    if not validation_success:
        logger.error("Validation data processing failed.")

    # 3. Process Test Data
    logger.info(">>> ORCHESTRATOR: Processing test data...")
    test_path = config_silver.SILVER_PROCESSED_DIR / "test.parquet"
    # Re-load for safety, though the objects are the same
    loaded_imputer = SimpleImputer.load(imputer_path)
    loaded_encoder = CategoricalEncoder.load(encoder_path)
    test_success, _, _ = gold_engineering_pipeline(
        input_filepath=test_path,
        imputer_to_apply=loaded_imputer,
        encoder_to_apply=loaded_encoder,
    )
    if not test_success:
        logger.error("Test data processing failed.")

    logger.info(">>> ORCHESTRATOR: All data processing complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
