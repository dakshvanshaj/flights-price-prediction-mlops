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

# Import all our preprocessing modules
from gold_data_preprocessing.feature_engineering import (
    create_cyclical_features,
    create_categorical_interaction_features,
)
from gold_data_preprocessing.missing_value_imputer import SimpleImputer
from gold_data_preprocessing.rare_category_grouper import RareCategoryGrouper
from gold_data_preprocessing.categorical_encoder import CategoricalEncoder
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: Path,
    imputer_to_apply: Optional[SimpleImputer] = None,
    grouper_to_apply: Optional[RareCategoryGrouper] = None,
    encoder_to_apply: Optional[CategoricalEncoder] = None,
) -> Tuple[
    bool,
    Optional[SimpleImputer],
    Optional[RareCategoryGrouper],
    Optional[CategoricalEncoder],
]:
    """
    Executes the full gold layer pipeline in the correct order:
    Cleaning -> Imputation -> Feature Engineering -> Rare Category Grouping -> Encoding.
    """
    file_name = input_filepath.name
    logger.info(f"--- Starting Gold Engineering Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/6: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        return False, None, None, None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: DATA CLEANING ===
    logger.info("=" * 25 + " STAGE 2/6: DATA CLEANING " + "=" * 25)
    df = drop_columns(df, columns_to_drop=config_gold.GOLD_DROP_COLS)
    df = drop_duplicates(df, keep="first")
    df = drop_missing_target_rows(df, config_gold.TARGET_COLUMN)

    # === STAGE 3: IMPUTATION ===
    logger.info("=" * 25 + " STAGE 3/6: IMPUTATION " + "=" * 25)
    fitted_imputer = None
    if imputer_to_apply:
        df = imputer_to_apply.transform(df)
    else:
        imputer = SimpleImputer(strategy_dict=config_gold.IMPUTER_STRATEGY)
        df = imputer.fit_transform(df)
        fitted_imputer = imputer

    # === STAGE 4: FEATURE ENGINEERING ===
    logger.info("=" * 25 + " STAGE 4/6: FEATURE ENGINEERING " + "=" * 25)
    df = create_cyclical_features(df, cyclical_map=config_gold.CYCLICAL_FEATURES_MAP)
    df = create_categorical_interaction_features(
        df, interaction_map=config_gold.INTERACTION_FEATURES_CONFIG
    )

    # === STAGE 5: RARE CATEGORY GROUPING ===
    logger.info("=" * 25 + " STAGE 5/6: RARE CATEGORY GROUPING " + "=" * 25)
    fitted_grouper = None
    if grouper_to_apply:
        df = grouper_to_apply.transform(df)
    else:
        grouper = RareCategoryGrouper(
            columns=config_gold.HIGH_CARDINALITY_COLS,
            threshold=config_gold.CARDINALITY_THRESHOLD,
        )
        df = grouper.fit_transform(df)
        fitted_grouper = grouper

    # === STAGE 6: ENCODE CATEGORICAL COLUMNS ===
    logger.info("=" * 25 + " STAGE 6/6: ENCODING " + "=" * 25)
    fitted_encoder = None
    if encoder_to_apply:
        df = encoder_to_apply.transform(df)
    else:
        encoder = CategoricalEncoder(encoding_config=config_gold.ENCODING_CONFIG)
        df = encoder.fit_transform(df)
        fitted_encoder = encoder

    # === FINAL STAGE: SAVE DATAFRAME ===
    logger.info("=" * 25 + " SAVING DATAFRAME " + "=" * 25)
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
    logger.info(
        f"--- Gold Pipeline: {'PASSED' if pipeline_successful else 'FAILED'} for {file_name} ---"
    )

    return pipeline_successful, fitted_imputer, fitted_grouper, fitted_encoder


def main():
    """Orchestrates the Gold workflow for all data splits."""
    setup_logging_from_yaml(log_path=config_logging.GOLD_PIPELINE_LOGS_PATH)

    imputer_path = config_gold.SIMPLE_IMPUTER_PATH
    grouper_path = config_gold.RARE_CATEGORY_GROUPER_PATH
    encoder_path = config_gold.CATEGORICAL_ENCODER_PATH

    # 1. Process Training Data
    logger.info(">>> ORCHESTRATOR: Processing training data...")
    train_path = config_silver.SILVER_PROCESSED_DIR / "train.parquet"
    train_success, fitted_imputer, fitted_grouper, fitted_encoder = (
        gold_engineering_pipeline(input_filepath=train_path)
    )

    if not all([train_success, fitted_imputer, fitted_grouper, fitted_encoder]):
        logger.critical("Training data processing failed. Aborting.")
        sys.exit(1)

    fitted_imputer.save(imputer_path)
    fitted_grouper.save(grouper_path)
    fitted_encoder.save(encoder_path)
    logger.info("Successfully fitted and saved all preprocessing objects.")

    # 2. Process Validation & Test Data
    for data_split in ["validation", "test"]:
        logger.info(f">>> ORCHESTRATOR: Processing {data_split} data...")
        data_path = config_silver.SILVER_PROCESSED_DIR / f"{data_split}.parquet"

        imputer = SimpleImputer.load(imputer_path)
        grouper = RareCategoryGrouper.load(grouper_path)
        encoder = CategoricalEncoder.load(encoder_path)

        success, _, _, _ = gold_engineering_pipeline(
            input_filepath=data_path,
            imputer_to_apply=imputer,
            grouper_to_apply=grouper,
            encoder_to_apply=encoder,
        )
        if not success:
            logger.error(f"{data_split.capitalize()} data processing failed.")

    logger.info(">>> ORCHESTRATOR: All data processing complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
