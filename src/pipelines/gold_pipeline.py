import sys
import logging
import yaml
from data_ingestion.data_loader import load_data
from shared.config import config_gold, config_logging, config_silver
from shared.utils import setup_logging_from_yaml, save_dataframe_based_on_validation
from pathlib import Path
from typing import Optional, Tuple

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
from gold_data_preprocessing.outlier_handling import OutlierTransformer
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler


logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: Path,
    params: Optional[dict] = None,
    imputer_to_apply: Optional[SimpleImputer] = None,
    grouper_to_apply: Optional[RareCategoryGrouper] = None,
    encoder_to_apply: Optional[CategoricalEncoder] = None,
    outlier_handler_to_apply: Optional[OutlierTransformer] = None,
    power_transformer_to_apply: Optional[PowerTransformer] = None,
    scaler_to_apply: Optional[Scaler] = None,
) -> Tuple[
    bool,
    Optional[SimpleImputer],
    Optional[RareCategoryGrouper],
    Optional[CategoricalEncoder],
    Optional[OutlierTransformer],
    Optional[PowerTransformer],
    Optional[Scaler],
]:
    """
    Executes the full gold layer pipeline in the correct order.
    """
    file_name = input_filepath.name
    logger.info(f"--- Starting Gold Engineering Pipeline for: {file_name} ---")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/9: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        # --- FIX: Ensure we return the correct number of None values ---
        return False, None, None, None, None, None, None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: DATA CLEANING ===
    logger.info("=" * 25 + " STAGE 2/9: DATA CLEANING " + "=" * 25)
    df = drop_columns(df, columns_to_drop=config_gold.GOLD_DROP_COLS)
    df = drop_duplicates(df, keep="first")
    df = drop_missing_target_rows(df, config_gold.TARGET_COLUMN)

    # === STAGE 3: IMPUTATION ===
    logger.info("=" * 25 + " STAGE 3/9: IMPUTATION " + "=" * 25)
    fitted_imputer = None
    if imputer_to_apply:
        df = imputer_to_apply.transform(df)
    else:
        IMPUTER_STRATEGY = params["imputation"]
        imputer = SimpleImputer(strategy_dict=IMPUTER_STRATEGY)
        df = imputer.fit_transform(df)
        fitted_imputer = imputer

    # === STAGE 4: FEATURE ENGINEERING ===
    logger.info("=" * 25 + " STAGE 4/9: FEATURE ENGINEERING " + "=" * 25)
    df = create_cyclical_features(df, cyclical_map=config_gold.CYCLICAL_FEATURES_MAP)
    df = create_categorical_interaction_features(
        df, interaction_map=config_gold.INTERACTION_FEATURES_CONFIG
    )

    # === STAGE 5: RARE CATEGORY GROUPING ===
    logger.info("=" * 25 + " STAGE 5/9: RARE CATEGORY GROUPING " + "=" * 25)
    fitted_grouper = None
    if grouper_to_apply:
        df = grouper_to_apply.transform(df)
    else:
        grouper_params = params["rare_category_grouping"]
        CARDINALITY_THRESHOLD = grouper_params["cardinality_threshold"]
        grouper = RareCategoryGrouper(
            columns=config_gold.HIGH_CARDINALITY_COLS,
            threshold=CARDINALITY_THRESHOLD,
        )
        df = grouper.fit_transform(df)
        fitted_grouper = grouper

    # === STAGE 6: ENCODE CATEGORICAL COLUMNS ===
    logger.info("=" * 25 + " STAGE 6/9: ENCODING " + "=" * 25)
    fitted_encoder = None
    if encoder_to_apply:
        df = encoder_to_apply.transform(df)
    else:
        encoder = CategoricalEncoder(encoding_config=config_gold.ENCODING_CONFIG)
        df = encoder.fit_transform(df)
        fitted_encoder = encoder

    # === STAGE 7: OUTLIER DETECTION AND HANDLING ===
    logger.info("=" * 25 + " STAGE 7/9: OUTLIER HANDLING " + "=" * 25)
    fitted_outlier_handler = None
    if outlier_handler_to_apply:
        df = outlier_handler_to_apply.transform(df)
    else:
        outlier_params = params["outlier_handling"]
        OUTLIER_DETECTION_STRATEGY = outlier_params["detection_strategy"]
        OUTLIER_HANDLING_STRATEGY = outlier_params["handling_strategy"]
        ISO_FOREST_CONTAMINATION = outlier_params["iso_forest_contamination"]
        outlier_handler = OutlierTransformer(
            detection_strategy=OUTLIER_DETECTION_STRATEGY,
            handling_strategy=OUTLIER_HANDLING_STRATEGY,
            columns=config_gold.OUTLIER_HANDLER_COLUMNS,
            contamination=ISO_FOREST_CONTAMINATION,
            random_state=42,
        )
        df = outlier_handler.fit_transform(df)
        fitted_outlier_handler = outlier_handler

    # === STAGE 8: POWER TRANSFORMATIONS ===
    logger.info("=" * 25 + " STAGE 8/9: POWER TRANSFORMATIONS " + "=" * 25)
    fitted_power_transformer = None
    if power_transformer_to_apply:
        df = power_transformer_to_apply.transform(df)
    else:
        power_params = params["power_transformer"]
        POWER_TRANSFORMER_STRATEGY = power_params["strategy"]
        power_transformer = PowerTransformer(
            columns=config_gold.POWER_TRANSFORMER_COLUMNS,
            strategy=POWER_TRANSFORMER_STRATEGY,
        )
        df = power_transformer.fit_transform(df)
        fitted_power_transformer = power_transformer

    # === STAGE 9: SCALING ===
    logger.info("=" * 25 + " STAGE 9/9: SCALING " + "=" * 25)
    fitted_scaler = None
    if scaler_to_apply:
        df = scaler_to_apply.transform(df)
    else:
        scaler = Scaler(
            columns=config_gold.SCALER_COLUMNS, strategy=config_gold.SCALER_STRATEGY
        )
        df = scaler.fit_transform(df)
        fitted_scaler = scaler

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

    return (
        pipeline_successful,
        fitted_imputer,
        fitted_grouper,
        fitted_encoder,
        fitted_outlier_handler,
        fitted_power_transformer,
        fitted_scaler,
    )


def main():
    """Orchestrates the Gold workflow for all data splits."""
    setup_logging_from_yaml(log_path=config_logging.GOLD_PIPELINE_LOGS_PATH)

    imputer_path = config_gold.SIMPLE_IMPUTER_PATH
    grouper_path = config_gold.RARE_CATEGORY_GROUPER_PATH
    encoder_path = config_gold.CATEGORICAL_ENCODER_PATH
    outlier_path = config_gold.OUTLIER_HANDLER_PATH
    power_path = config_gold.POWER_TRANSFORMER_PATH
    scaler_path = config_gold.SCALER_PATH

    # ==== Import Parameters From params.yaml =====
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    # Check the parameters dictionary for gold_pipeline
    gold_params = params["gold_pipeline"]

    # 1. Process Training Data
    logger.info(">>> ORCHESTRATOR: Processing training data...")
    train_path = config_silver.SILVER_PROCESSED_DIR / "train.parquet"
    (
        train_success,
        fitted_imputer,
        fitted_grouper,
        fitted_encoder,
        fitted_outlier_handler,
        fitted_power_transformer,
        fitted_scaler,
    ) = gold_engineering_pipeline(input_filepath=train_path, params=gold_params)

    if not all(
        [
            train_success,
            fitted_imputer,
            fitted_grouper,
            fitted_encoder,
            fitted_outlier_handler,
            fitted_power_transformer,
            fitted_scaler,
        ]
    ):
        logger.critical("Training data processing failed. Aborting.")
        sys.exit(1)

    fitted_imputer.save(imputer_path)
    fitted_grouper.save(grouper_path)
    fitted_encoder.save(encoder_path)
    fitted_outlier_handler.save(outlier_path)
    fitted_power_transformer.save(power_path)
    fitted_scaler.save(scaler_path)

    logger.info("Successfully fitted and saved all preprocessing objects.")

    # 2. Process Validation & Test Data
    for data_split in ["validation", "test"]:
        logger.info(f">>> ORCHESTRATOR: Processing {data_split} data...")
        data_path = config_silver.SILVER_PROCESSED_DIR / f"{data_split}.parquet"

        imputer = SimpleImputer.load(imputer_path)
        grouper = RareCategoryGrouper.load(grouper_path)
        encoder = CategoricalEncoder.load(encoder_path)
        outlier_handler = OutlierTransformer.load(outlier_path)
        transformer = PowerTransformer.load(power_path)
        scaler = Scaler.load(scaler_path)

        (
            success,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = gold_engineering_pipeline(
            input_filepath=data_path,
            imputer_to_apply=imputer,
            grouper_to_apply=grouper,
            encoder_to_apply=encoder,
            outlier_handler_to_apply=outlier_handler,
            power_transformer_to_apply=transformer,
            scaler_to_apply=scaler,
        )
        if not success:
            logger.error(f"{data_split.capitalize()} data processing failed.")

    logger.info(">>> ORCHESTRATOR: All data processing complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
