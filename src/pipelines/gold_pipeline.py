import sys
import logging
import yaml
import json
from data_ingestion.data_loader import load_data
from shared.config import config_gold, config_logging, config_silver, core_paths
from shared.utils import setup_logging_from_yaml, save_dataframe_based_on_validation
from pathlib import Path
from typing import Optional, Tuple

from gold_data_preprocessing.data_cleaning import (
    drop_columns,
    drop_duplicates,
    drop_missing_target_rows,
)
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
from data_validation.ge_components import run_checkpoint_on_dataframe
from data_validation.expectations.gold_expectations import build_gold_expectations


logger = logging.getLogger(__name__)


def gold_engineering_pipeline(
    input_filepath: Path,
    scaler_strategy_for_validation: str,
    params: Optional[dict] = None,
    is_tree_model: bool = False,
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
    logger.info(f"Tree-based model path: {'ENABLED' if is_tree_model else 'DISABLED'}")

    # === STAGE 1: DATA INGESTION ===
    logger.info("=" * 25 + " STAGE 1/11: DATA INGESTION " + "=" * 25)
    df = load_data(input_filepath)
    if df is None:
        return False, None, None, None, None, None, None
    logger.info(f"Successfully loaded {len(df)} rows.")

    # === STAGE 2: DATA CLEANING ===
    logger.info("=" * 25 + " STAGE 2/11: DATA CLEANING " + "=" * 25)
    df = drop_columns(df, columns_to_drop=config_gold.GOLD_DROP_COLS)
    df = drop_duplicates(df, keep="first")
    df = drop_missing_target_rows(df, config_gold.TARGET_COLUMN)

    # === STAGE 3: IMPUTATION ===
    logger.info("=" * 25 + " STAGE 3/11: IMPUTATION " + "=" * 25)
    fitted_imputer = None
    if imputer_to_apply:
        df = imputer_to_apply.transform(df)
    else:
        IMPUTER_STRATEGY = params["imputation"]
        imputer = SimpleImputer(strategy_dict=IMPUTER_STRATEGY)
        df = imputer.fit_transform(df)
        fitted_imputer = imputer

    # === STAGE 4: FEATURE ENGINEERING ===
    logger.info("=" * 25 + " STAGE 4/11: FEATURE ENGINEERING " + "=" * 25)
    df = create_cyclical_features(df, cyclical_map=config_gold.CYCLICAL_FEATURES_MAP)
    df = create_categorical_interaction_features(
        df, interaction_map=config_gold.INTERACTION_FEATURES_CONFIG
    )

    # === STAGE 5: ENCODE CATEGORICAL COLUMNS ===
    logger.info("=" * 25 + " STAGE 5/11: ENCODING " + "=" * 25)
    fitted_encoder = None
    if encoder_to_apply:
        df = encoder_to_apply.transform(df)
    else:
        encoder = CategoricalEncoder(encoding_config=config_gold.ENCODING_CONFIG)
        df = encoder.fit_transform(df)
        fitted_encoder = encoder

    fitted_grouper = None
    fitted_outlier_handler = None
    fitted_power_transformer = None
    fitted_scaler = None
    SCALER_STRATEGY = "none"

    if not is_tree_model:
        # === STAGE 6: RARE CATEGORY GROUPING ===
        logger.info("=" * 25 + " STAGE 6/11: RARE CATEGORY GROUPING " + "=" * 25)
        grouper_params = params.get("rare_category_grouping")
        if grouper_to_apply:
            df = grouper_to_apply.transform(df)
            fitted_grouper = grouper_to_apply
        elif grouper_params and grouper_params.get("cardinality_threshold"):
            CARDINALITY_THRESHOLD = grouper_params["cardinality_threshold"]
            grouper = RareCategoryGrouper(
                columns=config_gold.HIGH_CARDINALITY_COLS,
                threshold=CARDINALITY_THRESHOLD,
            )
            df = grouper.fit_transform(df)
            fitted_grouper = grouper
        else:
            logger.info("Skipping Rare Category Grouping: not configured in params.")

        # === STAGE 7: OUTLIER DETECTION AND HANDLING ===
        logger.info("=" * 25 + " STAGE 7/11: OUTLIER HANDLING " + "=" * 25)
        outlier_params = params.get("outlier_handling")
        if outlier_handler_to_apply:
            df = outlier_handler_to_apply.transform(df)
            fitted_outlier_handler = outlier_handler_to_apply
        elif outlier_params and outlier_params.get("detection_strategy"):
            outlier_handler = OutlierTransformer(
                detection_strategy=outlier_params["detection_strategy"],
                handling_strategy=outlier_params["handling_strategy"],
                columns=config_gold.OUTLIER_HANDLER_COLUMNS,
            )
            df = outlier_handler.fit_transform(df)
            fitted_outlier_handler = outlier_handler
        else:
            logger.info("Skipping Outlier Handling: not configured in params.")

        # === STAGE 8: POWER TRANSFORMATIONS ===
        logger.info("=" * 25 + " STAGE 8/11: POWER TRANSFORMATIONS " + "=" * 25)
        power_params = params.get("power_transformer")
        if power_transformer_to_apply:
            df = power_transformer_to_apply.transform(df)
            fitted_power_transformer = power_transformer_to_apply
        elif power_params and power_params.get("strategy"):
            power_transformer = PowerTransformer(
                columns=config_gold.POWER_TRANSFORMER_COLUMNS,
                strategy=power_params["strategy"],
            )
            df = power_transformer.fit_transform(df)
            fitted_power_transformer = power_transformer
        else:
            logger.info("Skipping Power Transformations: not configured in params.")

        # === STAGE 9: SCALING ===
        logger.info("=" * 25 + " STAGE 9/11: SCALING " + "=" * 25)
        scaler_params = params.get("scaler")
        if scaler_to_apply:
            df = scaler_to_apply.transform(df)
            SCALER_STRATEGY = scaler_strategy_for_validation
            fitted_scaler = scaler_to_apply
        elif scaler_params and scaler_params.get("strategy"):
            SCALER_STRATEGY = scaler_params["strategy"]
            scaler = Scaler(
                columns=config_gold.SCALER_COLUMNS, strategy=SCALER_STRATEGY
            )
            df = scaler.fit_transform(df)
            fitted_scaler = scaler
        else:
            logger.info("Skipping Scaling: not configured in params.")
    else:
        logger.info(
            "Tree model path enabled: Skipping Outlier Handling, Power Transformations, and Scaling."
        )

    # === STAGE 10: SANITIZE COLUMN NAMES ===
    logger.info("=" * 25 + " STAGE 10/11: SANITIZE COLUMN NAMES " + "=" * 25)
    df.columns = df.columns.str.replace(" ", "_", regex=False).str.lower()
    logger.info("Sanitized column names to be lowercase and snake_case.")

    # === STAGE 11: DATA VALIDATION (QUALITY GATE) ===
    logger.info("=" * 25 + " STAGE 11/11: FINAL VALIDATION " + "=" * 25)

    final_cols_path = config_gold.GOLD_FINAL_COLS_PATH
    # Check if ANY optional transformer was applied in the training run
    is_training_run = not any(
        [
            scaler_to_apply,
            power_transformer_to_apply,
            outlier_handler_to_apply,
            grouper_to_apply,
        ]
    )
    if is_training_run:
        logger.info(f"Training run: saving final column order to {final_cols_path}")
        final_cols_order = list(df.columns)
        with open(final_cols_path, "w") as f:
            json.dump(final_cols_order, f)
    else:
        logger.info(
            f"Validation/Test run: loading final column order from {final_cols_path}"
        )
        with open(final_cols_path, "r") as f:
            final_cols_order = json.load(f)

    gold_expectations = build_gold_expectations(
        expected_cols_ordered=final_cols_order,
        scaler_strategy=SCALER_STRATEGY,
        scaled_cols=config_gold.GOLD_SCALED_COLS
        if not is_tree_model
        else None,  # Pass scaled_cols conditionally
        is_tree_model=is_tree_model,
        target_col=config_gold.TARGET_COLUMN,
    )
    result = run_checkpoint_on_dataframe(
        project_root_dir=core_paths.GE_ROOT_DIR,
        datasource_name=config_gold.GOLD_DATA_SOURCE_NAME,
        asset_name=config_gold.GOLD_ASSET_NAME,
        batch_definition_name=config_gold.GOLD_BATCH_DEFINITION_NAME,
        suite_name=config_gold.GOLD_SUITE_NAME,
        validation_definition_name=config_gold.GOLD_VALIDATION_DEFINITION_NAME,
        checkpoint_name=config_gold.GOLD_CHECKPOINT_NAME,
        dataframe_to_validate=df,
        expectation_list=gold_expectations,
    )

    if not result.success:
        logger.warning("--- DATA VALIDATION FAILED! ---")
        logger.warning(
            "To see the full report, open the Data Docs: great_expectations/uncommitted/data_docs/local_site/index.html"
        )
        logger.warning("For quick debugging, here are the failing expectations:")

        for validation_result in result["results"]:
            if not validation_result["success"]:
                expectation_config = validation_result["expectation_config"]
                expectation_type = expectation_config["expectation_type"]
                kwargs = expectation_config["kwargs"]
                observed_value = validation_result["result"].get("observed_value")

                logger.warning(f"  - Expectation: {expectation_type}")
                logger.warning(f"    Kwargs: {kwargs}")
                logger.warning(f"    Observed Value: {observed_value}")
        logger.warning("------------------------------------")

    # === FINAL STAGE: SAVE DATAFRAME ===
    logger.info("=" * 25 + " SAVING DATAFRAME " + "=" * 25)

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

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    gold_params = params.get("gold_pipeline", {})
    is_tree_model = params.get("is_tree_model", False)

    # 1. Process Training Data
    logger.info("=" * 25 + ">>> ORCHESTRATOR: Processing training data..." + "=" * 25)
    train_path = config_silver.SILVER_PROCESSED_DIR / "train.parquet"

    scaler_params = gold_params.get("scaler", {})
    scaler_strategy = scaler_params.get("strategy", "none")

    (
        train_success,
        fitted_imputer,
        fitted_grouper,
        fitted_encoder,
        fitted_outlier_handler,
        fitted_power_transformer,
        fitted_scaler,
    ) = gold_engineering_pipeline(
        input_filepath=train_path,
        params=gold_params,
        scaler_strategy_for_validation=scaler_strategy,
        is_tree_model=is_tree_model,
    )

    if not train_success:
        logger.critical("Training data processing failed. Aborting.")
        sys.exit(1)

    # Save only the transformers that were actually fitted
    if fitted_imputer:
        fitted_imputer.save(imputer_path)
    if fitted_grouper:
        fitted_grouper.save(grouper_path)
    if fitted_encoder:
        fitted_encoder.save(encoder_path)
    if fitted_outlier_handler:
        fitted_outlier_handler.save(outlier_path)
    if fitted_power_transformer:
        fitted_power_transformer.save(power_path)
    if fitted_scaler:
        fitted_scaler.save(scaler_path)

    logger.info("Successfully fitted and saved all applicable preprocessing objects.")

    # 2. Process Validation & Test Data
    for data_split in ["validation", "test"]:
        logger.info(
            "=" * 25 + f">>> ORCHESTRATOR: Processing {data_split} data..." + "=" * 25
        )
        data_path = config_silver.SILVER_PROCESSED_DIR / f"{data_split}.parquet"

        # Load only the transformers that were saved
        imputer = SimpleImputer.load(imputer_path) if imputer_path.exists() else None
        grouper = (
            RareCategoryGrouper.load(grouper_path) if grouper_path.exists() else None
        )
        encoder = (
            CategoricalEncoder.load(encoder_path) if encoder_path.exists() else None
        )
        outlier_handler = (
            OutlierTransformer.load(outlier_path) if outlier_path.exists() else None
        )
        transformer = PowerTransformer.load(power_path) if power_path.exists() else None
        scaler = Scaler.load(scaler_path) if scaler_path.exists() else None

        scaler_strategy_for_validation = scaler.strategy if scaler else "none"

        (success, _, _, _, _, _, _) = gold_engineering_pipeline(
            input_filepath=data_path,
            imputer_to_apply=imputer,
            params=gold_params,
            grouper_to_apply=grouper,
            encoder_to_apply=encoder,
            outlier_handler_to_apply=outlier_handler,
            power_transformer_to_apply=transformer,
            scaler_to_apply=scaler,
            scaler_strategy_for_validation=scaler_strategy_for_validation,
            is_tree_model=is_tree_model,
        )
        if not success:
            logger.error(f"{data_split.capitalize()} data processing failed.")

    logger.info(">>> ORCHESTRATOR: All data processing complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
