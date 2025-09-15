"""
Core preprocessing and postprocessing logic for the prediction server.
"""
import logging
from typing import Any, Dict

import pandas as pd

from gold_data_preprocessing.feature_engineering import (
    create_categorical_interaction_features,
    create_cyclical_features,
)
from shared.config import config_gold, config_silver, config_training
from silver_data_preprocessing.silver_preprocessing import (
    create_date_features,
    rename_specific_columns,
    standardize_column_format,
)

logger = logging.getLogger(__name__)


def preprocessing_for_prediction(
    input_df: pd.DataFrame, preprocessors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies all data transformations to mirror the training pipeline.

    This function takes a raw input DataFrame and applies all necessary steps
    from the silver and gold pipelines to prepare it for the model.

    Args:
        input_df: The raw input data as a single-row DataFrame.
        preprocessors: A dictionary of loaded preprocessing artifacts.

    Returns:
        A DataFrame ready for prediction, with the exact columns and order
        the model expects.
    """
    df = input_df.copy()
    logger.info("Starting preprocessing for prediction...")

    # --- 1. Silver Pipeline Steps ---
    logger.debug("Applying Silver pipeline transformations...")
    df = rename_specific_columns(df, rename_mapping=config_silver.COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    df["date"] = pd.to_datetime(df["date"])
    df = create_date_features(df, date_column="date")

    # --- 2. Gold Pipeline Steps ---
    logger.debug("Applying Gold pipeline transformations...")
    # STAGE 2: DATA CLEANING (Drop columns not used in the model)
    cols_to_drop = [col for col in config_gold.GOLD_DROP_COLS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.debug(f"Dropped columns to align with gold pipeline: {cols_to_drop}")

    # STAGE 3: IMPUTATION
    if "imputer" in preprocessors:
        logger.debug("Applying transformer: imputer")
        df = preprocessors["imputer"].transform(df)

    # --- FIX: STAGE 4: FEATURE ENGINEERING ---
    # This step must happen AFTER imputation and BEFORE encoding to match the training pipeline.
    logger.debug("Applying feature engineering (cyclical, interactions)...")
    df = create_cyclical_features(df, cyclical_map=config_gold.CYCLICAL_FEATURES_MAP)
    df = create_categorical_interaction_features(
        df, interaction_map=config_gold.INTERACTION_FEATURES_CONFIG
    )
    # --- END OF FIX ---

    # STAGE 5: ENCODING
    if "encoder" in preprocessors:
        logger.debug("Applying transformer: encoder")
        df = preprocessors["encoder"].transform(df)

    # STAGES 6-9: Apply remaining optional transformers
    for name in ["grouper", "outlier_handler", "power_transformer", "scaler"]:
        if name in preprocessors:
            logger.debug(f"Applying transformer: {name}")
            df = preprocessors[name].transform(df)

    # STAGE 10: SANITIZE COLUMN NAMES
    df.columns = df.columns.str.replace(" ", "_", regex=False).str.lower()

    # --- 3. Final Column Conformance ---
    logger.debug("Enforcing final column schema...")
    final_model_cols = preprocessors["final_columns"]
    cols_for_prediction = [
        col for col in final_model_cols if col != config_training.TARGET_COLUMN
    ]

    # Add any missing columns that were created during training (e.g., by one-hot encoding)
    missing_cols = set(cols_for_prediction) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    if missing_cols:
        logger.debug(f"Added missing one-hot encoded columns: {missing_cols}")

    # Enforce the exact column order the model expects
    try:
        df = df[cols_for_prediction]
    except KeyError as e:
        logger.error(f"A required column is missing after preprocessing: {e}")
        raise

    logger.info("Preprocessing complete.")
    return df


def postprocessing_for_target(
    prediction_df: pd.DataFrame, preprocessors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies inverse transformations to the model's prediction.

    If the target variable was scaled or transformed during training, this
    function reverses those operations to return a prediction in the original,
    interpretable scale.

    Args:
        prediction_df: A DataFrame with the model's scaled prediction.
        preprocessors: The dictionary of loaded preprocessing artifacts.

    Returns:
        A DataFrame with the prediction transformed back to the original scale.
    """
    # The training pipeline may apply PowerTransform -> Scale,
    # so we must inverse in the reverse order: Scale -> PowerTransform.
    if "scaler" in preprocessors:
        logger.debug("Applying inverse scaling to prediction.")
        prediction_df = preprocessors["scaler"].inverse_transform(prediction_df)

    if "power_transformer" in preprocessors:
        logger.debug("Applying inverse power transform to prediction.")
        prediction_df = preprocessors["power_transformer"].inverse_transform(
            prediction_df
        )

    return prediction_df
