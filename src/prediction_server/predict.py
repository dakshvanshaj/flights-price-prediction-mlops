import pandas as pd
import logging
from silver_data_preprocessing.silver_preprocessing import (
    create_date_features,
    standardize_column_format,
    rename_specific_columns,
)
from gold_data_preprocessing.feature_engineering import (
    create_cyclical_features,
    create_categorical_interaction_features,
)
from shared.config import config_gold, config_training, config_silver

logger = logging.getLogger(__name__)


def preprocessing_for_prediction(input_df: pd.DataFrame, preprocessors: dict):
    """
    Applies all preprocessing steps to the input data to make it ready for prediction.
    This function mirrors the transformations from the silver and gold pipelines precisely.
    """
    df = input_df.copy()

    # --- 1. Silver Pipeline Steps ---
    # Rename and standardize column names to match the training pipeline
    df = rename_specific_columns(df, rename_mapping=config_silver.COLUMN_RENAME_MAPPING)
    df = standardize_column_format(df)
    # Convert date column to datetime objects for feature creation
    df["date"] = pd.to_datetime(df["date"])
    # Create all date-based features (e.g., month, day_of_week, day_of_year, etc.)
    df = create_date_features(df, date_column="date")

    # --- 2. Gold Pipeline Steps ---

    # STAGE 2: DATA CLEANING
    # Drop columns that are explicitly removed at the start of the gold pipeline.
    # This is a crucial step to match the training process exactly.
    cols_to_drop = [col for col in config_gold.GOLD_DROP_COLS if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns to align with gold pipeline: {cols_to_drop}")

    # STAGE 3: IMPUTATION
    df = preprocessors["imputer"].transform(df)

    # STAGE 4: FEATURE ENGINEERING
    df = create_cyclical_features(df, cyclical_map=config_gold.CYCLICAL_FEATURES_MAP)
    df = create_categorical_interaction_features(
        df, interaction_map=config_gold.INTERACTION_FEATURES_CONFIG
    )

    # STAGE 5: ENCODING
    df = preprocessors["encoder"].transform(df)

    # STAGES 6-9: Non-tree model transformations (apply if the preprocessor exists)
    if preprocessors.get("grouper"):
        df = preprocessors["grouper"].transform(df)
    if preprocessors.get("outlier_handler"):
        df = preprocessors["outlier_handler"].transform(df)
    if preprocessors.get("power_transformer"):
        df = preprocessors["power_transformer"].transform(df)
    if preprocessors.get("scaler"):
        df = preprocessors["scaler"].transform(df)

    # STAGE 10: SANITIZE COLUMN NAMES
    df.columns = df.columns.str.replace(" ", "_", regex=False).str.lower()

    # --- 3. Final Column Conformance ---
    # Get the exact list of columns the model was trained on
    final_model_cols = preprocessors["final_columns"]

    # Add any missing columns that were created during training (e.g., by one-hot encoding)
    # and fill them with 0. This prevents errors if the prediction input is missing a category
    # that was present in the training data.
    missing_cols = set(final_model_cols) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    if missing_cols:
        logger.info(f"Added missing columns and filled with 0: {missing_cols}")

    # Get the final list of columns for prediction, excluding the target variable
    cols_for_prediction = [
        col for col in final_model_cols if col != config_training.TARGET_COLUMN
    ]

    # Enforce the exact column order the model expects, raising an error if a
    # column is still missing.
    try:
        df = df[cols_for_prediction]
    except KeyError as e:
        logger.error(f"A required column is missing from the dataframe: {e}")
        raise

    return df
