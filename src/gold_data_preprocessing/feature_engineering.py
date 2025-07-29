import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_cyclical_features(
    df: pd.DataFrame, cyclical_map: Dict[str, int]
) -> pd.DataFrame:
    """
    Engineers cyclical features from a dictionary mapping columns to their periods.
    """
    logger.info(
        f"Starting cyclical feature engineering for columns: {list(cyclical_map.keys())}..."
    )
    df_copy = df.copy()

    for col, period in cyclical_map.items():
        if col not in df_copy.columns:
            logger.warning(f"Cyclical column '{col}' not found in DataFrame. Skipping.")
            continue

        logger.info(f"Processing cyclical feature: '{col}' with a period of {period}.")

        df_copy[col + "_sin"] = np.sin(2 * np.pi * df_copy[col] / period)
        df_copy[col + "_cos"] = np.cos(2 * np.pi * df_copy[col] / period)

    cols_to_drop = list(cyclical_map.keys())
    df_copy.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    logger.info(
        f"Dropped original cyclical columns after transformation: {cols_to_drop}"
    )

    logger.info("Cyclical feature engineering complete.")
    return df_copy


def create_categorical_interaction_features(
    df: pd.DataFrame, interaction_map: Dict[str, List[str]], separator: str = "_"
) -> pd.DataFrame:
    """
    Creates new categorical features by combining existing ones based on a configuration map.

    Args:
        df (pd.DataFrame): The input DataFrame.
        interaction_map (Dict[str, List[str]]): A dictionary mapping the new feature name
            to a list of columns to combine.
            e.g., {'route': ['from_location', 'to_location']}
        separator (str): The string to use when joining the column values.

    Returns:
        pd.DataFrame: The DataFrame with the new interaction features added.
    """
    logger.info("Creating categorical interaction features...")
    df_copy = df.copy()

    for new_col_name, cols_to_interact in interaction_map.items():
        # Check if all required source columns exist
        missing_cols = [col for col in cols_to_interact if col not in df_copy.columns]
        if missing_cols:
            logger.warning(
                f"Cannot create interaction feature '{new_col_name}'. "
                f"Missing source columns: {missing_cols}. Skipping."
            )
            continue

        # Combine the columns to create the new feature
        df_copy[new_col_name] = (
            df_copy[cols_to_interact].astype(str).agg(separator.join, axis=1)
        )
        logger.info(f"Created interaction feature: '{new_col_name}'.")

    logger.info("Categorical interaction feature creation complete.")
    return df_copy


def create_numerical_interaction_features(
    df: pd.DataFrame, interaction_map: Dict[str, tuple]
) -> pd.DataFrame:
    """
    Creates new numerical features by performing operations on existing numerical columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        interaction_map (Dict[str, tuple]): A dictionary mapping the new feature name
            to a tuple containing two columns and the operation ('add', 'subtract',
            'multiply', 'divide').
            e.g., {'time_per_distance': ('time', 'distance', 'divide')}

    Returns:
        pd.DataFrame: The DataFrame with the new numerical features added.
    """
    logger.info("Creating numerical interaction features...")
    df_copy = df.copy()

    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b.replace(0, 1e-6),  # Avoid division by zero
    }

    for new_col_name, (col1, col2, op) in interaction_map.items():
        if op not in operations:
            logger.warning(
                f"Invalid operation '{op}' for feature '{new_col_name}'. Skipping."
            )
            continue

        if col1 not in df_copy.columns or col2 not in df_copy.columns:
            logger.warning(
                f"Cannot create numerical feature '{new_col_name}'. "
                f"Missing one or both source columns: '{col1}', '{col2}'. Skipping."
            )
            continue

        logger.info(
            f"Creating numerical feature '{new_col_name}' by applying '{op}' to '{col1}' and '{col2}'."
        )
        df_copy[new_col_name] = operations[op](df_copy[col1], df_copy[col2])

    logger.info("Numerical interaction feature creation complete.")
    return df_copy
