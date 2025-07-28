import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def create_cyclical_features(
    df: pd.DataFrame, cyclical_map: Dict[str, int]
) -> pd.DataFrame:
    """
    Engineers cyclical features from a dictionary mapping columns to their periods.

    This function applies sine/cosine transformations to preserve the cyclical
    nature of time-based features and then removes the original columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cyclical_map (Dict[str, int]): A dictionary mapping column names to their
                                       maximum value (period).
                                       e.g., {'month': 12, 'day_of_week': 7}

    Returns:
        pd.DataFrame: A new DataFrame with the added cyclical features and
                      the original source columns removed.
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

        # Apply sine and cosine transformations
        df_copy[col + "_sin"] = np.sin(2 * np.pi * df_copy[col] / period)
        df_copy[col + "_cos"] = np.cos(2 * np.pi * df_copy[col] / period)

    # Drop the original columns that were just used for the transformation
    cols_to_drop = list(cyclical_map.keys())
    df_copy.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    logger.info(
        f"Dropped original cyclical columns after transformation: {cols_to_drop}"
    )

    logger.info("Cyclical feature engineering complete.")
    return df_copy
