import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Removes specified columns from a DataFrame.

    This function is a wrapper around pandas.DataFrame.drop() to provide
    standardized logging and ensure an immutable operation.

    Args:
        df: The input pandas DataFrame.
        columns_to_drop: A list of column names to be dropped.

    Returns:
        A new DataFrame with the specified columns removed.
    """
    df_copy = df.copy()
    initial_shape = df_copy.shape
    logger.info(f"Dropping columns: {columns_to_drop}")
    logger.debug(f"Initial DataFrame shape: {initial_shape}")

    df_copy = df_copy.drop(columns=columns_to_drop, errors="ignore")

    final_shape = df_copy.shape
    logger.info(f"Successfully dropped {len(columns_to_drop)} columns.")
    logger.debug(f"Final DataFrame shape: {final_shape}")

    return df_copy


def drop_duplicates(
    df: pd.DataFrame, subset_cols: Optional[List[str]] = None, keep: str = "first"
) -> pd.DataFrame:
    """
    Removes duplicate rows from a DataFrame.

    This is often used after removing unique identifiers (like user_code) to
    aggregate the data to a specific level of granularity (e.g., one record
    per unique route on a given day).

    Args:
        df: The input pandas DataFrame.
        subset_cols: List of column names to consider for identifying duplicates.
                     If None, all columns are used.
        keep: Determines which duplicate to keep ('first', 'last', False).

    Returns:
        A new DataFrame with duplicate rows removed.
    """
    df_copy = df.copy()
    initial_rows = len(df_copy)
    logger.info("Dropping duplicate rows...")
    logger.debug(f"Initial row count: {initial_rows}")

    df_copy = df_copy.drop_duplicates(subset=subset_cols, keep=keep)

    final_rows = len(df_copy)
    rows_removed = initial_rows - final_rows

    logger.info(
        f"Removed {rows_removed} duplicate row(s). Final row count: {final_rows}"
    )

    return df_copy


def drop_missing_target_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Drops rows where the target variable is missing.

    In supervised learning, rows without a target value are unusable for training.
    This is a safer alternative to dropping all rows with any missing value.

    Args:
        df: The input pandas DataFrame.
        target_column: The name of the target variable column (e.g., 'price').

    Returns:
        A new DataFrame with rows containing a missing target value removed.
    """
    df_copy = df.copy()
    initial_rows = len(df_copy)
    logger.info(
        f"Dropping rows with missing values in target column: '{target_column}'"
    )

    df_copy = df_copy.dropna(subset=[target_column])

    final_rows = len(df_copy)
    rows_removed = initial_rows - final_rows

    if rows_removed > 0:
        logger.info(
            f"Removed {rows_removed} row(s) with missing target. Final row count: {final_rows}"
        )
    else:
        logger.info("No rows with missing target found.")

    return df_copy
