import pandas as pd
import logging
import re
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def rename_specific_columns(
    df: pd.DataFrame, rename_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Renames specific columns in a DataFrame based on a provided mapping."""
    df = df.copy()
    df.rename(columns=rename_mapping, inplace=True)
    logger.info("Applied specific column renaming.")
    return df


def standardize_column_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes all column names in a DataFrame to a consistent snake_case format.
    Handles both 'camelCase' and 'Title Case Names'.
    """
    df = df.copy()

    clean_cols = []
    for col in df.columns:
        # 1. Remove spaces to correctly handle inputs like "First Name" -> "FirstName"
        temp_col = col.replace(" ", "")

        # 2. Insert an underscore before any capital letter (for camelCase)
        #    Example: "FirstName" -> "First_Name"
        temp_col = re.sub(r"(?<!^)(?=[A-Z])", "_", temp_col)

        # 3. Convert the entire string to lowercase
        #    Example: "First_Name" -> "first_name"
        standardized_col = temp_col.lower()

        clean_cols.append(standardized_col)

    df.columns = clean_cols
    logger.info("Standardized column name formats to snake_case.")
    logger.debug(f"Final standardized columns: {', '.join(df.columns)}")
    return df


def optimize_data_types(
    df: pd.DataFrame, date_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Optimizes data types and robustly parses date columns to reduce memory usage."""
    df = df.copy()

    initial_mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory Usage Before Optimization: {initial_mem_usage:.2f} MB")
    logger.debug(f"Initial dtypes:\n{df.dtypes}")

    COMMON_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]

    if date_cols is None:
        date_cols = []

    for col in df.columns:
        old_dtype = df[col].dtype

        if col in date_cols and old_dtype == "object":
            parsed_successfully = False
            for fmt in COMMON_DATE_FORMATS:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt)
                    parsed_successfully = True
                    break
                except (ValueError, TypeError):
                    continue

            if not parsed_successfully:
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.warning(f"Column '{col}' parsed using slow date inference.")
                    parsed_successfully = True
                except (ValueError, TypeError):
                    logger.error(f"Failed to parse column '{col}' as datetime.")

            if parsed_successfully:
                new_dtype = df[col].dtype
                logger.info(
                    f"[DataTypeOpt] Converted '{col}': {old_dtype} -> {new_dtype}"
                )
                continue

        if str(old_dtype).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif str(old_dtype).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif old_dtype == "object":
            # Convert to category if the number of unique values is less than 50%
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")

        new_dtype = df[col].dtype
        if new_dtype != old_dtype:
            logger.info(f"[DataTypeOpt] Converted '{col}': {old_dtype} -> {new_dtype}")

    final_mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    percent_reduction = (initial_mem_usage - final_mem_usage) / initial_mem_usage * 100

    logger.info(f"Memory Usage After Optimization: {final_mem_usage:.2f} MB")
    logger.info(f"Memory reduced by {percent_reduction:.2f}%.")
    return df


def sort_data_by_date(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Sorts the DataFrame by the specified date column for chronological consistency.
    """
    logger.info(f"Sorting DataFrame by '{date_column}' column...")
    if date_column not in df.columns:
        logger.error(f"Date column '{date_column}' not found. Skipping sorting.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        logger.error(f"Column '{date_column}' is not datetime. Skipping sorting.")
        return df
    if df[date_column].is_monotonic_increasing:
        logger.info(f"Data is already sorted by '{date_column}'. No changes made.")
        return df
    df_sorted = df.sort_values(by=date_column, ascending=True).reset_index(drop=True)
    logger.info("DataFrame successfully sorted by date.")
    return df_sorted


def handle_erroneous_duplicates(
    df: pd.DataFrame, subset_cols: List[str]
) -> pd.DataFrame:
    """Identifies and removes duplicate records based on a specific subset of columns."""
    initial_rows = len(df)
    logger.info(f"Checking for duplicates. Initial row count: {initial_rows}")
    df_cleaned = df.drop_duplicates(subset=subset_cols, keep="first")
    final_rows = len(df_cleaned)
    rows_removed = initial_rows - final_rows
    if rows_removed > 0:
        logger.info(
            f"[DuplicateHandling] Removed {rows_removed} erroneous duplicate row(s)."
        )
    else:
        logger.info("[DuplicateHandling] No erroneous duplicates found.")
    return df_cleaned


def create_date_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Creates new, memory-optimized date-based features from a datetime column.
    """
    logger.info(f"Creating date-part features from column '{date_column}'...")
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        logger.error(
            f"Column '{date_column}' is not a datetime type. Cannot create date features."
        )
        return df

    df_copy["year"] = df_copy[date_column].dt.year.astype("int16")
    df_copy["month"] = df_copy[date_column].dt.month.astype("int8")
    df_copy["day"] = df_copy[date_column].dt.day.astype("int8")
    df_copy["day_of_week"] = df_copy[date_column].dt.dayofweek.astype("int8")
    df_copy["day_of_year"] = df_copy[date_column].dt.dayofyear.astype("int16")
    df_copy["week_of_year"] = df_copy[date_column].dt.isocalendar().week.astype("int32")

    logger.info(
        "Successfully created and optimized date features: year, month, day, day_of_week, day_of_year, week_of_year."
    )
    return df_copy


def enforce_column_order(df: pd.DataFrame, column_order: List[str]) -> pd.DataFrame:
    """
    Enforces a specific column order on a DataFrame.

    This function checks if the DataFrame's columns match the expected set
    and reorders them if necessary to ensure a consistent schema.

    Args:
        df: The input DataFrame.
        column_order: A list of column names in the desired order.

    Returns:
        A new DataFrame with columns in the specified order, or the original
        DataFrame if the column sets do not match.
    """
    logger.info("Enforcing final column order...")

    # Safety check: ensure no columns are lost or unexpectedly added.
    if set(df.columns) != set(column_order):
        logger.warning(
            "Column sets do not match. Skipping reordering to prevent data loss."
        )
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        logger.debug(f"Expected columns: {column_order}")
        return df

    # Reorder the dataframe
    df_reordered = df[column_order]
    logger.info("Successfully enforced column order.")
    return df_reordered
