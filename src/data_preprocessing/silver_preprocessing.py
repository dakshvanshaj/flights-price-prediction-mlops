import pandas as pd
import logging
import re
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def rename_specific_columns(
    df: pd.DataFrame, rename_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Renames specific columns in a DataFrame based on a provided mapping.
    """
    df = df.copy()
    df.rename(columns=rename_mapping, inplace=True)
    logger.info("Applied specific column renaming.")
    return df


def standardize_column_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes all column names in a DataFrame to a consistent format.
    """
    df = df.copy()
    clean_cols = []
    for col in df.columns:
        # Converts to snake_case, e.g., 'flightType' -> 'flight_type'
        standardized_col = (
            re.sub(r"(?<!^)(?=[A-Z])", "_", col.strip()).lower().replace(" ", "_")
        )
        clean_cols.append(standardized_col)
    df.columns = clean_cols
    logger.info("Standardized column name formats for consistency.")
    logger.debug(f"Final standardized columns: {', '.join(df.columns)}")
    return df


def optimize_data_types(
    df: pd.DataFrame, date_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Optimizes data types and robustly parses date columns to reduce memory usage.
    """
    df = df.copy()

    initial_mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory Usage Before Optimization: {initial_mem_usage:.2f} MB")
    # This detailed log is useful for deep dives but verbose for normal runs.
    logger.debug(f"Initial dtypes:\n{df.dtypes}")

    # Prioritized list of common date formats for efficient parsing.
    COMMON_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]

    if date_cols is None:
        date_cols = []

    for col in df.columns:
        # Store original dtype for logging comparison.
        old_dtype = df[col].dtype

        if col in date_cols and old_dtype == "object":
            # This block attempts to parse specified date columns.
            parsed_successfully = False
            for fmt in COMMON_DATE_FORMATS:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt)
                    parsed_successfully = True
                    break
                except (ValueError, TypeError):
                    continue

            if not parsed_successfully:
                # As a last resort, try pandas' slow but powerful inference.
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

        # --- Standard Type Optimization Logic ---
        if str(old_dtype).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
            new_dtype = df[col].dtype
            if new_dtype != old_dtype:
                logger.info(
                    f"[DataTypeOpt] Converted '{col}': {old_dtype} -> {new_dtype}"
                )

        elif str(old_dtype).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
            new_dtype = df[col].dtype
            if new_dtype != old_dtype:
                logger.info(
                    f"[DataTypeOpt] Converted '{col}': {old_dtype} -> {new_dtype}"
                )

        elif old_dtype == "object":
            # Heuristic: convert low-cardinality strings to 'category' for memory savings.
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")
                new_dtype = df[col].dtype
                logger.info(
                    f"[DataTypeOpt] Converted '{col}': {old_dtype} -> {new_dtype}"
                )

    final_mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    percent_reduction = (initial_mem_usage - final_mem_usage) / initial_mem_usage * 100

    logger.info(f"Memory Usage After Optimization: {final_mem_usage:.2f} MB")
    logger.info(f"Memory reduced by {percent_reduction:.2f}%.")

    return df
