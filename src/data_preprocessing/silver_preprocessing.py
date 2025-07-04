# src/data_preprocessing/silver_preprocessing.py

import pandas as pd
import logging
import re
from typing import Dict, Optional, List, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Custom Exceptions for Clearer Error Handling ---
class ImputerNotFittedError(Exception):
    """Custom exception raised when transform() or save() is called before fit()."""

    pass


class ImputerLoadError(Exception):
    """Custom exception raised when loading a handler state fails."""

    pass


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


class MissingValueHandler:
    """
    Handles missing values in a DataFrame for a robust MLOps workflow.

    This class learns imputation values from a training set and applies them
    consistently to new data. It supports default strategies for numerical and
    categorical data, allows for column-specific overrides, and can be saved to
    and loaded from a file to ensure consistency between training and inference.
    """

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        column_strategies: Optional[Dict[str, Any]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initializes the MissingValueHandler with specified strategies.

        Args:
            numerical_strategy: Default strategy for numerical columns.
                                Supported: 'median', 'mean'.
            categorical_strategy: Default strategy for categorical columns.
                                  Supported: 'most_frequent'.
            column_strategies: Column-specific overrides.
                               Example: {'price': 'mean', 'agency': 'Unknown'}
            exclude_columns: Columns to completely ignore during imputation.
                             Example: ['travel_code', 'user_code']
        """
        if numerical_strategy not in ["median", "mean"]:
            raise ValueError("Default numerical_strategy must be 'median' or 'mean'")
        if categorical_strategy not in ["most_frequent"]:
            raise ValueError("Default categorical_strategy must be 'most_frequent'")

        self.default_numerical_strategy = numerical_strategy
        self.default_categorical_strategy = categorical_strategy
        self.column_strategies = column_strategies if column_strategies else {}
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.imputers_ = None

    def fit(self, df: pd.DataFrame):
        """
        Learns the imputation values from a DataFrame.

        Based on the defined strategies, this method calculates the imputation
        value for each column and stores it for later use.

        Args:
            df: The training DataFrame from which to learn imputation values.

        Returns:
            The fitted instance of the handler (`self`).
        """
        logger.info("Fitting MissingValueHandler: Learning imputation values...")
        self.imputers_ = {}

        for col in df.columns:
            if col in self.exclude_columns:
                logger.debug(f"Skipping column '{col}' as it is in the exclude list.")
                continue

            if col in self.column_strategies:
                strategy = self.column_strategies[col]
                if isinstance(strategy, str):
                    if strategy == "mean":
                        impute_value = df[col].mean()
                    elif strategy == "median":
                        impute_value = df[col].median()
                    elif strategy == "most_frequent":
                        impute_value = (
                            df[col].mode()[0] if not df[col].mode().empty else None
                        )
                    else:
                        impute_value = strategy
                else:
                    impute_value = strategy
            elif pd.api.types.is_numeric_dtype(df[col]):
                if self.default_numerical_strategy == "median":
                    impute_value = df[col].median()
                else:
                    impute_value = df[col].mean()
            elif pd.api.types.is_object_dtype(df[col]) or isinstance(
                df[col].dtype, pd.CategoricalDtype
            ):
                impute_value = df[col].mode()[0] if not df[col].mode().empty else None
            else:
                logger.debug(f"Skipping column '{col}' of type {df[col].dtype}.")
                continue

            if pd.isna(impute_value):
                logger.warning(
                    f"Learned imputation value for '{col}' is NaN. Skipping."
                )
                continue

            self.imputers_[col] = impute_value
            logger.debug(f"  - Learned imputer for '{col}': {impute_value}")

        logger.info("Fitting complete. Handler is ready to transform data.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in a DataFrame using the learned imputation values.

        Args:
            df: The DataFrame to transform (e.g., test set, new inference data).

        Returns:
            A new DataFrame with missing values filled.

        Raises:
            ImputerNotFittedError: If the handler has not been fitted yet.
        """
        if self.imputers_ is None:
            raise ImputerNotFittedError(
                "Handler must be fitted before transforming data."
            )

        df_copy = df.copy()
        logger.info("Transforming data: Applying learned imputation values...")

        for col, impute_value in self.imputers_.items():
            if col in df_copy.columns and df_copy[col].isnull().any():
                logger.debug(
                    f"  - Filling {df_copy[col].isnull().sum()} NaNs in '{col}'."
                )
                df_copy[col].fillna(impute_value, inplace=True)

        logger.info("Transformation complete.")
        return df_copy

    def save(self, filepath: str):
        """
        Saves the complete state of the fitted handler to a JSON file.

        This method serializes both the original configuration and the learned
        imputation values to ensure perfect reproducibility.

        Args:
            filepath: The path to the JSON file where the state will be saved.

        Raises:
            ImputerNotFittedError: If the handler has not been fitted yet.
            IOError: If there is an error writing the file to disk.
        """
        if self.imputers_ is None:
            raise ImputerNotFittedError("Handler must be fitted before saving.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "numerical_strategy": self.default_numerical_strategy,
                "categorical_strategy": self.default_categorical_strategy,
                "column_strategies": self.column_strategies,
                "exclude_columns": self.exclude_columns,
            },
            "imputers": {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in self.imputers_.items()
            },
        }

        logger.info(f"Saving handler state to '{filepath}'...")
        try:
            with open(filepath, "w") as f:
                json.dump(state, f, indent=4)
            logger.info("Handler state saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write to file '{filepath}'. Error: {e}")
            raise

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a pre-trained handler state from a JSON file.

        This classmethod reconstructs the handler with its original configuration
        and learned imputation values, ensuring consistent behavior.

        Args:
            filepath: The path to the JSON file containing the handler's state.

        Returns:
            A new, fully configured and fitted instance of MissingValueHandler.

        Raises:
            ImputerLoadError: If the file is not found or cannot be parsed.
        """
        logger.info(f"Loading handler state from '{filepath}'...")
        try:
            with open(filepath, "r") as f:
                state = json.load(f)
        except FileNotFoundError:
            raise ImputerLoadError(f"The file '{filepath}' was not found.")
        except json.JSONDecodeError as e:
            raise ImputerLoadError(
                f"Failed to decode JSON from '{filepath}'. Error: {e}"
            )

        config = state.get("config", {})
        imputers = state.get("imputers")

        if imputers is None:
            raise ImputerLoadError(
                f"Invalid state file: '{filepath}' is missing 'imputers' key."
            )

        handler = cls(
            numerical_strategy=config.get("numerical_strategy", "median"),
            categorical_strategy=config.get("categorical_strategy", "most_frequent"),
            column_strategies=config.get("column_strategies"),
            exclude_columns=config.get("exclude_columns"),
        )
        handler.imputers_ = imputers

        logger.info("Handler state loaded successfully. Ready to transform data.")
        return handler
