import pandas as pd
import logging
import re
from typing import Dict, Optional, List
import json
from pathlib import Path

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


def handle_erroneous_duplicates(
    df: pd.DataFrame, subset_cols: List[str]
) -> pd.DataFrame:
    """
    Identifies and removes duplicate records based on a specific subset of columns.

    This function is designed to remove true data errors (e.g., the same user
    recorded twice for the same flight) while preserving valid records (e.g.,
    different users on the same flight).

    Args:
        df: The input DataFrame.
        subset_cols: A list of column names that define a unique record.

    Returns:
        A DataFrame with erroneous duplicates removed.
    """
    # Log the number of rows before cleaning for context.
    initial_rows = len(df)
    logger.info(f"Checking for duplicates. Initial row count: {initial_rows}")

    # Use drop_duplicates with the specified subset and keep='first'.
    # This is safer than using inplace=True as it returns a new DataFrame.
    df_cleaned = df.drop_duplicates(subset=subset_cols, keep="first")

    # Log how many duplicate rows were removed.
    final_rows = len(df_cleaned)
    rows_removed = initial_rows - final_rows

    if rows_removed > 0:
        logger.info(
            f"[DuplicateHandling] Removed {rows_removed} erroneous duplicate row(s)."
        )
    else:
        logger.info("[DuplicateHandling] No erroneous duplicates found.")

    return df_cleaned


class MissingValueHandler:
    """
    A class to handle missing values in a DataFrame, designed for a robust MLOps
    workflow. It learns imputation values from a training set and applies them
    consistently to new data, supporting both general and column-specific strategies.
    """

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        column_strategies: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the MissingValueHandler.

        Args:
            numerical_strategy (str): The default strategy for all numerical columns.
                                      Supported: 'median' (default), 'mean'.
            categorical_strategy (str): The default strategy for all categorical columns.
                                        Supported: 'most_frequent' (default).
            column_strategies (dict, optional): A dictionary to specify strategies for
                                                individual columns. This overrides the default.
                                                Example: {'price': 'mean', 'agency': 'Unknown', 'distance': 0}
                                                any key other than 'mean', 'median', or 'most_frequent' will be treated as a constant fill value.
        """
        if numerical_strategy not in ["median", "mean"]:
            raise ValueError("Default numerical_strategy must be 'median' or 'mean'")
        if categorical_strategy not in ["most_frequent"]:
            raise ValueError("Default categorical_strategy must be 'most_frequent'")

        self.default_numerical_strategy = numerical_strategy
        self.default_categorical_strategy = categorical_strategy
        self.column_strategies = column_strategies if column_strategies else {}
        self.imputers_ = None

    def fit(self, df: pd.DataFrame):
        """
        Learns the imputation values from the training DataFrame based on the
        defined strategies and stores them in the `imputers_` attribute.

        Column-specific strategies in `column_strategies` will take precedence
        over the default strategies.

        Args:
            df (pd.DataFrame): The training DataFrame to learn from.

        Returns:
            self: The instance of the class itself.
        """
        logger.info("Fitting MissingValueHandler: Learning imputation values...")
        self.imputers_ = {}

        for col in df.columns:
            # --- Step 1: Check for a column-specific strategy first ---
            if col in self.column_strategies:
                strategy = self.column_strategies[col]
                if isinstance(strategy, str) and strategy == "mean":
                    impute_value = df[col].mean()
                elif isinstance(strategy, str) and strategy == "median":
                    impute_value = df[col].median()
                elif isinstance(strategy, str) and strategy == "most_frequent":
                    impute_value = (
                        df[col].mode()[0] if not df[col].mode().empty else None
                    )
                else:
                    # If it's not a recognized keyword, treat it as a constant fill value
                    impute_value = strategy
            # --- Step 2: If no column specific strategy, use the default based on dtype ---
            elif pd.api.types.is_numeric_dtype(df[col]):
                if self.default_numerical_strategy == "median":
                    impute_value = df[col].median()
                else:  # 'mean'
                    impute_value = df[col].mean()
            elif pd.api.types.is_object_dtype(
                df[col]
            ) or pd.api.types.is_categorical_dtype(df[col]):
                impute_value = df[col].mode()
                if not impute_value.empty:
                    impute_value = impute_value[0]
                else:
                    logger.warning(f"Column '{col}' has no mode; skipping imputation.")
                    continue
            else:
                logger.debug(
                    f"Skipping column '{col}' as it is not numeric, categorical, or user-specified."
                )
                continue

            # --- Final check before storing the learned value ---
            if pd.isna(impute_value):
                logger.warning(
                    f"Learned imputation value for column '{col}' is NaN. Skipping this column."
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
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: A new DataFrame with missing values imputed.
        """
        if self.imputers_ is None:
            logger.error(
                "FitError: The handler has not been fitted yet. Call fit() before transform()."
            )
            raise RuntimeError("You must call fit() before calling transform().")

        df_copy = df.copy()
        logger.info("Transforming data: Applying learned imputation values...")

        for col, impute_value in self.imputers_.items():
            if col in df_copy.columns and df_copy[col].isnull().any():
                logger.debug(
                    f"  - Filling {df_copy[col].isnull().sum()} NaNs in '{col}' with '{impute_value}'."
                )
                df_copy[col].fillna(impute_value, inplace=True)

        logger.info("Transformation complete.")
        return df_copy

    def save(self, filepath: str):
        """
        Saves the learned imputers to a JSON file for later use.

        Args:
            filepath (str): The path to the file where the imputer state will be saved.
        """
        if self.imputers_ is None:
            logger.error(
                "SaveError: The handler has not been fitted. Cannot save an empty state."
            )
            raise RuntimeError("You must fit the handler before saving.")

        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory for '{filepath}'. Error: {e}")
            raise

        imputers_serializable = {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in self.imputers_.items()
        }

        logger.info(f"Saving handler state to '{filepath}'...")
        logger.debug(f"Saving the following imputation values: {imputers_serializable}")
        try:
            with open(filepath, "w") as f:
                json.dump(imputers_serializable, f, indent=4)
            logger.info("Handler state saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write to file '{filepath}'. Error: {e}")
            raise

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a pre-trained handler state from a JSON file.

        Args:
            filepath (str): The path to the saved imputer JSON file.

        Returns:
            MissingValueHandler: An instance of the class with its state loaded.
        """
        logger.info(f"Loading handler state from '{filepath}'...")
        try:
            with open(filepath, "r") as f:
                imputer_state = json.load(f)
                logger.debug(f"Loaded the following imputation values: {imputer_state}")
        except FileNotFoundError:
            logger.error(f"LoadError: The file '{filepath}' was not found.")
            raise
        except json.JSONDecodeError as e:
            logger.error(
                f"LoadError: Failed to decode JSON from '{filepath}'. Error: {e}"
            )
            raise

        handler = cls.__new__(cls)
        handler.imputers_ = imputer_state
        handler.default_numerical_strategy = None
        handler.default_categorical_strategy = None
        handler.column_strategies = None

        logger.info("Handler state loaded successfully. Ready to transform data.")
        return handler
