import pandas as pd
import logging
from typing import Dict, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleImputer:
    """
    Handles missing values using simple statistical measures (mean, median, mode)
    and constant values, following a fit/transform pattern to prevent data leakage.
    The state of the fitted imputer can be saved to and loaded from a file.
    """

    def __init__(self, strategy_dict: Dict[str, Any]):
        """
        Initializes the SimpleImputer with a configuration dictionary.

        Args:
            strategy_dict: A dictionary defining the imputation strategy.
                Example:
                {
                    'median': ['price', 'distance'],
                    'mode': ['agency'],
                    'constant': {'some_column': 'Unknown'}
                }
        """
        self.strategy_dict = strategy_dict
        self.imputers_ = {}  # Stores the learned imputation values

    def fit(self, df: pd.DataFrame):
        """
        Learns the imputation values from the training DataFrame.

        Args:
            df: The training pandas DataFrame.

        Returns:
            The fitted imputer instance (`self`).
        """
        df_copy = df.copy()
        logger.info("Fitting SimpleImputer: Learning imputation values...")

        if not self.strategy_dict:
            logger.warning(
                "Imputer strategy dictionary is empty. Fit method will do nothing."
            )
            return self

        for strategy, columns in self.strategy_dict.items():
            if strategy not in ["mean", "median", "mode", "constant"]:
                raise ValueError(
                    f"Unsupported strategy '{strategy}' found in configuration."
                )

            if strategy == "constant":
                for col, value in columns.items():
                    self.imputers_[col] = value
                    logger.debug(f"  - Learned 'constant' imputer for '{col}': {value}")
            else:
                for col in columns:
                    if col not in df_copy.columns:
                        logger.warning(
                            f"Column '{col}' not found in DataFrame during fit. Skipping."
                        )
                        continue

                    if strategy == "mean":
                        impute_value = df_copy[col].mean()
                    elif strategy == "median":
                        impute_value = df_copy[col].median()
                    else:  # mode
                        impute_value = df_copy[col].mode()[0]

                    self.imputers_[col] = impute_value
                    logger.debug(
                        f"  - Learned '{strategy}' imputer for '{col}': {impute_value}"
                    )

        logger.info("Fitting complete.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned imputation to a DataFrame.

        Args:
            df: The DataFrame to transform (train, validation, or test).

        Returns:
            A new DataFrame with missing values imputed.
        """
        if not self.imputers_:
            raise RuntimeError("Imputer has not been fitted yet. Call .fit() first.")

        df_copy = df.copy()
        logger.info("Transforming data: Applying learned imputation values...")

        for col, impute_value in self.imputers_.items():
            if col not in df_copy.columns:
                logger.warning(
                    f"Column '{col}' was not found in DataFrame during transform. Skipping."
                )
                continue

            missing_count = df_copy[col].isnull().sum()
            if missing_count > 0:
                df_copy[col].fillna(impute_value, inplace=True)
                logger.info(f"Imputed {missing_count} missing values in '{col}'.")

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the imputer and transforms the data in one step.

        Args:
            df: The training pandas DataFrame.

        Returns:
            A new DataFrame with missing values imputed.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, filepath: Path):
        """Saves the fitted imputation values to a JSON file."""
        if not self.imputers_:
            raise RuntimeError("Imputer has not been fitted. Cannot save.")

        if filepath.suffix != ".json":
            raise ValueError("Filepath must have a .json extension")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving imputer state to {filepath}...")

        imputers_to_save = {
            k: v.item() if hasattr(v, "item") else v for k, v in self.imputers_.items()
        }

        with open(filepath, "w") as f:
            json.dump(imputers_to_save, f, indent=4)
        logger.info("Imputer state saved successfully.")

    @classmethod
    def load(cls, filepath: Path):
        """Loads a pre-fitted imputer from a JSON file."""
        logger.info(f"Loading imputer state from {filepath}...")
        if not filepath.exists():
            raise FileNotFoundError(f"Imputer state file not found at {filepath}")

        with open(filepath, "r") as f:
            imputers = json.load(f)

        instance = cls(strategy_dict={})
        instance.imputers_ = imputers
        logger.info("Imputer state loaded successfully.")
        return instance
