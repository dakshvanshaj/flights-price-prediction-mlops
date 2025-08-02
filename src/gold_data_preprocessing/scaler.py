import pandas as pd
import logging
from typing import List, Dict, Union
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class Scaler:
    """
    Applies scaling to numerical columns to bring them to a common scale.

    This class supports 'standard', 'minmax', and 'robust' scaling strategies.
    It follows the scikit-learn fit/transform pattern to prevent data leakage
    by learning the scaling parameters (e.g., mean/std, min/max, or median/iqr)
    from the training data only.
    """

    def __init__(self, columns: List[str], strategy: str = "standard"):
        """
        Initializes the Scaler.

        Args:
            columns (List[str]): A list of column names to scale.
            strategy (str, optional): The scaling strategy to use.
                                      Supported: 'standard', 'minmax', 'robust'.
                                      Defaults to 'standard'.
        """
        # --- 1. INPUT VALIDATION ---
        if not isinstance(columns, list) or not columns:
            raise ValueError("`columns` must be a non-empty list of strings.")
        if strategy not in ["standard", "minmax", "robust"]:
            raise ValueError(
                "Strategy must be one of 'standard', 'minmax', or 'robust'."
            )

        self.columns = columns
        self.strategy = strategy

        # This dictionary will store the learned scaling parameters for each column.
        self.params_: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame):
        """
        Learns the scaling parameters from the training data.

        Args:
            df (pd.DataFrame): The training DataFrame.

        Returns:
            self: The fitted scaler instance.
        """
        logger.info(f"Fitting Scaler with '{self.strategy}' strategy.")
        self.params_ = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping fit.")
                continue

            # --- 2. LEARN SCALING PARAMETERS ---
            if self.strategy == "standard":
                mean = df[col].mean()
                std = df[col].std()
                self.params_[col] = {"mean": mean, "std": std}
                logger.info(
                    f"Learned params for '{col}': mean={mean:.4f}, std={std:.4f}"
                )

            elif self.strategy == "minmax":
                # --- FIX: Use descriptive variable names ---
                min_val = df[col].min()
                max_val = df[col].max()
                self.params_[col] = {"min": min_val, "max": max_val}
                logger.info(
                    f"Learned params for '{col}': min={min_val:.4f}, max={max_val:.4f}"
                )

            elif self.strategy == "robust":
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                self.params_[col] = {"median": median, "iqr": iqr}
                logger.info(
                    f"Learned params for '{col}': median={median:.4f}, iqr={iqr:.4f}"
                )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned scaling transformation to the data.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        if not self.params_:
            raise RuntimeError("Transform called before fitting the scaler.")

        df_copy = df.copy()
        logger.info(f"Applying '{self.strategy}' scaling.")

        for col in self.columns:
            if col not in df_copy.columns:
                logger.warning(
                    f"Column '{col}' not found in DataFrame. Skipping transform."
                )
                continue

            params = self.params_[col]
            if self.strategy == "standard":
                # Handle case where standard deviation is zero
                if params["std"] > 1e-8:
                    df_copy[col] = (df_copy[col] - params["mean"]) / params["std"]
                else:
                    df_copy[col] = 0
                    logger.warning(
                        f"Standard deviation for '{col}' is zero. Scaled to 0."
                    )

            elif self.strategy == "minmax":
                range_val = params["max"] - params["min"]
                # Handle case where min and max are the same
                if range_val > 1e-8:
                    df_copy[col] = (df_copy[col] - params["min"]) / range_val
                else:
                    df_copy[col] = 0
                    logger.warning(f"Range for '{col}' is zero. Scaled to 0.")

            elif self.strategy == "robust":
                # Handle case where IQR is zero
                if params["iqr"] > 1e-8:
                    df_copy[col] = (df_copy[col] - params["median"]) / params["iqr"]
                else:
                    df_copy[col] = 0
                    logger.warning(f"IQR for '{col}' is zero. Scaled to 0.")

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """A convenience method to fit and then transform the same data."""
        # --- FIX: Corrected typo from 'tranform' to 'transform' ---
        return self.fit(df).transform(df)

    def save(self, filepath: Union[str, Path]):
        """Saves the fitted scaler instance to a file."""
        logger.info(f"Saving scaler to {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Union[str, Path]):
        """Loads a scaler instance from a file."""
        logger.info(f"Loading scaler from {filepath}")
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Scaler file not found at {filepath}")
        return joblib.load(filepath)
