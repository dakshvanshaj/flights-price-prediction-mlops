import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union
from pathlib import Path
import joblib
from scipy import stats, special

logger = logging.getLogger(__name__)


# Check if inv_yeojohnson is available, otherwise define a fallback.
# This function was added in SciPy 1.2.0.
_INV_YEOJOHNSON_AVAILABLE = hasattr(special, "inv_yeojohnson")

if not _INV_YEOJOHNSON_AVAILABLE:
    logger.debug(
        "scipy.special.inv_yeojohnson is not available in this environment. "
        "Using a custom implementation. Consider upgrading SciPy to version >= 1.2.0."
    )

    def _inv_yeojohnson(x, lmbda):
        """Custom implementation of the inverse Yeo-Johnson transformation."""
        x = np.asanyarray(x)
        y = np.empty_like(x, dtype=float)

        # Define masks for positive and negative values of x
        pos = x >= 0
        neg = ~pos

        # Positive values transformation
        if abs(lmbda) < np.finfo(float).eps:
            y[pos] = np.exp(x[pos]) - 1
        else:
            y[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # Negative values transformation
        if abs(lmbda - 2) < np.finfo(float).eps:
            y[neg] = 1 - np.exp(-x[neg])
        else:
            y[neg] = 1 - np.power(-(2 - lmbda) * x[neg] + 1, 1 / (2 - lmbda))

        return y


class PowerTransformer:
    """
    Applies power transformations to numerical columns to make their
    distribution more Gaussian-like.

    This class supports 'log', 'box-cox', and 'yeo-johnson' transformations.
    It follows the scikit-learn fit/transform pattern to prevent data leakage,
    learning the optimal transformation parameters (lambda) from the training
    data only.
    """

    def __init__(self, columns: List[str], strategy: str = "yeo-johnson"):
        """
        Initializes the PowerTransformer.

        Args:
            columns (List[str]): A list of column names to transform.
            strategy (str, optional): The transformation strategy to use.
                                      Supported: 'log', 'box-cox', 'yeo-johnson'.
                                      Defaults to 'yeo-johnson'.
        """
        # --- 1. INPUT VALIDATION ---
        # It's always a good idea to validate inputs to fail fast.
        if not isinstance(columns, list):
            raise ValueError("`columns` must be a non-empty list of strings.")
        if strategy not in ["log", "box-cox", "yeo-johnson"]:
            raise ValueError("Strategy must be one of 'log', 'box-cox', 'yeo-johnson'.")

        self.columns = columns
        self.strategy = strategy
        self._is_fitted = False
        # This dictionary will store the learned lambda values for box-cox and yeo-johnson.
        self.params_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame):
        """
        Learns the optimal transformation parameters (lambda) from the data.

        For 'log' strategy, no parameters are learned.
        For 'box-cox' and 'yeo-johnson', the optimal lambda is found and stored.

        Args:
            df (pd.DataFrame): The training DataFrame.

        Returns:
            self: The fitted transformer instance.
        """
        logger.info(f"Fitting PowerTransformer with '{self.strategy}' strategy.")
        self.params_ = {}  # Reset parameters on each new fit

        if not self.strategy or not self.columns:
            logger.info(
                "No strategy or columns provided for PowerTransformer. Skipping fit."
            )
            self._is_fitted = True
            return self

        # The 'log' transform is stateless, so we don't need to learn anything.
        if self.strategy == "log":
            logger.info("'log' strategy is stateless. No parameters were learned.")
            self._is_fitted = True
            return self

        # For 'box-cox' and 'yeo-johnson', we need to find the best lambda.
        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping fit.")
                continue

            # --- 2. BOX-COX CONSTRAINT CHECK ---
            # Box-Cox requires all data to be strictly positive.
            if self.strategy == "box-cox" and (df[col] <= 0).any():
                raise ValueError(
                    f"Column '{col}' contains non-positive values, "
                    "which is not supported by the 'box-cox' transformation. "
                    "Consider using 'yeo-johnson' instead."
                )

            # --- 3. LEARN THE LAMBDA ---
            # We use scipy.stats to find the optimal lambda.
            # The underscore '_' discards the transformed data, we only need the lambda.
            if self.strategy == "box-cox":
                _, lmbda = stats.boxcox(df[col])
            else:  # yeo-johnson
                _, lmbda = stats.yeojohnson(df[col])

            self.params_[col] = lmbda
            logger.info(f"Learned lambda for '{col}': {lmbda:.4f}")

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the power transformation using the learned parameters.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """

        if not self._is_fitted:
            raise RuntimeError("Transform called before fitting the transformer.")

        df_copy = df.copy()
        logger.info(f"Applying '{self.strategy}' transformation.")

        if not self.strategy:
            logger.info(" No strategy for PowerTransformer Provided.")
            return df_copy

        for col in self.columns:
            if col not in df_copy.columns:
                logger.warning(
                    f"Column '{col}' not found in DataFrame. Skipping transform."
                )
                continue

            # --- 4. APPLY THE TRANSFORMATION ---
            if self.strategy == "log":
                # Use log1p to handle zeros safely (log(1+x))
                df_copy[col] = np.log1p(df_copy[col])
            else:
                # Apply the transformation using the stored lambda
                lmbda = self.params_[col]
                if self.strategy == "box-cox":
                    df_copy[col] = stats.boxcox(df_copy[col], lmbda=lmbda)
                else:  # yeo-johnson
                    df_copy[col] = stats.yeojohnson(df_copy[col], lmbda=lmbda)

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """A convenience method to fit and then transform the same data."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the inverse power transformation using the learned parameters.

        Args:
            df (pd.DataFrame): The transformed DataFrame to inverse-transform.

        Returns:
            pd.DataFrame: The DataFrame with original-scale values.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Inverse transform called before fitting the transformer."
            )

        df_copy = df.copy()
        logger.info(f"Applying inverse '{self.strategy}' transformation.")

        if not self.strategy or not self.columns:
            logger.info("No strategy or columns provided. Skipping inverse transform.")
            return df_copy

        for col in self.columns:
            if col not in df_copy.columns:
                logger.warning(
                    f"Column '{col}' not found in DataFrame. Skipping inverse transform."
                )
                continue

            if self.strategy == "log":
                # Inverse of log1p is expm1
                df_copy[col] = np.expm1(df_copy[col])
            else:
                if col not in self.params_:
                    logger.warning(
                        f"Column '{col}' was not transformed. Skipping inverse transform."
                    )
                    continue

                lmbda = self.params_[col]
                if self.strategy == "box-cox":
                    df_copy[col] = special.inv_boxcox(df_copy[col], lmbda)
                else:  # yeo-johnson
                    if _INV_YEOJOHNSON_AVAILABLE:
                        df_copy[col] = special.inv_yeojohnson(df_copy[col], lmbda)
                    else:
                        df_copy[col] = _inv_yeojohnson(df_copy[col], lmbda)

        return df_copy

    def save(self, filepath: Union[str, Path]):
        """Saves the fitted transformer instance to a file."""
        logger.info(f"Saving transformer to {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Union[str, Path]):
        """Loads a transformer instance from a file."""
        logger.info(f"Loading transformer from {filepath}")
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Transformer file not found at {filepath}")
        return joblib.load(filepath)
