import pandas as pd
import logging
from typing import List, Dict, Union, Any
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest

# It's good practice to get a logger specific to this module.
logger = logging.getLogger(__name__)


class OutlierTransformer:
    """
    A transformer to detect and handle outliers in numerical columns.

    This class follows the scikit-learn fit/transform pattern. It learns the
    outlier boundaries or fits outlier detection models from the training
    data (`fit`) and then applies the chosen handling strategy to any
    dataset (`transform`).

    This approach prevents data leakage by ensuring that the logic is
    determined only once from the training set.

    Attributes:
        detection_strategy (str): The method to use for detecting outliers.
                                  Supported: 'iqr', 'zscore', 'isolation_forest'.
        handling_strategy (str): The method to use for handling outliers.
                                 Supported: 'winsorize' (capping), 'trim' (removing).
        columns (List[str]): A list of column names to apply the transformation to.
        ... (other parameters)
        bounds_ (Dict[str, Dict[str, float]]): Stores learned bounds for 'iqr' and 'zscore'.
        models_ (Dict[str, Any]): Stores fitted models for 'isolation_forest'.
    """

    def __init__(
        self,
        columns: List[str],
        detection_strategy: str = "iqr",
        handling_strategy: str = "winsorize",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        **iso_forest_kwargs,
    ):
        """
        Initializes the OutlierTransformer with specified strategies and parameters.

        Args:
            ...
            **iso_forest_kwargs: Keyword arguments to pass to the IsolationForest model.
                                 e.g., contamination=0.1, random_state=42.
        """
        if not isinstance(columns, list) or not columns:
            raise ValueError("`columns` must be a non-empty list of strings.")
        if detection_strategy not in ["iqr", "zscore", "isolation_forest"]:
            raise ValueError(
                "`detection_strategy` must be 'iqr', 'zscore', or 'isolation_forest'."
            )
        if handling_strategy not in ["winsorize", "trim"]:
            raise ValueError("`handling_strategy` must be 'winsorize' or 'trim'.")
        if detection_strategy == "isolation_forest" and handling_strategy != "trim":
            raise ValueError(
                "IsolationForest detection is only compatible with 'trim' handling."
            )

        self.columns = columns
        self.detection_strategy = detection_strategy
        self.handling_strategy = handling_strategy
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.iso_forest_kwargs = iso_forest_kwargs

        self.bounds_: Dict[str, Dict[str, float]] = {}
        self.models_: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame):
        """
        Learns outlier boundaries or fits outlier detection models.
        """
        logger.info(
            f"Fitting OutlierTransformer with '{self.detection_strategy}' strategy."
        )
        self.bounds_ = {}
        self.models_ = {}

        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping fit.")
                continue

            if self.detection_strategy in ["iqr", "zscore"]:
                if self.detection_strategy == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.iqr_multiplier * IQR
                    upper_bound = Q3 + self.iqr_multiplier * IQR
                else:  # zscore
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - self.zscore_threshold * std
                    upper_bound = mean + self.zscore_threshold * std
                self.bounds_[col] = {"lower": lower_bound, "upper": upper_bound}
                logger.info(f"Learned bounds for '{col}': {self.bounds_[col]}")

            elif self.detection_strategy == "isolation_forest":
                model = IsolationForest(**self.iso_forest_kwargs)
                model.fit(df[[col]])
                self.models_[col] = model
                logger.info(f"Fitted IsolationForest model for '{col}'.")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles outliers in the DataFrame based on the fitted state and the
        detection strategy used during fit.
        """
        if not self.bounds_ and not self.models_:
            raise RuntimeError("Transform called before fitting the transformer.")

        df_copy = df.copy()
        logger.info(
            f"Transforming data using '{self.detection_strategy}' detection "
            f"and '{self.handling_strategy}' handling."
        )

        if self.detection_strategy in ["iqr", "zscore"]:
            if self.handling_strategy == "trim":
                initial_rows = len(df_copy)
                outlier_mask = self.predict_outliers(df_copy)
                # Keep rows where NO column is flagged as an outlier
                keep_mask = ~outlier_mask.any(axis=1)
                df_copy = df_copy[keep_mask]
                rows_removed = initial_rows - len(df_copy)
                logger.info(
                    f"Trimmed {rows_removed} rows based on {self.detection_strategy} bounds."
                )

            elif self.handling_strategy == "winsorize":
                for col, bounds in self.bounds_.items():
                    if col in df_copy.columns:
                        df_copy[col] = df_copy[col].clip(
                            lower=bounds["lower"], upper=bounds["upper"]
                        )
                logger.info("Capped outliers using winsorization.")

        elif self.detection_strategy == "isolation_forest":
            initial_rows = len(df_copy)
            outlier_mask = self.predict_outliers(df_copy)
            keep_mask = ~outlier_mask.any(axis=1)
            df_copy = df_copy[keep_mask]
            rows_removed = initial_rows - len(df_copy)
            logger.info(f"Trimmed {rows_removed} rows based on IsolationForest models.")

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """A convenience method to fit and then transform the same data."""
        return self.fit(df).transform(df)

    def predict_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each specified column, predicts whether each row is an outlier.

        This method is useful for inspection and visualization, showing exactly
        which data points are flagged as outliers by the fitted strategy.

        Args:
            df (pd.DataFrame): The DataFrame to predict on.

        Returns:
            pd.DataFrame: A DataFrame with boolean values, where True indicates
                          an outlier. The shape matches the input DataFrame,
                          but it only contains the columns the transformer was
                          fitted on.
        """
        if not self.bounds_ and not self.models_:
            raise RuntimeError("Predict called before fitting the transformer.")

        # Ensure we only work with columns the transformer knows about.
        cols_to_check = [col for col in self.columns if col in df.columns]
        outlier_df = pd.DataFrame(index=df.index)

        logger.info(f"Predicting outliers using '{self.detection_strategy}' strategy.")

        if self.detection_strategy in ["iqr", "zscore"]:
            for col in cols_to_check:
                bounds = self.bounds_[col]
                mask = (df[col] < bounds["lower"]) | (df[col] > bounds["upper"])
                outlier_df[col] = mask

        elif self.detection_strategy == "isolation_forest":
            for col in cols_to_check:
                model = self.models_[col]
                predictions = model.predict(df[[col]])
                mask = predictions == -1  # Outliers are -1
                outlier_df[col] = mask

        return outlier_df

    def save(self, filepath: Union[str, Path]):
        """Saves the fitted transformer instance to a file using joblib."""
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
