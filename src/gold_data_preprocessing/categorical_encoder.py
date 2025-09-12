import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """
    Wraps scikit-learn's ColumnTransformer to encode categorical features
    using either one-hot or ordinal encoding, based on a configuration dictionary.
    This allows different transformations to be applied to different columns.
    """

    def __init__(self, encoding_config: Dict[str, Any]):
        """
        Initializes the CategoricalEncoder using a configuration dictionary.

        Args:
            encoding_config (Dict[str, Any]): A dictionary defining the encoding strategy.
                Example:
                {
                    'onehot_cols': ['airline', 'source'],
                    'ordinal_cols': ['total_stops'],
                    'ordinal_mapping': {
                        'total_stops': ['non-stop', '1 stop', '2 stops']
                    }
                }
        """
        self.onehot_cols = encoding_config.get("onehot_cols", [])
        self.ordinal_cols = encoding_config.get("ordinal_cols", [])
        self.ordinal_mapping = encoding_config.get("ordinal_mapping", {})
        self.preprocessor = self._create_preprocessor()
        self._is_fitted = False

    def _create_preprocessor(self) -> ColumnTransformer:
        """Dynamically builds the ColumnTransformer based on the provided config."""
        transformers = []

        if self.onehot_cols:
            logger.info(f"Setting up One-Hot encoding for: {self.onehot_cols}")
            transformers.append(
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.onehot_cols,
                )
            )

        if self.ordinal_cols:
            logger.info(f"Setting up Ordinal encoding for: {self.ordinal_cols}")
            # Separate columns with explicit mapping from those that need auto-inference
            mapped_ordinal_cols = [
                col for col in self.ordinal_cols if col in self.ordinal_mapping
            ]
            auto_ordinal_cols = [
                col for col in self.ordinal_cols if col not in self.ordinal_mapping
            ]

            if mapped_ordinal_cols:
                # Create the list of categories in the correct order for the OrdinalEncoder
                try:
                    ordered_categories = [
                        self.ordinal_mapping[col] for col in mapped_ordinal_cols
                    ]  # list of list for category's order
                except KeyError as e:
                    raise KeyError(
                        f"Missing category mapping in 'ordinal_mapping' for column: {e}"
                    )

                transformers.append(
                    (
                        "ordinal",
                        OrdinalEncoder(categories=ordered_categories),
                        mapped_ordinal_cols,
                    )
                )

            if auto_ordinal_cols:
                logger.info("Using auto-inference for Integer encoding.")
                transformers.append(
                    (
                        "integer_encoding",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        auto_ordinal_cols,
                    )
                )

        if not transformers:
            logger.warning(
                "No encoding columns were provided. Encoder will only pass through data."
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    def fit(self, df: pd.DataFrame):
        """Fits the ColumnTransformer to the data."""
        logger.info("Fitting CategoricalEncoder...")
        # Ensure all specified columns are present in the DataFrame
        all_cols = self.onehot_cols + self.ordinal_cols
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following encoding columns are not in the DataFrame: {missing_cols}"
            )

        self.preprocessor.fit(df)
        self._is_fitted = True
        logger.info("Encoder fitting complete.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted encoder and returns a DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("Encoder has not been fitted yet. Call .fit() first.")

        logger.info("Transforming data with CategoricalEncoder...")
        transformed_data = self.preprocessor.transform(df)

        # Get the new column names after transformation
        # This correctly handles one-hot, ordinal, and pass-through columns
        new_cols = self.preprocessor.get_feature_names_out()

        df_transformed = pd.DataFrame(
            transformed_data, columns=new_cols, index=df.index
        )
        logger.info(f"Transformation complete. New shape: {df_transformed.shape}")
        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the data in one step."""
        return self.fit(df).transform(df)

    def save(self, filepath: Path):
        """Saves the fitted encoder to a file using joblib."""
        if not self._is_fitted:
            raise RuntimeError("Encoder has not been fitted. Cannot save.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving encoder state to {filepath}...")
        joblib.dump(self, filepath)
        logger.info("Encoder state saved successfully.")

    @classmethod
    def load(cls, filepath: Path):
        """Loads a pre-fitted encoder from a file."""
        logger.info(f"Loading encoder state from {filepath}...")
        if not filepath.exists():
            raise FileNotFoundError(f"Encoder state file not found at {filepath}")
        return joblib.load(filepath)
