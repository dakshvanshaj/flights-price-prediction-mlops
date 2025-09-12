import pandas as pd
import logging
from typing import List, Dict
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class RareCategoryGrouper:
    """
    Groups infrequent categorical values into a single 'Other' category.

    This transformer learns which categories are common from the training data
    based on a frequency threshold and applies this rule consistently across
    all datasets to prevent overfitting and manage high cardinality.
    """

    def __init__(self, columns: List[str], threshold: float = 0.01):
        """
        Initializes the RareCategoryGrouper.

        Args:
            columns (List[str]): The list of categorical columns to process.
            threshold (float): The minimum frequency for a category to be kept.
                               Categories with a frequency below this will be
                               grouped into 'Other'.
        """
        if not (0 < threshold < 1):
            raise ValueError("Threshold must be a float between 0 and 1.")
        self.columns = columns
        self.threshold = threshold
        self.frequent_categories_map_: Dict[str, list] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Learns the frequent categories from the training DataFrame.
        """
        logger.info("Fitting RareCategoryGrouper...")
        df_copy = df.copy()

        if not self.columns:
            logger.info("No columns specified for fitting.")
            self._is_fitted = True
            logger.info("Fitting complete.")
            return self

        for col in self.columns:
            if col not in df_copy.columns:
                logger.warning(f"Column '{col}' not in DataFrame. Skipping.")
                continue

            # Calculate category frequencies as a percentage
            counts = df_copy[col].value_counts(normalize=True)
            # Identify categories that meet the frequency threshold
            frequent_categories = counts[counts >= self.threshold].index.tolist()

            self.frequent_categories_map_[col] = frequent_categories
            logger.info(
                f"For column '{col}', found {len(frequent_categories)} frequent categories "
                f"(out of {len(counts)}) to keep based on a >={self.threshold:.2%} threshold."
            )

        self._is_fitted = True
        logger.info("Fitting complete.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the grouping of rare categories to a DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Grouper has not been fitted yet. Call .fit() first.")

        logger.info("Transforming data with RareCategoryGrouper...")
        df_copy = df.copy()

        if not self.columns:
            logger.info("No columns specified for transformation.")
            return df_copy

        for col, frequent_categories in self.frequent_categories_map_.items():
            if col not in df_copy.columns:
                continue

            # Replace categories that are not in the learned frequent list with 'Other'
            df_copy[col] = df_copy[col].where(
                df_copy[col].isin(frequent_categories), "Other"
            )
            logger.info(f"Grouped rare categories for column '{col}'.")

        logger.info("Transformation complete.")
        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the data in one step."""
        return self.fit(df).transform(df)

    def save(self, filepath: Path):
        """Saves the fitted grouper to a file using joblib."""
        if not self._is_fitted:
            raise RuntimeError("Grouper has not been fitted. Cannot save.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"RareCategoryGrouper state saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path):
        """Loads a pre-fitted grouper from a file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Grouper state file not found at {filepath}")
        grouper = joblib.load(filepath)
        logger.info(f"RareCategoryGrouper state loaded from {filepath}")
        return grouper
