"""
Tests for the CategoricalEncoder class in `gold_data_preprocessing.categorical_encoder`.

This test suite verifies the functionality of the `CategoricalEncoder`, a custom
transformer for one-hot and ordinal encoding of categorical features based on a
configuration dictionary.

The tests cover:
- Successful initialization and configuration validation.
- Correct application of `fit` and `transform` methods.
- Graceful handling of unseen categories during transformation.
- Error handling for invalid configurations or usage (e.g., transform before fit).
- State persistence through `save` and `load` methods.
"""

import pandas as pd
import pytest
from pathlib import Path
import numpy as np
from copy import deepcopy

# The class to be tested
from gold_data_preprocessing.categorical_encoder import CategoricalEncoder


# --- Fixtures ---


@pytest.fixture
def sample_df_for_encoding() -> pd.DataFrame:
    """
    Provides a sample DataFrame for testing the CategoricalEncoder.

    This fixture includes a variety of categorical columns suitable for testing
    different encoding strategies:
    - `airline`, `source`: For one-hot encoding.
    - `total_stops`: For mapped ordinal encoding.
    - `class`: For auto-inferred integer encoding.
    - `numeric_col`: A non-categorical column to ensure it's ignored.

    Returns:
        pd.DataFrame: A DataFrame ready for encoding tests.
    """
    data = {
        "airline": ["Indigo", "Air India", "Indigo", "Vistara"],
        "source": ["BLR", "DEL", "BLR", "BOM"],
        "total_stops": ["non-stop", "1 stop", "1 stop", "non-stop"],
        "class": ["Economy", "Business", "Economy", "Premium Economy"],
        "numeric_col": [10, 20, 30, 40],
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_encoding_config() -> dict:
    """
    Provides a valid configuration dictionary for the CategoricalEncoder.

    This configuration specifies:
    - One-hot encoding for 'airline' and 'source'.
    - Ordinal encoding for 'total_stops' with an explicit mapping.

    Returns:
        dict: A configuration dictionary for the encoder.
    """
    return {
        "onehot_cols": ["airline", "source"],
        "ordinal_cols": ["total_stops"],
        "ordinal_mapping": {"total_stops": ["non-stop", "1 stop", "2 stops"]},
    }


# --- Test Class ---


class TestCategoricalEncoder:
    """
    A test class to group all tests related to the CategoricalEncoder.
    """

    @pytest.fixture(autouse=True)
    def mock_config(self, monkeypatch):
        """
        Automatically mocks the TARGET_COLUMN config dependency for all tests.

        This ensures test isolation from the global configuration by patching the
        TARGET_COLUMN variable that the CategoricalEncoder might depend on.

        Args:
            monkeypatch: Pytest fixture for modifying variables at runtime.
        """
        monkeypatch.setattr(
            "gold_data_preprocessing.categorical_encoder.config_training.TARGET_COLUMN",
            "price",
        )

    def test_initialization_success(self, valid_encoding_config: dict):
        """
        Tests that the encoder initializes correctly with a valid configuration.

        Args:
            valid_encoding_config (dict): The fixture providing a valid encoder config.
        """
        # ARRANGE & ACT
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)

        # ASSERT
        assert encoder.onehot_cols == ["airline", "source"]
        assert encoder.ordinal_cols == ["total_stops"]
        assert not encoder._is_fitted

    def test_initialization_allows_ordinal_without_mapping(self):
        """
        Tests that initialization succeeds for an ordinal column without a mapping.
        This scenario should trigger automatic integer encoding.
        """
        # ARRANGE
        config = {"ordinal_cols": ["total_stops"]}

        # ACT & ASSERT
        try:
            CategoricalEncoder(encoding_config=config)
        except (ValueError, KeyError):
            pytest.fail(
                "Initialization failed for an ordinal column without an explicit mapping."
            )

    def test_mixed_mapped_and_auto_ordinal_encoding(
        self, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that the encoder correctly handles a mix of mapped ordinal and
        auto-inferred integer encoding in the same transformation.
        """
        # ARRANGE
        config = {
            "ordinal_cols": ["total_stops", "class"],
            "ordinal_mapping": {
                "total_stops": ["non-stop", "1 stop", "2 stops"]
            },  # 'class' mapping is missing, should be auto-encoded
        }
        encoder = CategoricalEncoder(encoding_config=config)

        # ACT
        transformed_df = encoder.fit_transform(sample_df_for_encoding)

        # ASSERT: Mapped Ordinal ('total_stops')
        # Mapping: "non-stop" -> 0, "1 stop" -> 1
        # Input: ["non-stop", "1 stop", "1 stop", "non-stop"]
        expected_ordinal = np.array([0.0, 1.0, 1.0, 0.0])
        assert np.array_equal(transformed_df["total_stops"].values, expected_ordinal)

        # ASSERT: Auto-Integer ('class')
        # Sorted categories: 'Business' -> 0, 'Economy' -> 1, 'Premium Economy' -> 2
        # Input: ["Economy", "Business", "Economy", "Premium Economy"]
        expected_integers = np.array([1.0, 0.0, 1.0, 2.0])
        assert np.array_equal(transformed_df["class"].values, expected_integers)

    def test_fit_raises_error_on_missing_columns(
        self, valid_encoding_config: dict, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that `fit` raises a ValueError if a specified encoding column
        is not present in the input DataFrame.

        Args:
            valid_encoding_config (dict): The base valid configuration.
            sample_df_for_encoding (pd.DataFrame): The input DataFrame for fitting.
        """
        # ARRANGE
        config_with_missing = deepcopy(valid_encoding_config)
        config_with_missing["onehot_cols"].append(
            "destination"
        )  # This column doesn't exist
        encoder = CategoricalEncoder(encoding_config=config_with_missing)

        # ACT & ASSERT
        with pytest.raises(ValueError) as excinfo:
            encoder.fit(sample_df_for_encoding)
        assert "The following encoding columns are not in the DataFrame" in str(
            excinfo.value
        )

    def test_mapped_ordinal_and_onehot_encoding(
        self, valid_encoding_config: dict, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that `fit_transform` correctly applies both mapped ordinal and one-hot encoding.

        Args:
            valid_encoding_config (dict): The configuration for encoding.
            sample_df_for_encoding (pd.DataFrame): The input DataFrame.
        """
        # ARRANGE
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)

        # ACT
        transformed_df = encoder.fit_transform(sample_df_for_encoding)

        # ASSERT
        # Check that non-encoded columns are preserved and new columns are created
        assert "numeric_col" in transformed_df.columns
        assert "total_stops" in transformed_df.columns
        assert "airline_Indigo" in transformed_df.columns

        # Check ordinal encoding results
        expected_ordinal = np.array([0.0, 1.0, 1.0, 0.0])
        assert np.array_equal(transformed_df["total_stops"].values, expected_ordinal)

        # Check one-hot encoding results for the first row
        assert transformed_df.loc[0, "airline_Indigo"] == 1
        assert transformed_df.loc[0, "airline_Air India"] == 0

    def test_integer_encoding_auto_infers_categories(
        self, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that ordinal columns without a mapping are auto-encoded to integers
        based on the sorted order of the unique categories.

        Args:
            sample_df_for_encoding (pd.DataFrame): The input DataFrame.
        """
        # ARRANGE
        config = {"ordinal_cols": ["class"]}
        encoder = CategoricalEncoder(encoding_config=config)

        # ACT
        transformed_df = encoder.fit_transform(sample_df_for_encoding)

        # ASSERT
        # OrdinalEncoder assigns integers based on sorted category names:
        # 'Business' -> 0, 'Economy' -> 1, 'Premium Economy' -> 2
        expected_integers = np.array([1.0, 0.0, 1.0, 2.0])
        assert "class" in transformed_df.columns
        assert np.array_equal(transformed_df["class"].values, expected_integers)

    def test_mixed_encoding_works_correctly(self, sample_df_for_encoding: pd.DataFrame):
        """
        Tests a complex scenario with one-hot, mapped ordinal, and auto-integer encoding.

        Args:
            sample_df_for_encoding (pd.DataFrame): The input DataFrame.
        """
        # ARRANGE
        config = {
            "onehot_cols": ["source"],
            "ordinal_cols": ["total_stops", "class"],  # 'class' will be auto-encoded
            "ordinal_mapping": {"total_stops": ["non-stop", "1 stop", "2 stops"]},
        }
        encoder = CategoricalEncoder(encoding_config=config)

        # ACT
        transformed_df = encoder.fit_transform(sample_df_for_encoding)

        # ASSERT: Mapped Ordinal
        expected_ordinal = np.array([0.0, 1.0, 1.0, 0.0])
        assert np.array_equal(transformed_df["total_stops"].values, expected_ordinal)

        # ASSERT: Auto-Integer
        expected_integers = np.array([1.0, 0.0, 1.0, 2.0])
        assert np.array_equal(transformed_df["class"].values, expected_integers)

        # ASSERT: One-Hot
        assert "source_BLR" in transformed_df.columns
        assert transformed_df.loc[0, "source_BLR"] == 1
        assert transformed_df.loc[1, "source_DEL"] == 1

    def test_integer_encoding_handles_unseen_categories(
        self, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that unseen categories in an auto-encoded column are mapped to -1.

        Args:
            sample_df_for_encoding (pd.DataFrame): The DataFrame for fitting.
        """
        # ARRANGE
        config = {"ordinal_cols": ["class"]}
        encoder = CategoricalEncoder(encoding_config=config)
        encoder.fit(sample_df_for_encoding)
        # Create a test DataFrame with all necessary columns, not just the one under test,
        # as the ColumnTransformer expects all columns from fit time.
        test_data = pd.DataFrame(
            {
                "airline": ["Indigo", "Vistara"],
                "source": ["BLR", "BOM"],
                "total_stops": ["non-stop", "1 stop"],
                "class": ["First Class", "Economy"],  # This is the column under test
                "numeric_col": [10, 20],
            }
        )

        # ACT
        transformed_df = encoder.transform(test_data)

        # ASSERT
        assert transformed_df["class"].iloc[0] == -1.0  # Unseen 'First Class'
        assert transformed_df["class"].iloc[1] == 1.0  # Seen 'Economy' maps to 1

    def test_transform_handles_unseen_onehot_categories_gracefully(
        self, valid_encoding_config: dict, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that unseen categories in one-hot columns are handled correctly,
        resulting in all-zero indicator columns for that observation.

        Args:
            valid_encoding_config (dict): The configuration for encoding.
            sample_df_for_encoding (pd.DataFrame): The DataFrame for fitting.
        """
        # ARRANGE
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        encoder.fit(sample_df_for_encoding)
        # Create test data with an unseen 'airline' category
        test_data = pd.DataFrame(
            {
                "airline": ["Jet Airways"],
                "source": ["BOM"],
                "total_stops": ["1 stop"],
                "class": ["Economy"],
                "numeric_col": [50],
            }
        )

        # ACT
        transformed_df = encoder.transform(test_data)

        # ASSERT
        # The new category should not create a new column, and all existing
        # 'airline' columns should be 0 for this row.
        assert "airline_Jet Airways" not in transformed_df.columns
        assert transformed_df["airline_Indigo"].iloc[0] == 0
        assert transformed_df["airline_Air India"].iloc[0] == 0
        assert transformed_df["airline_Vistara"].iloc[0] == 0

    def test_transform_before_fit_raises_error(
        self, valid_encoding_config: dict, sample_df_for_encoding: pd.DataFrame
    ):
        """
        Tests that calling `transform` before `fit` raises a RuntimeError.

        Args:
            valid_encoding_config (dict): The configuration for the encoder.
            sample_df_for_encoding (pd.DataFrame): The input DataFrame.
        """
        # ARRANGE
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)

        # ACT & ASSERT
        with pytest.raises(RuntimeError, match="Encoder has not been fitted yet."):
            encoder.transform(sample_df_for_encoding)

    def test_save_and_load_preserves_state(
        self,
        valid_encoding_config: dict,
        sample_df_for_encoding: pd.DataFrame,
        tmp_path: Path,
    ):
        """
        Tests that saving a fitted encoder and loading it back preserves its state,
        producing identical transformations.

        Args:
            valid_encoding_config (dict): The configuration for the encoder.
            sample_df_for_encoding (pd.DataFrame): The input DataFrame.
            tmp_path (Path): Pytest fixture for a temporary directory.
        """
        # ARRANGE
        filepath = tmp_path / "encoder.joblib"
        original_encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        original_encoder.fit(sample_df_for_encoding)

        # ACT
        original_encoder.save(filepath)
        loaded_encoder = CategoricalEncoder.load(filepath)

        # ASSERT
        assert loaded_encoder._is_fitted
        # Compare the transformations from the original and loaded encoders
        original_transform = original_encoder.transform(sample_df_for_encoding)
        loaded_transform = loaded_encoder.transform(sample_df_for_encoding)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    def test_load_raises_file_not_found(self, tmp_path: Path):
        """
        Tests that `load` raises FileNotFoundError when the specified file does not exist.

        Args:
            tmp_path (Path): Pytest fixture for a temporary directory.
        """
        # ARRANGE
        filepath = tmp_path / "non_existent_encoder.joblib"

        # ACT & ASSERT
        with pytest.raises(FileNotFoundError):
            CategoricalEncoder.load(filepath)
