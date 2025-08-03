import pandas as pd
import pytest
from pathlib import Path
import numpy as np
from copy import deepcopy

# Assuming the file is in src/gold_data_preprocessing/
from gold_data_preprocessing.categorical_encoder import CategoricalEncoder


@pytest.fixture
def sample_df_for_encoding() -> pd.DataFrame:
    """
    Provides a sample DataFrame for testing the CategoricalEncoder.
    """
    data = {
        "airline": ["Indigo", "Air India", "Indigo", "Vistara"],
        "source": ["BLR", "DEL", "BLR", "BOM"],
        "total_stops": ["non-stop", "1 stop", "1 stop", "non-stop"],
        "numeric_col": [10, 20, 30, 40],
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_encoding_config() -> dict:
    """
    Provides a valid configuration for the CategoricalEncoder.
    """
    return {
        "onehot_cols": ["airline", "source"],
        "ordinal_cols": ["total_stops"],
        "ordinal_mapping": {"total_stops": ["non-stop", "1 stop", "2 stops"]},
    }


class TestCategoricalEncoder:
    """
    Tests for the CategoricalEncoder class.
    """

    def test_initialization_success(self, valid_encoding_config):
        """
        Tests successful initialization with a valid configuration.
        """
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        assert encoder.onehot_cols == ["airline", "source"]
        assert encoder.ordinal_cols == ["total_stops"]
        assert not encoder._is_fitted

    def test_initialization_missing_ordinal_mapping(self):
        """
        Tests that initialization fails if ordinal_cols are provided without ordinal_mapping.
        """
        config = {"ordinal_cols": ["total_stops"]}
        with pytest.raises(
            ValueError,
            match="'ordinal_mapping' must be provided in config for 'ordinal_cols'.",
        ):
            CategoricalEncoder(encoding_config=config)

    def test_initialization_missing_specific_mapping(self):
        """
        Tests that initialization fails if a mapping is missing for a specified ordinal column.
        """
        config = {
            "ordinal_cols": ["total_stops", "class"],
            "ordinal_mapping": {"total_stops": ["non-stop", "1 stop"]},
        }
        with pytest.raises(
            KeyError,
            match="Missing category mapping in 'ordinal_mapping' for column: 'class'",
        ):
            CategoricalEncoder(encoding_config=config)

    def test_fit_raises_error_on_missing_columns(
        self, valid_encoding_config, sample_df_for_encoding
    ):
        """
        Tests that fit raises a ValueError if specified columns are not in the DataFrame.
        """
        config_with_missing = deepcopy(valid_encoding_config)
        config_with_missing["onehot_cols"].append(
            "destination"
        )  # This col is not in the df
        encoder = CategoricalEncoder(encoding_config=config_with_missing)
        with pytest.raises(ValueError) as excinfo:
            encoder.fit(sample_df_for_encoding)

        assert "The following encoding columns are not in the DataFrame" in str(
            excinfo.value
        )
        assert "'destination'" in str(excinfo.value)

    def test_fit_transform_correctly_encodes_data(
        self, valid_encoding_config, sample_df_for_encoding
    ):
        """
        Tests that fit_transform correctly applies one-hot and ordinal encoding.
        """
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        transformed_df = encoder.fit_transform(sample_df_for_encoding)

        # Check shape and column names
        assert "numeric_col" in transformed_df.columns
        assert "total_stops" in transformed_df.columns
        assert "airline_Indigo" in transformed_df.columns
        assert "source_BLR" in transformed_df.columns
        assert (
            transformed_df.shape[1] == 1 + 1 + 3 + 3
        )  # numeric + ordinal + airline + source

        # Check ordinal encoding values
        expected_ordinal = np.array([0.0, 1.0, 1.0, 0.0])
        assert np.array_equal(transformed_df["total_stops"].values, expected_ordinal)

        # Check one-hot encoding for a specific row
        # Row 0: Indigo, BLR -> airline_Indigo=1, source_BLR=1
        assert transformed_df.loc[0, "airline_Indigo"] == 1
        assert transformed_df.loc[0, "airline_Air India"] == 0
        assert transformed_df.loc[0, "source_BLR"] == 1
        assert transformed_df.loc[0, "source_DEL"] == 0

        # Check passthrough column (ColumnTransformer may convert int to float)
        pd.testing.assert_series_equal(
            sample_df_for_encoding["numeric_col"],
            transformed_df["numeric_col"],
            check_dtype=False,
        )

    def test_transform_handles_unseen_categories_gracefully(
        self, valid_encoding_config, sample_df_for_encoding
    ):
        """
        Tests that unseen categories in one-hot columns are handled correctly.
        """
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        encoder.fit(sample_df_for_encoding)

        test_data = pd.DataFrame(
            {
                "airline": ["Jet Airways"],  # Unseen airline
                "source": ["BOM"],  # Seen source
                "total_stops": ["1 stop"],  # Seen stop
                "numeric_col": [50],
            }
        )
        transformed_df = encoder.transform(test_data)

        # Unseen category 'Jet Airways' should result in all zeros for airline columns
        assert transformed_df["airline_Indigo"].iloc[0] == 0
        assert transformed_df["airline_Air India"].iloc[0] == 0
        assert transformed_df["airline_Vistara"].iloc[0] == 0

    def test_transform_before_fit_raises_error(
        self, valid_encoding_config, sample_df_for_encoding
    ):
        """
        Tests that calling transform before fit raises a RuntimeError.
        """
        encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        with pytest.raises(RuntimeError, match="Encoder has not been fitted yet."):
            encoder.transform(sample_df_for_encoding)

    def test_save_and_load_preserves_state(
        self, valid_encoding_config, sample_df_for_encoding, tmp_path: Path
    ):
        """
        Tests that saving and loading the encoder preserves its state.
        """
        filepath = tmp_path / "encoder.joblib"
        original_encoder = CategoricalEncoder(encoding_config=valid_encoding_config)
        original_encoder.fit(sample_df_for_encoding)
        original_encoder.save(filepath)

        loaded_encoder = CategoricalEncoder.load(filepath)
        assert loaded_encoder._is_fitted

        # Compare transformed output
        original_transform = original_encoder.transform(sample_df_for_encoding)
        loaded_transform = loaded_encoder.transform(sample_df_for_encoding)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    def test_load_raises_file_not_found(self, tmp_path: Path):
        """
        Tests that loading from a non-existent file raises FileNotFoundError.
        """
        filepath = tmp_path / "non_existent_encoder.joblib"
        with pytest.raises(FileNotFoundError):
            CategoricalEncoder.load(filepath)
