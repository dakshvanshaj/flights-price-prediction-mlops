import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from scipy import stats

# Assuming the file is in src/gold_data_preprocessing/
from gold_data_preprocessing.power_transformer import PowerTransformer


@pytest.fixture
def df_positive_skew() -> pd.DataFrame:
    """Provides a DataFrame with positive, skewed data suitable for all transformations."""
    # Using a log-normal distribution to create skewed data
    np.random.seed(42)
    data = {
        "col_pos": np.random.lognormal(mean=2, sigma=1, size=100),
        "col_mixed": np.random.randn(100) * 5
        + 10,  # Contains positive and potentially negative
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_with_non_positive() -> pd.DataFrame:
    """Provides a DataFrame with zero and negative values to test Box-Cox constraints."""
    return pd.DataFrame({"col_non_pos": [1, 2, 0, -1, 5]})


class TestPowerTransformer:
    """
    Tests for the PowerTransformer class.
    """

    # 1. Initialization Tests
    def test_initialization_success(self):
        """Tests successful initialization with valid arguments."""
        transformer = PowerTransformer(columns=["col_pos"], strategy="yeo-johnson")
        assert transformer.columns == ["col_pos"]
        assert transformer.strategy == "yeo-johnson"

    def test_initialization_invalid_strategy(self):
        """Tests that an invalid strategy raises a ValueError."""
        with pytest.raises(
            ValueError, match="Strategy must be one of 'log', 'box-cox', 'yeo-johnson'."
        ):
            PowerTransformer(columns=["col_pos"], strategy="invalid_strategy")

    def test_initialization_empty_columns(self):
        """Tests that an empty column list raises a ValueError."""
        with pytest.raises(
            ValueError, match="`columns` must be a non-empty list of strings."
        ):
            PowerTransformer(columns=[])

    # 2. Fit Tests
    def test_fit_log_is_stateless(self, df_positive_skew):
        """Tests that the 'log' strategy does not learn any parameters."""
        transformer = PowerTransformer(columns=["col_pos"], strategy="log")
        transformer.fit(df_positive_skew)
        assert not transformer.params_  # params_ should be empty

    def test_fit_boxcox_learns_lambda(self, df_positive_skew):
        """Tests that 'box-cox' correctly learns and stores the lambda parameter."""
        transformer = PowerTransformer(columns=["col_pos"], strategy="box-cox")
        transformer.fit(df_positive_skew)
        assert "col_pos" in transformer.params_
        assert isinstance(transformer.params_["col_pos"], float)

    def test_fit_yeojohnson_learns_lambda(self, df_positive_skew):
        """Tests that 'yeo-johnson' correctly learns and stores the lambda parameter."""
        transformer = PowerTransformer(columns=["col_mixed"], strategy="yeo-johnson")
        transformer.fit(df_positive_skew)
        assert "col_mixed" in transformer.params_
        assert isinstance(transformer.params_["col_mixed"], float)

    def test_fit_boxcox_raises_error_on_non_positive_data(self, df_with_non_positive):
        """Tests that 'box-cox' raises a ValueError when data is not strictly positive."""
        transformer = PowerTransformer(columns=["col_non_pos"], strategy="box-cox")
        with pytest.raises(ValueError, match="contains non-positive values"):
            transformer.fit(df_with_non_positive)

    # 3. Transform Tests
    def test_transform_log(self, df_positive_skew):
        """Tests that the 'log' transformation is applied correctly."""
        transformer = PowerTransformer(columns=["col_pos"], strategy="log")
        transformer.fit(df_positive_skew)
        transformed_df = transformer.transform(df_positive_skew)
        expected = np.log1p(df_positive_skew["col_pos"])
        pd.testing.assert_series_equal(transformed_df["col_pos"], expected)

    def test_transform_boxcox(self, df_positive_skew):
        """Tests that 'box-cox' transformation is applied correctly using the stored lambda."""
        transformer = PowerTransformer(columns=["col_pos"], strategy="box-cox")
        transformer.fit(df_positive_skew)
        lmbda = transformer.params_["col_pos"]
        transformed_df = transformer.transform(df_positive_skew)
        expected = pd.Series(
            stats.boxcox(df_positive_skew["col_pos"], lmbda=lmbda), name="col_pos"
        )
        pd.testing.assert_series_equal(transformed_df["col_pos"], expected)

    def test_transform_yeojohnson(self, df_positive_skew):
        """Tests that 'yeo-johnson' transformation is applied correctly using the stored lambda."""
        transformer = PowerTransformer(columns=["col_mixed"], strategy="yeo-johnson")
        transformer.fit(df_positive_skew)
        lmbda = transformer.params_["col_mixed"]
        transformed_df = transformer.transform(df_positive_skew)
        expected = pd.Series(
            stats.yeojohnson(df_positive_skew["col_mixed"], lmbda=lmbda),
            name="col_mixed",
        )
        pd.testing.assert_series_equal(transformed_df["col_mixed"], expected)

    def test_transform_before_fit_raises_error(self, df_positive_skew):
        """Tests that calling transform before fit raises a RuntimeError for stateful strategies."""
        transformer = PowerTransformer(columns=["col_pos"], strategy="yeo-johnson")
        with pytest.raises(
            RuntimeError, match="Transform called before fitting the transformer."
        ):
            transformer.transform(df_positive_skew)

    # 4. Fit-Transform and Persistence
    def test_fit_transform_is_consistent(self, df_positive_skew):
        """Tests that fit_transform gives the same result as fit then transform."""
        # Yeo-Johnson
        transformer1 = PowerTransformer(columns=["col_mixed"], strategy="yeo-johnson")
        transformer2 = PowerTransformer(columns=["col_mixed"], strategy="yeo-johnson")

        # Method 1: fit_transform
        transformed1 = transformer1.fit_transform(df_positive_skew)

        # Method 2: fit, then transform
        transformer2.fit(df_positive_skew)
        transformed2 = transformer2.transform(df_positive_skew)

        pd.testing.assert_frame_equal(transformed1, transformed2)

    def test_save_and_load_preserves_state(self, df_positive_skew, tmp_path):
        """Tests that saving and loading the transformer preserves its state."""
        filepath = tmp_path / "power_transformer.joblib"
        original_transformer = PowerTransformer(
            columns=["col_mixed"], strategy="yeo-johnson"
        )
        original_transformer.fit(df_positive_skew)
        original_transformer.save(filepath)

        loaded_transformer = PowerTransformer.load(filepath)
        assert loaded_transformer.params_ == original_transformer.params_
        assert loaded_transformer.strategy == original_transformer.strategy

        # Verify that the loaded transformer produces the same output
        original_transform = original_transformer.transform(df_positive_skew)
        loaded_transform = loaded_transformer.transform(df_positive_skew)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    def test_load_raises_file_not_found(self, tmp_path: Path):
        """Tests that loading from a non-existent file raises FileNotFoundError."""
        filepath = tmp_path / "non_existent_transformer.joblib"
        with pytest.raises(FileNotFoundError):
            PowerTransformer.load(filepath)
