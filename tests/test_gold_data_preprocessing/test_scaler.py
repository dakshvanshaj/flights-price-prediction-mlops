import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Assuming the file is in src/gold_data_preprocessing/
from gold_data_preprocessing.scaler import Scaler


@pytest.fixture
def df_for_scaling() -> pd.DataFrame:
    """Provides a sample DataFrame for testing scaling strategies."""
    data = {
        "col_standard": np.array([10, 20, 30, 40, 50]),
        "col_minmax": np.array([10, 20, 30, 40, 50]),
        "col_robust": np.array([10, 20, 30, 100, 110]),  # With outliers
        "col_constant": np.array([5, 5, 5, 5, 5]),  # For zero division test
    }
    return pd.DataFrame(data)


class TestScaler:
    """
    Tests for the Scaler class.
    """

    # 1. Initialization Tests
    def test_initialization_success(self):
        """Tests successful initialization with valid arguments."""
        scaler = Scaler(columns=["col_standard"], strategy="standard")
        assert scaler.columns == ["col_standard"]
        assert scaler.strategy == "standard"

    def test_initialization_invalid_strategy(self):
        """Tests that an invalid strategy raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="Strategy must be one of 'standard', 'minmax', or 'robust'.",
        ):
            Scaler(columns=["col_standard"], strategy="invalid_strategy")

    def test_initialization_empty_columns(self):
        """Tests that an empty column list raises a ValueError."""
        with pytest.raises(
            ValueError, match="`columns` must be a non-empty list of strings."
        ):
            Scaler(columns=[])

    # 2. Fit Tests
    def test_fit_standard(self, df_for_scaling):
        """Tests that 'standard' strategy learns mean and std."""
        scaler = Scaler(columns=["col_standard"], strategy="standard")
        scaler.fit(df_for_scaling)
        assert "col_standard" in scaler.params_
        assert scaler.params_["col_standard"]["mean"] == pytest.approx(30.0)
        assert scaler.params_["col_standard"]["std"] == pytest.approx(
            df_for_scaling["col_standard"].std()
        )

    def test_fit_minmax(self, df_for_scaling):
        """Tests that 'minmax' strategy learns min and max."""
        scaler = Scaler(columns=["col_minmax"], strategy="minmax")
        scaler.fit(df_for_scaling)
        assert "col_minmax" in scaler.params_
        assert scaler.params_["col_minmax"]["min"] == 10
        assert scaler.params_["col_minmax"]["max"] == 50

    def test_fit_robust(self, df_for_scaling):
        """Tests that 'robust' strategy learns median and iqr."""
        scaler = Scaler(columns=["col_robust"], strategy="robust")
        scaler.fit(df_for_scaling)
        assert "col_robust" in scaler.params_
        assert scaler.params_["col_robust"]["median"] == 30.0
        # For [10, 20, 30, 100, 110], Q1=20, Q3=100, IQR=80
        assert scaler.params_["col_robust"]["iqr"] == 80.0

    # 3. Transform Tests
    def test_transform_standard(self, df_for_scaling):
        """Tests that 'standard' scaling results in mean ~0 and std ~1."""
        scaler = Scaler(columns=["col_standard"], strategy="standard")
        transformed_df = scaler.fit_transform(df_for_scaling)
        assert transformed_df["col_standard"].mean() == pytest.approx(0.0)
        assert transformed_df["col_standard"].std() == pytest.approx(1.0, abs=1e-1)

    def test_transform_minmax(self, df_for_scaling):
        """Tests that 'minmax' scaling results in a [0, 1] range."""
        scaler = Scaler(columns=["col_minmax"], strategy="minmax")
        transformed_df = scaler.fit_transform(df_for_scaling)
        assert transformed_df["col_minmax"].min() == pytest.approx(0.0)
        assert transformed_df["col_minmax"].max() == pytest.approx(1.0)
        assert transformed_df["col_minmax"].iloc[2] == pytest.approx(
            0.5
        )  # Middle value

    def test_transform_robust(self, df_for_scaling):
        """Tests that 'robust' scaling is applied correctly."""
        scaler = Scaler(columns=["col_robust"], strategy="robust")
        transformed_df = scaler.fit_transform(df_for_scaling)
        # (value - median) / iqr = (30 - 30) / 80 = 0
        assert transformed_df["col_robust"].iloc[2] == pytest.approx(0.0)
        # (10 - 30) / 80 = -0.25
        assert transformed_df["col_robust"].iloc[0] == pytest.approx(-20 / 80)

    def test_transform_handles_zero_division(self, df_for_scaling):
        """Tests that zero division is handled gracefully by scaling to 0."""
        for strategy in ["standard", "minmax", "robust"]:
            scaler = Scaler(columns=["col_constant"], strategy=strategy)
            transformed_df = scaler.fit_transform(df_for_scaling)
            # All values should be scaled to 0 if the divisor (std, range, iqr) is 0
            assert (transformed_df["col_constant"] == 0).all()

    def test_transform_before_fit_raises_error(self, df_for_scaling):
        """Tests that calling transform before fit raises a RuntimeError."""
        scaler = Scaler(columns=["col_standard"], strategy="standard")
        with pytest.raises(
            RuntimeError, match="Transform called before fitting the scaler."
        ):
            scaler.transform(df_for_scaling)

    # 4. Persistence Tests
    def test_save_and_load_preserves_state(self, df_for_scaling, tmp_path):
        """Tests that saving and loading the scaler preserves its state."""
        filepath = tmp_path / "scaler.joblib"
        original_scaler = Scaler(columns=["col_standard"], strategy="standard")
        original_scaler.fit(df_for_scaling)
        original_scaler.save(filepath)

        loaded_scaler = Scaler.load(filepath)
        assert loaded_scaler.params_ == original_scaler.params_
        assert loaded_scaler.strategy == original_scaler.strategy

        # Verify that the loaded scaler produces the same output
        original_transform = original_scaler.transform(df_for_scaling)
        loaded_transform = loaded_scaler.transform(df_for_scaling)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    def test_load_raises_file_not_found(self, tmp_path: Path):
        """Tests that loading from a non-existent file raises FileNotFoundError."""
        filepath = tmp_path / "non_existent_scaler.joblib"
        with pytest.raises(FileNotFoundError):
            Scaler.load(filepath)
