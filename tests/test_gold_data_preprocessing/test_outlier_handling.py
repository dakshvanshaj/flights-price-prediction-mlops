import pandas as pd
import numpy as np
import pytest


# Assuming the file is in src/gold_data_preprocessing/
from gold_data_preprocessing.outlier_handling import OutlierTransformer


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """
    Provides a sample DataFrame with obvious outliers for testing.
    - 'col1' has clear outliers at both ends for IQR testing.
    - 'col2' is normally distributed with a very distant outlier.
    """
    data = {
        "col1": [1, 50, 52, 55, 58, 60, 62, 65, 68, 70, 150],
        "col2": list(np.random.normal(loc=100, scale=10, size=10)) + [500],
        "col3": range(11),  # A column that should be ignored
    }
    return pd.DataFrame(data)


class TestOutlierTransformer:
    """
    Tests for the OutlierTransformer class.
    """

    # 1. Initialization Tests
    def test_initialization_success(self):
        """Tests that the transformer initializes correctly with valid arguments."""
        transformer = OutlierTransformer(columns=["col1"], detection_strategy="iqr")
        assert transformer.columns == ["col1"]
        assert transformer.detection_strategy == "iqr"

    def test_initialization_invalid_detection_strategy(self):
        """Tests that an invalid detection strategy raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="`detection_strategy` must be 'iqr', 'zscore', or 'isolation_forest'.",
        ):
            OutlierTransformer(columns=["col1"], detection_strategy="invalid_strategy")

    def test_initialization_invalid_handling_strategy(self):
        """Tests that an invalid handling strategy raises a ValueError."""
        with pytest.raises(
            ValueError, match="`handling_strategy` must be 'winsorize' or 'trim'."
        ):
            OutlierTransformer(columns=["col1"], handling_strategy="invalid_strategy")

    def test_initialization_incompatible_strategies(self):
        """Tests that combining isolation_forest with winsorize raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="IsolationForest detection is only compatible with 'trim' handling.",
        ):
            OutlierTransformer(
                columns=["col1"],
                detection_strategy="isolation_forest",
                handling_strategy="winsorize",
            )

    # 2. Fit Tests
    def test_fit_iqr(self, df_with_outliers):
        """Tests that the IQR bounds are calculated correctly during fit."""
        transformer = OutlierTransformer(columns=["col1"], detection_strategy="iqr")
        transformer.fit(df_with_outliers)
        assert "col1" in transformer.bounds_
        # Manual calculation for col1 using pandas' linear interpolation:
        # Q1=53.5, Q3=66.5, IQR=13.0, mult=1.5
        # Lower = 53.5 - 1.5 * 13.0 = 34.0
        # Upper = 66.5 + 1.5 * 13.0 = 86.0
        assert transformer.bounds_["col1"]["lower"] == pytest.approx(34.0)
        assert transformer.bounds_["col1"]["upper"] == pytest.approx(86.0)

    def test_fit_zscore(self, df_with_outliers):
        """Tests that the z-score bounds are calculated correctly during fit."""
        transformer = OutlierTransformer(
            columns=["col2"], detection_strategy="zscore", zscore_threshold=2.0
        )
        transformer.fit(df_with_outliers)
        assert "col2" in transformer.bounds_
        mean = df_with_outliers["col2"].mean()
        std = df_with_outliers["col2"].std()
        assert transformer.bounds_["col2"]["lower"] == pytest.approx(mean - 2 * std)
        assert transformer.bounds_["col2"]["upper"] == pytest.approx(mean + 2 * std)

    def test_fit_isolation_forest(self, df_with_outliers):
        """Tests that the IsolationForest model is fitted correctly."""
        transformer = OutlierTransformer(
            columns=["col2"],
            detection_strategy="isolation_forest",
            handling_strategy="trim",
            contamination=0.1,
            random_state=42,
        )
        transformer.fit(df_with_outliers)
        assert "col2" in transformer.models_
        assert hasattr(transformer.models_["col2"], "predict")

    # 3. Transform Tests
    def test_transform_winsorize_iqr(self, df_with_outliers):
        """Tests that the winsorize (capping) strategy works correctly with IQR."""
        transformer = OutlierTransformer(
            columns=["col1"], detection_strategy="iqr", handling_strategy="winsorize"
        )
        transformer.fit(df_with_outliers)
        transformed_df = transformer.transform(df_with_outliers)
        bounds = transformer.bounds_["col1"]
        # Check that the outliers were capped to the learned bounds
        assert transformed_df["col1"].min() == pytest.approx(bounds["lower"])
        assert transformed_df["col1"].max() == pytest.approx(bounds["upper"])
        assert (
            df_with_outliers["col1"].iloc[0] < bounds["lower"]
        )  # Verify original was an outlier
        assert (
            df_with_outliers["col1"].iloc[-1] > bounds["upper"]
        )  # Verify original was an outlier

    def test_transform_trim_iqr(self, df_with_outliers):
        """Tests that the trim (removal) strategy works correctly with IQR."""
        transformer = OutlierTransformer(
            columns=["col1"], detection_strategy="iqr", handling_strategy="trim"
        )
        transformer.fit(df_with_outliers)
        transformed_df = transformer.transform(df_with_outliers)
        # Original df has 11 rows, 2 are outliers in col1 (1 and 150)
        assert len(transformed_df) == 9
        assert 1 not in transformed_df["col1"].values
        assert 150 not in transformed_df["col1"].values

    def test_transform_trim_isolation_forest(self, df_with_outliers):
        """Tests that the trim strategy works correctly with IsolationForest."""
        transformer = OutlierTransformer(
            columns=["col2"],
            detection_strategy="isolation_forest",
            handling_strategy="trim",
            contamination=0.1,
            random_state=42,
        )
        transformer.fit(df_with_outliers)
        transformed_df = transformer.transform(df_with_outliers)
        # The model should identify the single large outlier (500)
        assert len(transformed_df) < len(df_with_outliers)
        assert 500 not in transformed_df["col2"].values

    def test_transform_before_fit_raises_error(self, df_with_outliers):
        """Tests that calling transform before fit raises a RuntimeError."""
        transformer = OutlierTransformer(columns=["col1"])
        with pytest.raises(
            RuntimeError, match="Transform called before fitting the transformer."
        ):
            transformer.transform(df_with_outliers)

    # 4. Predict Outliers Tests
    def test_predict_outliers_iqr(self, df_with_outliers):
        """Tests that the predict_outliers method correctly identifies outliers with IQR."""
        transformer = OutlierTransformer(columns=["col1"], detection_strategy="iqr")
        transformer.fit(df_with_outliers)
        outlier_mask = transformer.predict_outliers(df_with_outliers)

        expected_mask = pd.Series(
            [True, False, False, False, False, False, False, False, False, False, True],
            name="col1",
        )
        pd.testing.assert_series_equal(outlier_mask["col1"], expected_mask)

    # 5. Persistence Tests
    def test_save_and_load_preserves_state_iqr(self, df_with_outliers, tmp_path):
        """Tests saving and loading for an IQR-fitted transformer."""
        filepath = tmp_path / "outlier_transformer.joblib"
        original_transformer = OutlierTransformer(
            columns=["col1"], detection_strategy="iqr"
        )
        original_transformer.fit(df_with_outliers)
        original_transformer.save(filepath)

        loaded_transformer = OutlierTransformer.load(filepath)
        assert loaded_transformer.bounds_ == original_transformer.bounds_

        original_transform = original_transformer.transform(df_with_outliers)
        loaded_transform = loaded_transformer.transform(df_with_outliers)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    def test_save_and_load_preserves_state_iso_forest(self, df_with_outliers, tmp_path):
        """Tests saving and loading for an IsolationForest-fitted transformer."""
        filepath = tmp_path / "outlier_transformer_iso.joblib"
        original_transformer = OutlierTransformer(
            columns=["col2"],
            detection_strategy="isolation_forest",
            handling_strategy="trim",
            random_state=42,
        )
        original_transformer.fit(df_with_outliers)
        original_transformer.save(filepath)

        loaded_transformer = OutlierTransformer.load(filepath)
        assert "col2" in loaded_transformer.models_

        original_transform = original_transformer.transform(df_with_outliers)
        loaded_transform = loaded_transformer.transform(df_with_outliers)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    def test_load_raises_file_not_found(self, tmp_path):
        """Tests that loading from a non-existent file raises FileNotFoundError."""
        filepath = tmp_path / "non_existent_transformer.joblib"
        with pytest.raises(FileNotFoundError):
            OutlierTransformer.load(filepath)
