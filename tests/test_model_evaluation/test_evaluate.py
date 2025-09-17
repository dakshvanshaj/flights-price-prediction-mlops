"""
Tests for the model evaluation functions in `src.model_evaluation.evaluation`.

This test suite covers three main functions:
1.  `calculate_all_regression_metrics`: Ensures that all standard regression
    metrics are calculated correctly.
2.  `unscale_predictions`: Verifies that scaled predictions can be accurately
    inverse-transformed back to their original scale.
3.  `time_based_cross_validation`: Contains tests for both tree-based and
    non-tree-based models to ensure the time-series CV loop, model fitting,
    and metric collection work as expected.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from sklearn.linear_model import LinearRegression

from gold_data_preprocessing.scaler import Scaler
from gold_data_preprocessing.power_transformer import PowerTransformer
from model_evaluation.evaluation import (
    unscale_predictions,
    calculate_all_regression_metrics,
    time_based_cross_validation,
)
from shared.config import config_training

# --- Fixtures ---


@pytest.fixture(scope="module")
def y_true_and_pred_series() -> tuple[pd.Series, pd.Series]:
    """Provides simple, known true and predicted series for metric calculation."""
    y_true = pd.Series([100, 150, 200, 250])
    y_pred = pd.Series([110, 145, 210, 240])
    return y_true, y_pred


@pytest.fixture(scope="module")
def fitted_transformers() -> dict:
    """
    Creates and fits a Scaler on sample data.

    Returns a dictionary containing the fitted scaler, the original data,
    and the transformed data. This provides a simple, perfectly reversible
    transformation for testing the `unscale_predictions` function.
    """
    target_col = config_training.TARGET_COLUMN
    original_data = pd.DataFrame(
        {target_col: np.array([10, 20, 30, 40, 50], dtype=float)}
    )

    scaler = Scaler(columns=[target_col], strategy="minmax")
    scaled_data = scaler.fit_transform(original_data)

    # For this test, we only need to test the unscaling of a single transformer.
    # The PowerTransformer's own inverse is tested in its own test file.
    # We create a "dummy" PowerTransformer that does nothing.
    dummy_power_transformer = PowerTransformer(columns=[], strategy="yeo-johnson")
    dummy_power_transformer._is_fitted = True  # Manually set as fitted

    return {
        "scaler": scaler,
        "power_transformer": dummy_power_transformer,  # Use the dummy
        "original_y": original_data[target_col],
        "scaled_y": scaled_data[target_col],  # Use the data from the scaler
    }


@pytest.fixture(scope="module")
def sample_data_for_cv() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a sample DataFrame and Series for cross-validation tests."""
    n_samples = 100
    X = pd.DataFrame(
        {"feature1": range(n_samples), "feature2": np.random.rand(n_samples)}
    )
    y = pd.Series(range(n_samples), dtype=float, name=config_training.TARGET_COLUMN)
    return X, y


# --- Test Classes ---


class TestEvaluationMetrics:
    """Tests for the metric calculation and unscaling functions."""

    def test_calculate_all_regression_metrics(self, y_true_and_pred_series):
        """
        Tests that the metrics calculation function returns a correctly structured
        dictionary with plausible values.
        """
        y_true, y_pred = y_true_and_pred_series
        metrics = calculate_all_regression_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert "r2_score" in metrics
        assert metrics["mean_absolute_error"] == pytest.approx(8.75)

    def test_unscale_predictions(self, fitted_transformers):
        """
        Tests that the inverse transformation pipeline correctly restores scaled
        data back to its original scale.
        """
        scaler = fitted_transformers["scaler"]
        power_transformer = fitted_transformers["power_transformer"]
        original_y = fitted_transformers["original_y"]
        scaled_y = fitted_transformers["scaled_y"]
        scaled_pred = scaled_y.to_numpy()

        y_true_unscaled, y_pred_unscaled = unscale_predictions(
            scaled_y, scaled_pred, scaler, power_transformer
        )

        pd.testing.assert_series_equal(
            original_y, y_true_unscaled, check_exact=False, atol=1e-6
        )
        pd.testing.assert_series_equal(
            original_y, y_pred_unscaled, check_exact=False, atol=1e-6
        )


class TestTimeBasedCrossValidation:
    """Tests for the time-based cross-validation function."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Creates a mock model that returns valid predictions."""
        model = MagicMock(spec=LinearRegression)
        # Configure the mock to return a valid prediction array of a fixed size.
        # This size must match the `test_size` used in the CV tests.
        model.predict.return_value = np.array([1] * 10)
        return model

    def test_cv_for_tree_model(self, sample_data_for_cv, mock_model):
        """
        Tests the CV logic for a tree-based model where no unscaling is needed.
        """
        X, y = sample_data_for_cv
        n_splits = 5
        test_size = 10

        result = time_based_cross_validation(
            model=mock_model,
            X=X,
            y=y,
            n_splits=n_splits,
            test_size=test_size,
            is_tree_model=True,
        )

        assert mock_model.fit.call_count == n_splits
        assert mock_model.predict.call_count == n_splits
        assert "score" in result
        assert "unscaled" not in result
        assert len(result["score"]) == n_splits

    def test_cv_for_linear_model(
        self, sample_data_for_cv, mock_model, fitted_transformers
    ):
        """
        Tests the CV logic for a non-tree model where predictions are unscaled.
        """
        X, y = sample_data_for_cv
        scaler = fitted_transformers["scaler"]
        power_transformer = fitted_transformers["power_transformer"]
        n_splits = 5
        test_size = 10

        result = time_based_cross_validation(
            model=mock_model,
            X=X,
            y=y,
            n_splits=n_splits,
            test_size=test_size,
            scaler=scaler,
            power_transformer=power_transformer,
            is_tree_model=False,
        )

        assert mock_model.fit.call_count == n_splits
        assert "scaled" in result
        assert "unscaled" in result
        assert len(result["scaled"]) == n_splits
        assert len(result["unscaled"]) == n_splits
