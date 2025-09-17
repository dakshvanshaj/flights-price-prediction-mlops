"""
Tests for the model training script `src.model_training.train`.

These tests verify that:
1.  The model factory `get_model` correctly instantiates models from string names
    and raises an error for unsupported models.
2.  The `train_model` function calls the underlying model's `fit` method with the
    correct arguments, including the special handling for LightGBM's
    categorical features.

Mocking is used to isolate the functions from actual model training, making the
tests fast and focused on the orchestration logic.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

# Functions to be tested
from model_training.train import get_model, train_model

# --- Fixtures ---


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a simple DataFrame and Series for testing training calls."""
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([1, 2, 3])
    return X, y


# --- Tests for get_model --- #


def test_get_model_success():
    """
    Tests that `get_model` successfully instantiates a known model
    with the correct parameters.
    """
    # ARRANGE
    model_name = "Ridge"
    params = {"alpha": 0.5}

    # ACT
    model = get_model(model_name, params)

    # ASSERT
    assert isinstance(model, Ridge)
    assert model.alpha == 0.5


def test_get_model_failure():
    """
    Tests that `get_model` raises a ValueError for an unsupported model name.
    """
    # ARRANGE
    model_name = "UnsupportedModel"
    params = {}

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Unsupported model: 'UnsupportedModel'"):
        get_model(model_name, params)


# --- Tests for train_model --- #


def test_train_model_standard(sample_training_data):
    """
    Tests that `train_model` calls the `fit` method on a standard model.
    """
    # ARRANGE
    X_train, y_train = sample_training_data
    # Create a mock model that we can spy on
    mock_model = MagicMock(spec=Ridge)

    # ACT
    train_model(X_train, y_train, mock_model)

    # ASSERT
    # Verify that the fit method was called exactly once with the correct data
    mock_model.fit.assert_called_once_with(X_train, y_train)


def test_train_model_with_lgbm_categoricals(sample_training_data):
    """
    Tests the special case where `train_model` is given an LGBMRegressor
    and a list of categorical features.
    """
    # ARRANGE
    X_train, y_train = sample_training_data
    mock_lgbm = MagicMock(spec=LGBMRegressor)
    categorical_cols = ["feature2"]

    # ACT
    train_model(X_train, y_train, mock_lgbm, categorical_features=categorical_cols)

    # ASSERT
    # Verify that `fit` was called with the special `categorical_feature` argument
    mock_lgbm.fit.assert_called_once_with(
        X_train, y_train, categorical_feature=categorical_cols
    )
