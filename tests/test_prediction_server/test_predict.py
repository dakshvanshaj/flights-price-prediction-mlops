"""
Tests for the prediction server's preprocessing and postprocessing logic.

These tests verify the orchestration logic in `src.prediction_server.predict`.
They ensure that all the necessary transformation functions and fitted preprocessor
objects are called in the correct order during both the preprocessing and
postprocessing stages.

Mocking is used to isolate the functions from their dependencies, allowing us to
check the sequence of calls without performing any actual data transformations.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Functions to be tested
from prediction_server.predict import (
    preprocessing_for_prediction,
    postprocessing_for_target,
)

# --- Fixtures ---


@pytest.fixture
def sample_raw_input() -> pd.DataFrame:
    """Provides a single-row DataFrame mimicking a raw API input."""
    return pd.DataFrame(
        [
            {
                "date": "2024-08-15",
                "airline": "Indigo",
                "from_location": "BLR",
                "to_location": "DEL",
                # Add other raw features as needed
            }
        ]
    )


@pytest.fixture
def mock_preprocessors() -> dict:
    """Creates a dictionary of mock preprocessing objects."""
    # Each object is a MagicMock, so we can track calls to its methods (e.g., .transform)
    return {
        "imputer": MagicMock(),
        "encoder": MagicMock(),
        "grouper": MagicMock(),
        "outlier_handler": MagicMock(),
        "power_transformer": MagicMock(),
        "scaler": MagicMock(),
        "final_columns": ["price", "feature_a", "feature_b"],  # Dummy final columns
    }


# --- Test for preprocessing_for_prediction ---


@patch("prediction_server.predict.rename_specific_columns")
@patch("prediction_server.predict.standardize_column_format")
@patch("prediction_server.predict.create_date_features")
@patch("prediction_server.predict.create_cyclical_features")
@patch("prediction_server.predict.create_categorical_interaction_features")
def test_preprocessing_orchestration(
    mock_interactions,
    mock_cyclical,
    mock_dates,
    mock_standardize,
    mock_rename,
    sample_raw_input,
    mock_preprocessors,
):
    """
    Tests that `preprocessing_for_prediction` calls all transformation steps
    in the correct sequence.
    """
    # ARRANGE
    # Configure mocks to return a DataFrame to allow chaining
    mock_rename.return_value = sample_raw_input
    mock_standardize.return_value = sample_raw_input
    mock_dates.return_value = sample_raw_input
    mock_cyclical.return_value = sample_raw_input
    mock_interactions.return_value = sample_raw_input

    # Configure mock transformers to also return a chainable DataFrame
    for name, mock_obj in mock_preprocessors.items():
        if hasattr(mock_obj, "transform"):
            # Make the mock return a dummy df to allow the pipeline to continue
            mock_obj.transform.return_value = pd.DataFrame({"feature_a": [1]})

    # ACT
    preprocessing_for_prediction(sample_raw_input, mock_preprocessors)

    # ASSERT
    # Verify that each standalone function was called exactly once
    mock_rename.assert_called_once()
    mock_standardize.assert_called_once()
    mock_dates.assert_called_once()
    mock_cyclical.assert_called_once()
    mock_interactions.assert_called_once()

    # Verify that each preprocessor's transform method was called exactly once
    mock_preprocessors["imputer"].transform.assert_called_once()
    mock_preprocessors["encoder"].transform.assert_called_once()
    mock_preprocessors["grouper"].transform.assert_called_once()
    mock_preprocessors["outlier_handler"].transform.assert_called_once()
    mock_preprocessors["power_transformer"].transform.assert_called_once()
    mock_preprocessors["scaler"].transform.assert_called_once()


# --- Test for postprocessing_for_target ---


def test_postprocessing_orchestration():
    """
    Tests that `postprocessing_for_target` calls inverse transforms in the
    correct reverse order (scaler -> power_transformer).
    """
    # ARRANGE
    prediction_df = pd.DataFrame({"price": [0.5]})

    # Use a Mock as a manager to check the call order of its children
    mock_manager = MagicMock()

    mock_preprocessors = {
        "scaler": mock_manager.scaler,
        "power_transformer": mock_manager.power_transformer,
    }

    # Configure mocks to return a DataFrame to allow chaining
    mock_manager.scaler.inverse_transform.return_value = prediction_df
    mock_manager.power_transformer.inverse_transform.return_value = prediction_df

    # ACT
    postprocessing_for_target(prediction_df, mock_preprocessors)

    # ASSERT
    # The `method_calls` attribute records calls to the mock's children in order.
    # This is a robust way to check the sequence of operations.
    call_order = [call[0] for call in mock_manager.method_calls]

    assert call_order == [
        "scaler.inverse_transform",
        "power_transformer.inverse_transform",
    ]
