# tests/pipelines/test_silver_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# The module we are testing
from src.pipelines import silver_pipeline


@pytest.fixture
def mock_silver_ge_checkpoint(monkeypatch):
    """Mocks the Great Expectations checkpoint function in the silver pipeline module."""
    mock_result = MagicMock()
    mock_run_checkpoint = MagicMock(return_value=mock_result)
    monkeypatch.setattr(
        silver_pipeline, "run_checkpoint_on_dataframe", mock_run_checkpoint
    )
    # Return both the result and the mock function to allow for assertions
    return mock_result, mock_run_checkpoint


@pytest.fixture
def setup_silver_test_env(tmp_path: Path, monkeypatch):
    """
    Creates a temporary directory structure for silver pipeline tests
    and monkeypatches the config variables to use these temporary paths.
    """
    bronze_processed_dir = tmp_path / "bronze_processed"
    silver_quarantine_dir = tmp_path / "silver_quarantine"
    imputer_dir = tmp_path / "imputer"

    bronze_processed_dir.mkdir()
    silver_quarantine_dir.mkdir()
    imputer_dir.mkdir()

    # Patch the quarantine directory used inside the pipeline function
    monkeypatch.setattr(silver_pipeline, "SILVER_QUARANTINE_DIR", silver_quarantine_dir)

    # Create a dummy input file and imputer path
    test_file = bronze_processed_dir / "test_data.csv"
    test_file.touch()  # Content is irrelevant as pd.read_csv is mocked
    imputer_path = imputer_dir / "imputer.joblib"

    return test_file, silver_quarantine_dir, imputer_path


@pytest.fixture
def mock_silver_input_df():
    """Provides a realistic mock DataFrame for silver pipeline tests."""
    mock_df_data = {
        "user_code": ["U1"],
        "from_location": ["A"],
        "to_location": ["B"],
        "date": pd.to_datetime(["2023-01-01"]),
        "time": ["10:00"],
        "agency": ["Agency1"],
    }
    return pd.DataFrame(mock_df_data)


def test_run_silver_pipeline_success(
    monkeypatch, setup_silver_test_env, mock_silver_ge_checkpoint, mock_silver_input_df
):
    """
    Tests that the pipeline returns a DataFrame when validation succeeds.
    """
    test_file, _, imputer_path = setup_silver_test_env

    # Configure the mock GE result to simulate SUCCESS
    mock_ge_result, _ = mock_silver_ge_checkpoint
    mock_ge_result.success = True

    mock_df = mock_silver_input_df
    monkeypatch.setattr(silver_pipeline, "load_data", MagicMock(return_value=mock_df))

    mock_handler_instance = MagicMock()
    mock_handler_instance.transform.return_value = mock_df  # imputer returns a df
    mock_handler_class = MagicMock()
    mock_handler_class.load.return_value = mock_handler_instance
    monkeypatch.setattr(silver_pipeline, "MissingValueHandler", mock_handler_class)

    mock_to_csv = MagicMock()
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    # Run the pipeline in inference mode
    result_df = silver_pipeline.run_silver_pipeline(
        input_filepath=str(test_file),
        imputer_path=str(imputer_path),
        train_mode=False,
    )

    # Assertions
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    mock_to_csv.assert_not_called()  # Should not be quarantined


def test_run_silver_pipeline_failure(
    monkeypatch, setup_silver_test_env, mock_silver_ge_checkpoint, mock_silver_input_df
):
    """
    Tests that the pipeline returns None and quarantines the file when validation fails.
    """
    test_file, quarantine_dir, imputer_path = setup_silver_test_env
    file_name = test_file.name

    # Configure the mock GE result to simulate FAILURE
    mock_ge_result, _ = mock_silver_ge_checkpoint
    mock_ge_result.success = False

    mock_df = mock_silver_input_df
    monkeypatch.setattr(silver_pipeline, "load_data", MagicMock(return_value=mock_df))

    mock_handler_instance = MagicMock()
    mock_handler_instance.transform.return_value = mock_df
    mock_handler_class = MagicMock()
    mock_handler_class.load.return_value = mock_handler_instance
    monkeypatch.setattr(silver_pipeline, "MissingValueHandler", mock_handler_class)

    mock_to_csv = MagicMock()
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    # Run the pipeline in inference mode
    result_df = silver_pipeline.run_silver_pipeline(
        input_filepath=str(test_file),
        imputer_path=str(imputer_path),
        train_mode=False,
    )

    # Assertions
    assert result_df is None
    mock_to_csv.assert_called_once_with(quarantine_dir / file_name, index=False)


def test_run_silver_pipeline_train_mode(
    monkeypatch, setup_silver_test_env, mock_silver_ge_checkpoint, mock_silver_input_df
):
    """
    Tests that the pipeline correctly fits and saves an imputer when in train_mode.
    """
    test_file, _, imputer_path = setup_silver_test_env

    # Configure mocks
    mock_ge_result, _ = mock_silver_ge_checkpoint
    mock_ge_result.success = True

    mock_df = mock_silver_input_df
    monkeypatch.setattr(silver_pipeline, "load_data", MagicMock(return_value=mock_df))

    # Mock all intermediate preprocessing functions to isolate the test
    # to the pipeline's orchestration logic. We make them return the
    # dataframe they receive, so the `mock_df` is passed through unchanged.
    def identity_func(df, **kwargs):
        return df

    preprocessing_functions_to_mock = [
        "rename_specific_columns",
        "standardize_column_format",
        "optimize_data_types",
        "sort_data_by_date",
        "handle_erroneous_duplicates",
        "create_date_features",
        "enforce_column_order",
    ]
    for func_name in preprocessing_functions_to_mock:
        monkeypatch.setattr(silver_pipeline, func_name, identity_func)

    # Mock the MissingValueHandler class to track its instance and calls
    mock_handler_instance = MagicMock()
    mock_handler_instance.transform.return_value = (
        mock_df  # Ensure transform returns a DF
    )
    mock_handler_class = MagicMock(return_value=mock_handler_instance)
    monkeypatch.setattr(silver_pipeline, "MissingValueHandler", mock_handler_class)

    # Run the pipeline in training mode
    result_df = silver_pipeline.run_silver_pipeline(
        input_filepath=str(test_file),
        imputer_path=str(imputer_path),
        train_mode=True,  # Key change for this test
        column_strategies={},  # Pass dummy args
        exclude_cols_imputation=[],
    )

    # Assertions
    assert isinstance(result_df, pd.DataFrame)
    mock_handler_class.assert_called_once()  # Was the imputer created?
    mock_handler_instance.fit.assert_called_once_with(
        mock_df
    )  # Was it fitted with the correct df?
    # Assert that save was called with the STRING representation of the path
    mock_handler_instance.save.assert_called_once_with(str(imputer_path))
    mock_handler_instance.transform.assert_called_once_with(
        mock_df
    )  # Was data transformed?
