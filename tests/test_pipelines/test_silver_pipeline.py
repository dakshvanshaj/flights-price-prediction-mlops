# tests/pipelines/test_silver_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest


# The module we are testing
from src.pipelines import silver_pipeline
from shared import config


@pytest.fixture
def mock_ge_checkpoint(monkeypatch):
    """Mocks the Great Expectations checkpoint function."""
    mock_result = MagicMock()
    mock_run_checkpoint = MagicMock(return_value=mock_result)
    monkeypatch.setattr(
        silver_pipeline, "run_checkpoint_on_dataframe", mock_run_checkpoint
    )
    return mock_result


@pytest.fixture
def setup_silver_test_env(tmp_path: Path, monkeypatch):
    """
    Creates a temporary directory structure and a realistic mock input file
    for silver pipeline tests.
    """
    bronze_processed_dir = tmp_path / "bronze_processed"
    silver_processed_dir = tmp_path / "silver_processed"
    silver_quarantine_dir = tmp_path / "silver_quarantine"

    bronze_processed_dir.mkdir()
    silver_processed_dir.mkdir()
    silver_quarantine_dir.mkdir()

    # Patch the config variables in the 'config' module to use our temp dirs
    monkeypatch.setattr(config, "BRONZE_PROCESSED_DIR", bronze_processed_dir)
    monkeypatch.setattr(config, "SILVER_PROCESSED_DIR", silver_processed_dir)
    monkeypatch.setattr(config, "SILVER_QUARANTINE_DIR", silver_quarantine_dir)

    # Create a realistic mock DataFrame
    mock_data = {
        "travel_code": range(1, 12),
        "user_code": range(101, 112),
        "from": [f"City_{i % 2}" for i in range(11)],
        "to": [f"City_{(i + 1) % 2}" for i in range(11)],
        "flight_type": ["Economy"] * 6 + ["Business"] * 5,
        "price": [i * 1000 for i in range(1, 12)],
        "time": [2.5 + i * 0.1 for i in range(11)],
        "distance": [1100 + i * 10 for i in range(11)],
        "agency": ["AirlineA"] * 11,
        "date": pd.to_datetime(pd.date_range(start="2024-01-01", periods=11)),
    }
    mock_df = pd.DataFrame(mock_data)
    test_file_path = bronze_processed_dir / "test_data.csv"
    mock_df.to_csv(test_file_path, index=False)

    return test_file_path, silver_processed_dir, silver_quarantine_dir


def test_run_silver_pipeline_success(
    setup_silver_test_env, mock_ge_checkpoint, monkeypatch
):
    """
    Tests that the pipeline returns True and saves the file to the processed
    directory when validation and the save operation succeed.
    """
    input_path, processed_dir, quarantine_dir = setup_silver_test_env

    # --- MOCK SUCCESS ---
    mock_ge_checkpoint.success = True
    mock_save_helper = MagicMock(return_value=True)
    monkeypatch.setattr(
        silver_pipeline, "save_dataframe_based_on_validation", mock_save_helper
    )

    # --- RUN PIPELINE ---
    pipeline_success = silver_pipeline.run_silver_pipeline(
        input_filepath=str(input_path)
    )

    # --- ASSERT ---
    assert pipeline_success is True
    mock_save_helper.assert_called_once()
    call_args = mock_save_helper.call_args[1]
    assert call_args["result"].success is True
    assert call_args["success_dir"] == processed_dir
    assert call_args["failure_dir"] == quarantine_dir


def test_run_silver_pipeline_failure_on_validation(
    setup_silver_test_env, mock_ge_checkpoint, monkeypatch
):
    """
    Tests that the pipeline returns False and quarantines the file when
    Great Expectations validation fails.
    """
    input_path, processed_dir, quarantine_dir = setup_silver_test_env

    # --- MOCK FAILURE ---
    mock_ge_checkpoint.success = False
    mock_save_helper = MagicMock(return_value=True)
    monkeypatch.setattr(
        silver_pipeline, "save_dataframe_based_on_validation", mock_save_helper
    )

    # --- RUN PIPELINE ---
    pipeline_success = silver_pipeline.run_silver_pipeline(
        input_filepath=str(input_path)
    )

    # --- ASSERT ---
    assert pipeline_success is False
    mock_save_helper.assert_called_once()
    call_args = mock_save_helper.call_args[1]
    assert call_args["result"].success is False
    assert call_args["success_dir"] == processed_dir
    assert call_args["failure_dir"] == quarantine_dir


def test_run_silver_pipeline_failure_on_save(
    setup_silver_test_env, mock_ge_checkpoint, monkeypatch
):
    """
    Tests that the pipeline returns False when validation succeeds but the
    final save operation fails.
    """
    input_path, _, _ = setup_silver_test_env

    # --- MOCK SUCCESSFUL VALIDATION BUT FAILED SAVE ---
    mock_ge_checkpoint.success = True
    mock_save_helper = MagicMock(return_value=False)
    monkeypatch.setattr(
        silver_pipeline, "save_dataframe_based_on_validation", mock_save_helper
    )

    # --- RUN PIPELINE ---
    pipeline_success = silver_pipeline.run_silver_pipeline(
        input_filepath=str(input_path)
    )

    # --- ASSERT ---
    assert pipeline_success is False
    mock_save_helper.assert_called_once()
