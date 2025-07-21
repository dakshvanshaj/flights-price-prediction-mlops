# tests/pipelines/test_bronze_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# The module we are testing
from src.pipelines import bronze_pipeline


@pytest.fixture
def mock_ge_functions(monkeypatch):
    """Mocks all Great Expectations component functions in the bronze pipeline module."""
    monkeypatch.setattr(bronze_pipeline, "get_ge_context", MagicMock())
    monkeypatch.setattr(bronze_pipeline, "get_or_create_datasource", MagicMock())
    monkeypatch.setattr(bronze_pipeline, "get_or_create_csv_asset", MagicMock())
    monkeypatch.setattr(bronze_pipeline, "get_or_create_batch_definition", MagicMock())
    monkeypatch.setattr(bronze_pipeline, "get_or_create_expectation_suite", MagicMock())
    monkeypatch.setattr(bronze_pipeline, "add_expectations_to_suite", MagicMock())
    monkeypatch.setattr(
        bronze_pipeline, "get_or_create_validation_definition", MagicMock()
    )
    monkeypatch.setattr(bronze_pipeline, "get_or_create_checkpoint", MagicMock())

    # Mock the final checkpoint run and return a mock result object
    mock_result = MagicMock()
    mock_run_checkpoint = MagicMock(return_value=mock_result)
    monkeypatch.setattr(bronze_pipeline, "run_checkpoint", mock_run_checkpoint)

    # Return the mock result so its 'success' attribute can be set in tests
    return mock_result


@pytest.fixture
def setup_bronze_test_env(tmp_path: Path, monkeypatch):
    """
    Creates a temporary directory structure for bronze pipeline tests
    and monkeypatches the config variables to use these temporary paths.
    """
    raw_data_source_dir = tmp_path / "train_validation_test"
    processed_dir = tmp_path / "processed"
    quarantine_dir = tmp_path / "quarantine"
    raw_data_source_dir.mkdir()
    processed_dir.mkdir()
    quarantine_dir.mkdir()

    # This is the key: we tell the bronze_pipeline module to use our temp dirs
    monkeypatch.setattr(bronze_pipeline, "RAW_DATA_SOURCE", raw_data_source_dir)
    monkeypatch.setattr(bronze_pipeline, "BRONZE_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bronze_pipeline, "BRONZE_QUARANTINE_DIR", quarantine_dir)

    test_file = raw_data_source_dir / "test_data.csv"
    test_file.touch()

    return test_file, processed_dir, quarantine_dir


def test_run_bronze_pipeline_success(
    monkeypatch, setup_bronze_test_env, mock_ge_functions
):
    """
    Tests that the pipeline moves the file to the 'processed' directory
    when validation succeeds.
    """
    test_file, processed_dir, _ = setup_bronze_test_env
    file_name = test_file.name

    # Configure the mock to simulate SUCCESS
    mock_ge_functions.success = True

    mock_shutil_move = MagicMock()
    monkeypatch.setattr(bronze_pipeline.shutil, "move", mock_shutil_move)

    success = bronze_pipeline.run_bronze_pipeline(file_name=file_name)

    assert success is True
    mock_shutil_move.assert_called_once_with(
        src=test_file, dst=processed_dir / file_name
    )


def test_run_bronze_pipeline_failure(
    monkeypatch, setup_bronze_test_env, mock_ge_functions
):
    """
    Tests that the pipeline moves the file to the 'quarantine' directory
    when validation fails.
    """
    test_file, _, quarantine_dir = setup_bronze_test_env
    file_name = test_file.name

    # Configure the mock to simulate FAILURE
    mock_ge_functions.success = False

    mock_shutil_move = MagicMock()
    monkeypatch.setattr(bronze_pipeline.shutil, "move", mock_shutil_move)

    success = bronze_pipeline.run_bronze_pipeline(file_name=file_name)

    assert success is False
    mock_shutil_move.assert_called_once_with(
        src=test_file, dst=quarantine_dir / file_name
    )
