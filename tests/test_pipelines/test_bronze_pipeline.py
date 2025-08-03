import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# The module we are testing
from src.pipelines import bronze_pipeline

# By using 'patch' as a context manager or decorator, we can avoid using monkeypatch
# for every single function, making the fixtures cleaner.


@pytest.fixture
def mock_ge_and_file_utils():
    """
    Mocks all external dependencies for the bronze pipeline in a single fixture.
    This includes Great Expectations (GE) functions and file handling utils.
    """
    # We patch the entire modules to intercept any function called from them.
    with (
        patch("src.pipelines.bronze_pipeline.get_ge_context"),
        patch("src.pipelines.bronze_pipeline.get_or_create_datasource"),
        patch("src.pipelines.bronze_pipeline.get_or_create_csv_asset"),
        patch("src.pipelines.bronze_pipeline.get_or_create_batch_definition"),
        patch("src.pipelines.bronze_pipeline.get_or_create_expectation_suite"),
        patch("src.pipelines.bronze_pipeline.build_bronze_expectations"),
        patch("src.pipelines.bronze_pipeline.add_expectations_to_suite"),
        patch("src.pipelines.bronze_pipeline.get_or_create_validation_definition"),
        patch("src.pipelines.bronze_pipeline.get_action_list"),
        patch("src.pipelines.bronze_pipeline.get_or_create_checkpoint"),
        patch("src.pipelines.bronze_pipeline.run_checkpoint") as mock_run_checkpoint,
        patch(
            "src.pipelines.bronze_pipeline.handle_file_based_on_validation"
        ) as mock_handle_file,
    ):
        # Create a mock result object that run_checkpoint will return
        mock_validation_result = MagicMock()
        mock_run_checkpoint.return_value = mock_validation_result

        # Yield the critical mocks that we need to control in our tests
        yield {
            "validation_result": mock_validation_result,
            "handle_file": mock_handle_file,
        }


@pytest.fixture
def setup_bronze_test_env(tmp_path: Path, monkeypatch):
    """
    Creates a temporary directory structure for bronze pipeline tests
    and monkeypatches the config variables to use these temporary paths.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    quarantine_dir = tmp_path / "quarantine"

    raw_dir.mkdir()
    processed_dir.mkdir()
    quarantine_dir.mkdir()

    # Create a dummy test file inside the raw directory
    test_file = raw_dir / "test_data.csv"
    test_file.touch()

    # Monkeypatch the config objects that the pipeline script imports
    monkeypatch.setattr(bronze_pipeline.config_bronze, "RAW_DATA_SOURCE", raw_dir)
    monkeypatch.setattr(
        bronze_pipeline.config_bronze, "BRONZE_PROCESSED_DIR", processed_dir
    )
    monkeypatch.setattr(
        bronze_pipeline.config_bronze, "BRONZE_QUARANTINE_DIR", quarantine_dir
    )

    return test_file, processed_dir, quarantine_dir


def test_run_bronze_pipeline_success(setup_bronze_test_env, mock_ge_and_file_utils):
    """
    Tests the success scenario: GE validation passes and file move succeeds.
    """
    # ARRANGE
    test_file, processed_dir, _ = setup_bronze_test_env
    file_name = test_file.name

    # Configure the mocks to simulate a successful run
    mock_ge_and_file_utils["validation_result"].success = True
    mock_ge_and_file_utils["handle_file"].return_value = True

    # ACT
    pipeline_result = bronze_pipeline.run_bronze_pipeline(file_name=file_name)

    # ASSERT
    assert pipeline_result is True
    # Verify that the file handler was called with the correct success status and directories
    mock_ge_and_file_utils["handle_file"].assert_called_once()
    call_args = mock_ge_and_file_utils["handle_file"].call_args[1]
    assert call_args["result"].success is True
    assert call_args["success_dir"] == processed_dir


def test_run_bronze_pipeline_failure_on_validation(
    setup_bronze_test_env, mock_ge_and_file_utils
):
    """
    Tests the failure scenario: GE validation fails.
    """
    # ARRANGE
    test_file, _, quarantine_dir = setup_bronze_test_env
    file_name = test_file.name

    # Configure the mocks to simulate a validation failure
    mock_ge_and_file_utils["validation_result"].success = False
    mock_ge_and_file_utils[
        "handle_file"
    ].return_value = True  # Assume move would succeed

    # ACT
    pipeline_result = bronze_pipeline.run_bronze_pipeline(file_name=file_name)

    # ASSERT
    assert pipeline_result is False
    # Verify that the file handler was called with the correct failure status
    mock_ge_and_file_utils["handle_file"].assert_called_once()
    call_args = mock_ge_and_file_utils["handle_file"].call_args[1]
    assert call_args["result"].success is False
    assert call_args["failure_dir"] == quarantine_dir


def test_run_bronze_pipeline_failure_on_file_move(
    setup_bronze_test_env, mock_ge_and_file_utils
):
    """
    Tests the failure scenario: GE validation passes but the file move fails.
    """
    # ARRANGE
    test_file, _, _ = setup_bronze_test_env
    file_name = test_file.name

    # Configure the mocks to simulate a successful validation but a failed move
    mock_ge_and_file_utils["validation_result"].success = True
    mock_ge_and_file_utils["handle_file"].return_value = False

    # ACT
    pipeline_result = bronze_pipeline.run_bronze_pipeline(file_name=file_name)

    # ASSERT
    assert pipeline_result is False
    mock_ge_and_file_utils["handle_file"].assert_called_once()


def test_run_bronze_pipeline_file_not_found(setup_bronze_test_env, caplog):
    """
    Tests that the pipeline returns False immediately if the input file doesn't exist.
    """
    # ARRANGE
    # We don't need the mocks for GE or file handling, as the function should exit early.

    # ACT
    pipeline_result = bronze_pipeline.run_bronze_pipeline(
        file_name="nonexistent_file.csv"
    )

    # ASSERT
    assert pipeline_result is False
    # Check that a specific error message was logged
    assert "File not found" in caplog.text
