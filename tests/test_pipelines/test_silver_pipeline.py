import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# The module we are testing
from src.pipelines import silver_pipeline


@pytest.fixture
def mock_silver_dependencies():
    """
    Mocks all external dependencies for the silver pipeline, including data loading,
    all preprocessing functions, GE validation, and the final save utility.
    """
    # Use a single `with patch` block to manage all mocks cleanly
    with (
        patch("src.pipelines.silver_pipeline.load_data") as mock_load,
        patch("src.pipelines.silver_pipeline.rename_specific_columns") as mock_rename,
        patch(
            "src.pipelines.silver_pipeline.standardize_column_format"
        ) as mock_standardize,
        patch("src.pipelines.silver_pipeline.optimize_data_types") as mock_optimize,
        patch(
            "src.pipelines.silver_pipeline.create_date_features"
        ) as mock_create_dates,
        patch(
            "src.pipelines.silver_pipeline.handle_erroneous_duplicates"
        ) as mock_handle_dupes,
        patch("src.pipelines.silver_pipeline.sort_data_by_date") as mock_sort,
        patch(
            "src.pipelines.silver_pipeline.enforce_column_order"
        ) as mock_enforce_order,
        patch(
            "src.pipelines.silver_pipeline.run_checkpoint_on_dataframe"
        ) as mock_run_checkpoint,
        patch(
            "src.pipelines.silver_pipeline.save_dataframe_based_on_validation"
        ) as mock_save_df,
    ):
        # --- Configure the default behavior of the mocks ---

        # Mock preprocessing functions to just return the DataFrame they receive
        mock_rename.side_effect = lambda df, **kwargs: df
        mock_standardize.side_effect = lambda df, **kwargs: df
        mock_optimize.side_effect = lambda df, **kwargs: df
        mock_create_dates.side_effect = lambda df, **kwargs: df
        mock_handle_dupes.side_effect = lambda df, **kwargs: df
        mock_sort.side_effect = lambda df, **kwargs: df
        mock_enforce_order.side_effect = lambda df, **kwargs: df

        # Mock the GE validation result object
        mock_validation_result = MagicMock()
        mock_run_checkpoint.return_value = mock_validation_result

        # Yield the mocks that we need to control in our tests
        yield {
            "load_data": mock_load,
            "validation_result": mock_validation_result,
            "save_df": mock_save_df,
        }


@pytest.fixture
def setup_silver_test_env(tmp_path: Path, monkeypatch):
    """
    Creates a temporary directory structure for silver pipeline tests
    and monkeypatches the config variables to use these temporary paths.
    """
    bronze_dir = tmp_path / "bronze_processed"
    silver_processed_dir = tmp_path / "silver_processed"
    silver_quarantine_dir = tmp_path / "silver_quarantine"

    bronze_dir.mkdir()
    silver_processed_dir.mkdir()
    silver_quarantine_dir.mkdir()

    # Create a dummy input file
    test_file = bronze_dir / "test_data.csv"
    test_file.touch()

    # Monkeypatch the config objects that the pipeline script imports
    monkeypatch.setattr(
        silver_pipeline.config_bronze, "BRONZE_PROCESSED_DIR", bronze_dir
    )
    monkeypatch.setattr(
        silver_pipeline.config_silver, "SILVER_PROCESSED_DIR", silver_processed_dir
    )
    monkeypatch.setattr(
        silver_pipeline.config_silver, "SILVER_QUARANTINE_DIR", silver_quarantine_dir
    )

    return test_file, silver_processed_dir, silver_quarantine_dir


def test_run_silver_pipeline_success(setup_silver_test_env, mock_silver_dependencies):
    """
    Tests the success scenario: all steps complete successfully.
    """
    # ARRANGE
    test_file, processed_dir, _ = setup_silver_test_env

    # Configure mocks for a successful run
    mock_silver_dependencies["load_data"].return_value = pd.DataFrame({"a": [1]})
    mock_silver_dependencies["validation_result"].success = True
    mock_silver_dependencies["save_df"].return_value = True

    # ACT
    pipeline_result = silver_pipeline.run_silver_pipeline(input_filepath=test_file)

    # ASSERT
    assert pipeline_result is True
    mock_silver_dependencies["save_df"].assert_called_once()
    call_args = mock_silver_dependencies["save_df"].call_args[1]
    assert call_args["result"].success is True
    assert call_args["success_dir"] == processed_dir


def test_run_silver_pipeline_failure_on_validation(
    setup_silver_test_env, mock_silver_dependencies
):
    """
    Tests the failure scenario: GE validation fails.
    """
    # ARRANGE
    test_file, _, quarantine_dir = setup_silver_test_env

    # Configure mocks for a validation failure
    mock_silver_dependencies["load_data"].return_value = pd.DataFrame({"a": [1]})
    mock_silver_dependencies["validation_result"].success = False
    mock_silver_dependencies["save_df"].return_value = True  # Assume save would work

    # ACT
    pipeline_result = silver_pipeline.run_silver_pipeline(input_filepath=test_file)

    # ASSERT
    assert pipeline_result is False
    mock_silver_dependencies["save_df"].assert_called_once()
    call_args = mock_silver_dependencies["save_df"].call_args[1]
    assert call_args["result"].success is False
    assert call_args["failure_dir"] == quarantine_dir


def test_run_silver_pipeline_failure_on_save(
    setup_silver_test_env, mock_silver_dependencies
):
    """
    Tests the failure scenario: validation passes but the file save fails.
    """
    # ARRANGE
    test_file, _, _ = setup_silver_test_env

    # Configure mocks for a save failure
    mock_silver_dependencies["load_data"].return_value = pd.DataFrame({"a": [1]})
    mock_silver_dependencies["validation_result"].success = True
    mock_silver_dependencies["save_df"].return_value = False

    # ACT
    pipeline_result = silver_pipeline.run_silver_pipeline(input_filepath=test_file)

    # ASSERT
    assert pipeline_result is False
    mock_silver_dependencies["save_df"].assert_called_once()


def test_run_silver_pipeline_file_not_found(caplog):
    """
    Tests that the pipeline returns False immediately if the input file doesn't exist.
    """
    # ARRANGE
    non_existent_file = Path("/tmp/non_existent_file.csv")

    # ACT
    pipeline_result = silver_pipeline.run_silver_pipeline(
        input_filepath=non_existent_file
    )

    # ASSERT
    assert pipeline_result is False
    assert "File not found" in caplog.text


def test_run_silver_pipeline_load_data_fails(
    setup_silver_test_env, mock_silver_dependencies, caplog
):
    """
    Tests that the pipeline returns False if the data loading step fails.
    """
    # ARRANGE
    test_file, _, _ = setup_silver_test_env
    mock_silver_dependencies[
        "load_data"
    ].return_value = None  # Simulate loading failure

    # ACT
    pipeline_result = silver_pipeline.run_silver_pipeline(input_filepath=test_file)

    # ASSERT
    assert pipeline_result is False
    assert "Failed to load data" in caplog.text
