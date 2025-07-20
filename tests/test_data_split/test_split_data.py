# tests/data_preparation/test_split_data.py

import pytest
from pathlib import Path
import pandas as pd

# NOTE: The import of `split_data` is moved inside the test functions.
# This is crucial for the monkeypatch to work correctly, as it ensures
# the configuration is patched *before* the function is imported.


@pytest.fixture
def setup_test_environment(
    tmp_path: Path, monkeypatch, raw_data_for_splitting: pd.DataFrame
):
    temp_input_dir = tmp_path / "raw"
    temp_input_dir.mkdir()

    temp_splits_dir = temp_input_dir / "train_valdation_test"
    temp_splits_dir.mkdir()

    input_csv = temp_input_dir / "flights.csv"

    raw_data_for_splitting.to_csv(input_csv, index=False)

    import shared.config as config
    # 3. Use monkeypatch to override config values for the duration of a test.
    # We import the config module that the script-under-test will use.
    # This requires `pythonpath = ["src"]` in your pyproject.toml

    monkeypatch.setattr(config, "INPUT_CSV_PATH", input_csv)
    monkeypatch.setattr(config, "SPLIT_DATA_DIR", temp_splits_dir)

    # Return a dictionary with paths and expected counts for easy access in tests

    return {
        "splits_dir": temp_splits_dir,
        "total_rows": len(raw_data_for_splitting),
        "train_rows": 70,  # Based on DEV_SET_SIZE = 0.7 from config
        "val_rows": 15,  # Based on VAL_SET_SIZE = 0.15 from config
        "test_rows": 15,  # Based on TEST_SET_SIZE = 0.15 from config
    }


def test_split_data_creates_files_correctly(setup_test_environment):
    """
    Tests if split_data correctly creates the expected output files.
    """
    # Arrange: The setup_test_environment fixture has already prepared everything.
    data_splits_dir = setup_test_environment["splits_dir"]
    expected_counts = setup_test_environment
    # Import the function *after* the environment has been patched
    from data_split.split_data import split_data_chronologically

    # Act: Run the function we want to test
    split_data_chronologically()

    # Assert: Check that the main files were created in the temp directory
    train_file = data_splits_dir / "train.csv"
    val_file = data_splits_dir / "validation.csv"
    test_file = data_splits_dir / "test.csv"

    assert train_file.exists(), "Training set was not created."
    assert val_file.exists(), "Validation set was not created."
    assert test_file.exists(), "Test set was not created."

    # Assert: Load the created files and check their row counts
    train_df = pd.read_csv(data_splits_dir / "train.csv")
    val_df = pd.read_csv(data_splits_dir / "validation.csv")
    test_df = pd.read_csv(data_splits_dir / "test.csv")

    assert len(train_df) == expected_counts["train_rows"]
    assert len(val_df) == expected_counts["val_rows"]
    assert len(test_df) == expected_counts["test_rows"]
    # Final sanity check
    total_output_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_output_rows == expected_counts["total_rows"]
