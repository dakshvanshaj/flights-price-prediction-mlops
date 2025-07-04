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
    """
    This fixture prepares a controlled environment for testing split_data.
    1. Creates temporary input and output directories ("sandbox").
    2. Creates a dummy input CSV file inside the sandbox.
    3. Uses monkeypatch to redirect the script's config variables to the sandbox.
    """
    # 1. Create temporary directories that mirror the structure in config.py
    temp_input_dir = tmp_path / "raw_data"
    temp_input_dir.mkdir()

    temp_initial_splits_dir = tmp_path / "_initial_data_splits"
    temp_initial_splits_dir.mkdir()

    temp_drift_dir = temp_initial_splits_dir / "drift_simulation_data"
    temp_drift_dir.mkdir()

    # 2. Create a dummy input file using the raw_data_for_splitting fixture
    input_csv = temp_input_dir / "flights.csv"
    raw_data_for_splitting.to_csv(input_csv, index=False)

    # 3. Use monkeypatch to override config values for the duration of a test.
    # We import the config module that the script-under-test will use.
    # This requires `pythonpath = ["src"]` in your pyproject.toml
    import shared.config as config

    monkeypatch.setattr(config, "INPUT_CSV_PATH", input_csv)
    monkeypatch.setattr(config, "INITIAL_DATA_SPLITS", temp_initial_splits_dir)
    monkeypatch.setattr(config, "DRIFT_SIMULATION_DIR", temp_drift_dir)

    # Return a dictionary with paths and expected counts for easy access in tests
    return {
        "data_splits_dir": temp_initial_splits_dir,
        "drift_dir": temp_drift_dir,
        "total_rows": len(raw_data_for_splitting),
        "dev_rows": 70,  # Based on DEV_SET_SIZE = 0.7 from config
        "eval_rows": 15,  # Based on EVAL_SET_SIZE = 0.15 from config
        "drift_rows": 15,  # The remainder of the 100 rows
    }


def test_split_data_creates_files(setup_test_environment):
    """
    Tests if split_data correctly creates the expected output files.
    """
    # Arrange: The setup_test_environment fixture has already prepared everything.
    data_splits_dir = setup_test_environment["data_splits_dir"]
    drift_dir = setup_test_environment["drift_dir"]

    # Import the function *after* the environment has been patched
    from data_split.split_data import split_data

    # Act: Run the function we want to test
    split_data()

    # Assert: Check that the main files were created in the temp directory
    dev_file = data_splits_dir / "development_data.csv"
    eval_file = data_splits_dir / "evaluation_holdout.csv"

    assert dev_file.exists(), "Development set was not created."
    assert eval_file.exists(), "Evaluation hold-out set was not created."

    # --- FIXED ASSERTIONS ---
    # The last 15 days of our 100-day sample span from late March to early April.
    # Therefore, we expect TWO monthly drift files.
    drift_files = sorted(list(drift_dir.glob("*.csv")))
    assert len(drift_files) == 2, "Expected exactly two monthly drift files."
    assert drift_files[0].name == "flights_2023-03.csv"
    assert drift_files[1].name == "flights_2023-04.csv"


def test_split_data_row_counts(setup_test_environment):
    """
    Tests if the row counts in the created files match the expected splits.
    """
    # Arrange: The fixture sets up the environment and provides expected values
    data_splits_dir = setup_test_environment["data_splits_dir"]
    drift_dir = setup_test_environment["drift_dir"]
    expected_counts = setup_test_environment

    # Import the function *after* the environment has been patched
    from data_split.split_data import split_data

    # Act: Run the function
    split_data()

    # Assert: Load the created files and check their row counts
    dev_df = pd.read_csv(data_splits_dir / "development_data.csv")
    eval_df = pd.read_csv(data_splits_dir / "evaluation_holdout.csv")

    # --- FIXED LOGIC ---
    # Load all created drift files and concatenate them to get the total drift rows
    drift_files = sorted(list(drift_dir.glob("*.csv")))
    all_drift_dfs = [pd.read_csv(f) for f in drift_files]
    combined_drift_df = pd.concat(all_drift_dfs, ignore_index=True)

    assert len(dev_df) == expected_counts["dev_rows"]
    assert len(eval_df) == expected_counts["eval_rows"]
    assert len(combined_drift_df) == expected_counts["drift_rows"]

    # Final sanity check
    total_output_rows = len(dev_df) + len(eval_df) + len(combined_drift_df)
    assert total_output_rows == expected_counts["total_rows"]
