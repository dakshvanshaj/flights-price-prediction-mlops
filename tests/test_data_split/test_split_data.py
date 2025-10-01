"""
Tests for the data splitting script located in `src.data_split.split_data`.

This test suite verifies that the chronological data splitting logic works correctly,
ensuring that:
1. The data is split into train, validation, and test sets with the correct proportions.
2. The chronological order is strictly maintained between the sets (no data leakage).
3. The script handles file system interactions and errors gracefully.

The tests rely heavily on fixtures to create an isolated, temporary environment,
preventing the tests from interfering with the actual project data.
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np

# The function to be tested
from src.data_split.split_data import split_data_chronologically

# --- Fixtures ---


@pytest.fixture
def raw_data_for_splitting() -> pd.DataFrame:
    """
    Creates a sample DataFrame with an out-of-order date column.

    This fixture is designed to robustly test the sorting logic within the
    `split_data_chronologically` function by ensuring the data is not
    pre-sorted.

    Returns:
        pd.DataFrame: A DataFrame with 100 rows, a 'date' column, and a 'feature' column.
    """
    # Create dates out of order to ensure the sorting logic is properly tested
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="D"))
    np.random.shuffle(dates.values)

    data = {"date": dates, "feature": range(100)}
    return pd.DataFrame(data)


@pytest.fixture
def setup_split_data_env(
    tmp_path: Path, monkeypatch, raw_data_for_splitting: pd.DataFrame
):
    """
    Sets up a temporary file system and patches config paths for testing.

    This powerful fixture does the following:
    1. Creates temporary 'raw' and 'splits' directories using the built-in `tmp_path` fixture.
    2. Saves the `raw_data_for_splitting` DataFrame to a CSV file inside the temporary 'raw' directory.
    3. Uses the `monkeypatch` fixture to redirect the global config variables (e.g., `RAW_CSV_PATH`)
       that are used by the `split_data_chronologically` function to point to our temporary paths.

    This ensures the test is completely isolated from the actual project file system.

    Args:
        tmp_path (Path): A built-in pytest fixture providing a temporary directory path.
        monkeypatch: A built-in pytest fixture for modifying classes, methods, or variables at runtime.
        raw_data_for_splitting (pd.DataFrame): The fixture providing the input data.

    Returns:
        dict: A dictionary containing paths and expected row counts for easy access in tests.
    """
    # 1. Create temporary directories for the test environment
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "splits"
    raw_dir.mkdir()
    split_dir.mkdir()

    # 2. Create the dummy input CSV file that the script will read
    input_csv_path = raw_dir / "flights.csv"
    raw_data_for_splitting.to_csv(input_csv_path, index=False)

    # 3. Intercept the script's file paths and redirect them to our temporary dirs
    monkeypatch.setattr(
        "src.data_split.split_data.config_split.RAW_CSV_PATH", input_csv_path
    )
    monkeypatch.setattr(
        "src.data_split.split_data.config_split.SPLIT_DATA_DIR", split_dir
    )
    monkeypatch.setattr(
        "src.data_split.split_data.core_paths.SPLIT_DATA_DIR", split_dir
    )  # Patch both for safety

    # Return a dictionary of test parameters for the assertions
    return {
        "split_dir": split_dir,
        "total_rows": len(raw_data_for_splitting),
        "train_rows": 70,  # Based on TRAIN_SET_SIZE = 0.7
        "val_rows": 15,  # Based on VAL_SET_SIZE = 0.15
        "test_rows": 15,  # The remainder
    }


# --- Tests ---


def test_split_data_creates_files_with_correct_counts(setup_split_data_env):
    """
    Tests that the split function creates all output files (train, val, test)
    and that they contain the correct number of rows.

    Args:
        setup_split_data_env: The fixture that prepares the temporary file system
                              and provides expected row counts.
    """
    # ARRANGE
    split_dir = setup_split_data_env["split_dir"]
    expected_counts = setup_split_data_env

    # ACT
    split_data_chronologically()

    # ASSERT
    train_file = split_dir / "train.csv"
    val_file = split_dir / "validation.csv"
    test_file = split_dir / "test.csv"

    assert train_file.exists(), "Training file was not created."
    assert val_file.exists(), "Validation file was not created."
    assert test_file.exists(), "Test file was not created."

    # Verify the row counts for each split file
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    assert len(train_df) == expected_counts["train_rows"]
    assert len(val_df) == expected_counts["val_rows"]
    assert len(test_df) == expected_counts["test_rows"]
    assert len(train_df) + len(val_df) + len(test_df) == expected_counts["total_rows"]


def test_split_is_chronological(setup_split_data_env):
    """
    Tests the most critical logic: that the data is split chronologically.

    It asserts that the latest date in the training set is earlier than the
    earliest date in the validation set, and so on, preventing data leakage.

    Args:
        setup_split_data_env: The fixture that prepares the temporary file system.
    """
    # ARRANGE
    split_dir = setup_split_data_env["split_dir"]

    # ACT
    split_data_chronologically()

    # ASSERT
    train_df = pd.read_csv(split_dir / "train.csv", parse_dates=["date"])
    val_df = pd.read_csv(split_dir / "validation.csv", parse_dates=["date"])
    test_df = pd.read_csv(split_dir / "test.csv", parse_dates=["date"])

    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_split_data_handles_missing_file(monkeypatch, caplog):
    """
    Tests that the function logs an error and exits gracefully if the input
    CSV file does not exist.

    Args:
        monkeypatch: Fixture to redirect the script's input path to a non-existent file.
        caplog: Fixture to capture log output to verify the error message.
    """
    # ARRANGE: Point the script's input path to a file that doesn't exist
    non_existent_path = Path("/tmp/this_file_does_not_exist.csv")
    monkeypatch.setattr(
        "src.data_split.split_data.config_split.RAW_CSV_PATH", non_existent_path
    )

    # ACT
    split_data_chronologically()

    # ASSERT: Check that the expected error message was logged
    assert "Failed to load data" in caplog.text
    assert str(non_existent_path) in caplog.text
