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
    Creates a sample DataFrame with 100 rows and an out-of-order date column
    to robustly test the sorting and splitting logic.
    """
    # Create dates out of order to ensure the sorting logic is tested
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="D"))
    np.random.shuffle(dates.values)

    data = {"date": dates, "feature": range(100)}
    return pd.DataFrame(data)


@pytest.fixture
def setup_split_data_env(
    tmp_path: Path, monkeypatch, raw_data_for_splitting: pd.DataFrame
):
    """
    Sets up a temporary file system and patches the config for testing split_data.
    This fixture ensures that the function-under-test reads from and writes to
    temporary directories, isolating the test from the actual project file system.
    """
    # 1. Create temporary directories
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "splits"
    raw_dir.mkdir()
    split_dir.mkdir()

    # 2. Create a dummy input CSV file in the temporary raw directory
    input_csv_path = raw_dir / "flights.csv"
    raw_data_for_splitting.to_csv(input_csv_path, index=False)

    # 3. Monkeypatch the config modules *where they are used* in the script-under-test.
    #    This is the key to fixing the file creation failures.
    monkeypatch.setattr(
        "src.data_split.split_data.config_split.RAW_CSV_PATH", input_csv_path
    )
    monkeypatch.setattr(
        "src.data_split.split_data.config_split.SPLIT_DATA_DIR", split_dir
    )
    monkeypatch.setattr(
        "src.data_split.split_data.core_paths.SPLIT_DATA_DIR", split_dir
    )  # Patch both for safety

    # Return paths and expected counts for easy access in tests
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
    Tests that the split function creates train, validation, and test files
    with the correct number of rows in each.
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

    # Verify row counts for each split
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
    The latest date in the training set must be earlier than the earliest
    date in the validation set, and so on.
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
    Tests that the function logs an error and exits gracefully if the
    input CSV file does not exist.
    """
    # ARRANGE
    non_existent_path = Path("/tmp/this_file_does_not_exist.csv")
    monkeypatch.setattr(
        "src.data_split.split_data.config_split.RAW_CSV_PATH", non_existent_path
    )

    # ACT
    split_data_chronologically()

    # ASSERT
    assert "Failed to load data" in caplog.text
    assert str(non_existent_path) in caplog.text
