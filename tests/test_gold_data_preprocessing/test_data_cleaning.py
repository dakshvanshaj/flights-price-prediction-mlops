import pytest
import pandas as pd
import numpy as np
from gold_data_preprocessing.data_cleaning import (
    drop_columns,
    drop_duplicates,
    drop_missing_target_rows,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Provides a standard DataFrame for testing all cleaning functions.
    - Contains columns to be dropped.
    - Contains a full duplicate row (index 2 is a copy of 1).
    - Contains a partial duplicate (index 4 has same 'target' as 3).
    - Contains missing values in the 'target' column.
    """
    data = {
        "feature_1": [1, 2, 2, 3, 4, 5],
        "feature_2": [10.0, 20.0, 20.0, 30.0, 40.0, 50.0],
        "target": [100, 200, 200, 300, 300, np.nan],
        "col_to_drop": ["x", "y", "y", "z", "w", "q"],
        "another_col_to_drop": [True, False, False, True, False, True],
    }
    return pd.DataFrame(data)


# --- Tests for drop_columns ---


def test_drop_columns_successfully(sample_df: pd.DataFrame):
    """
    Tests that specified columns are correctly removed.
    """
    # ARRANGE
    df = sample_df.copy()
    columns_to_drop = ["col_to_drop", "another_col_to_drop"]
    expected_columns = ["feature_1", "feature_2", "target"]

    # ACT
    result_df = drop_columns(df, columns_to_drop=columns_to_drop)

    # ASSERT
    assert list(result_df.columns) == expected_columns


def test_drop_columns_gracefully_handles_nonexistent_column(sample_df: pd.DataFrame):
    """
    Tests that the function does not error out if asked to drop a column
    that is not in the DataFrame.
    """
    # ARRANGE
    df = sample_df.copy()
    columns_to_drop = ["col_to_drop", "nonexistent_col"]
    expected_columns = ["feature_1", "feature_2", "target", "another_col_to_drop"]

    # ACT
    result_df = drop_columns(df, columns_to_drop=columns_to_drop)

    # ASSERT
    assert list(result_df.columns) == expected_columns


# --- Tests for drop_duplicates ---


@pytest.mark.parametrize(
    "subset_cols, keep, expected_rows",
    [
        # Test Case 1: Drop duplicates across all columns (row 2 is the duplicate).
        (None, "first", 5),
        # Test Case 2: Drop duplicates on a subset (row 2 duplicates row 1 on these cols).
        (["feature_1", "feature_2"], "first", 5),
        # Test Case 3: Keep the 'last' occurrence of the full duplicate.
        (None, "last", 5),
        # Test Case 4: Drop all occurrences of full duplicates (removes rows 1 and 2).
        (None, False, 4),
    ],
)
def test_drop_duplicates(
    sample_df: pd.DataFrame, subset_cols: list, keep: str, expected_rows: int
):
    """
    Tests drop_duplicates with various parameters for subset and keep.
    """
    # ARRANGE
    df = sample_df.copy()

    # ACT
    result_df = drop_duplicates(df, subset_cols=subset_cols, keep=keep)

    # ASSERT
    assert len(result_df) == expected_rows


# --- Tests for drop_missing_target_rows ---


def test_drop_missing_target_rows_removes_correct_row(sample_df: pd.DataFrame):
    """
    Tests that only the row(s) with a missing target are dropped.
    """
    # ARRANGE
    df = sample_df.copy()
    target_column = "target"
    initial_rows = len(df)

    # ACT
    result_df = drop_missing_target_rows(df, target_column=target_column)

    # ASSERT
    assert len(result_df) == initial_rows - 1
    assert result_df[target_column].isnull().sum() == 0


def test_drop_missing_target_rows_with_no_missing_values(sample_df: pd.DataFrame):
    """
    Tests that the function does nothing if there are no missing target values.
    """
    # ARRANGE
    df = sample_df.copy().dropna(
        subset=["target"]
    )  # Create a DF with no missing targets
    initial_rows = len(df)

    # ACT
    result_df = drop_missing_target_rows(df, target_column="target")

    # ASSERT
    assert len(result_df) == initial_rows
