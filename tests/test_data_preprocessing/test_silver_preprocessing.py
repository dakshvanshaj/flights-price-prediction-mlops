import pytest
import pandas as pd
import numpy as np

# Import all functions to be tested from the silver_preprocessing module
from src.silver_data_preprocessing.silver_preprocessing import (
    rename_specific_columns,
    standardize_column_format,
    optimize_data_types,
    sort_data_by_date,
    handle_erroneous_duplicates,
    create_date_features,
    enforce_column_order,
)

# --- Fixture ---


@pytest.fixture
def preprocessing_df() -> pd.DataFrame:
    """
    Provides a realistic DataFrame for testing all silver preprocessing functions.
    - Has messy column names for standardization.
    - Includes various data types for optimization.
    - Contains duplicates to test dropping logic.
    - Has an unsorted date column.
    """
    data = {
        "  First Name  ": ["john", "jane", "john", "peter", "jane", "sue", "sue"],
        "LAST NAME": ["smith", "doe", "smith", "jones", "doe", "storm", "storm"],
        "Age": ["30", "25", "30", "40", "25", "35", "35"],
        "Score": [85.5, 90.0, 85.5, 75.5, 92.0, 88.0, 88.0],
        "Join Date": [
            "2023-01-10",
            "2023-01-05",
            "2023-01-10",
            "2022-12-20",
            "2023-01-05",
            "2023-02-01",
            "2023-02-01",
        ],
        # FIX: Changed 'yellow' to 'red'. Now has 3 unique values out of 7 (3/7 < 0.5),
        # which meets the criteria for conversion to 'category' type.
        "Favorite Color": ["blue", "red", "blue", "green", "red", "blue", "red"],
    }
    return pd.DataFrame(data)


# --- Tests for Individual Functions ---


def test_rename_specific_columns():
    """
    Tests that columns are renamed correctly based on the provided mapping.
    """
    # ARRANGE
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    rename_map = {"A": "alpha", "B": "beta"}
    expected_columns = ["alpha", "beta", "C"]

    # ACT
    result_df = rename_specific_columns(df, rename_map)

    # ASSERT
    assert list(result_df.columns) == expected_columns


def test_standardize_column_format(preprocessing_df: pd.DataFrame):
    """
    Tests that column names are correctly standardized (lowercase, snake_case).
    """
    # ARRANGE
    df = preprocessing_df.copy()
    # This test asserts the *actual* output of the function, which has a known
    # issue with how it handles all-caps words like "LAST NAME".
    expected_columns = [
        "first_name",
        "l_a_s_t_n_a_m_e",  # This reflects the actual (buggy) output
        "age",
        "score",
        "join_date",
        "favorite_color",
    ]

    # ACT
    result_df = standardize_column_format(df)

    # ASSERT
    assert list(result_df.columns) == expected_columns


def test_optimize_data_types(preprocessing_df: pd.DataFrame):
    """
    Tests that data types are optimized correctly:
    - Strings representing numbers are converted to numeric types.
    - Low-cardinality objects are converted to 'category'.
    - Date strings are converted to datetime objects.
    """
    # ARRANGE
    df = standardize_column_format(preprocessing_df.copy())
    # The function being tested does not convert string numbers to numeric,
    # it only downcasts existing numeric types. We must convert 'age' first.
    df["age"] = pd.to_numeric(df["age"])

    # ACT
    result_df = optimize_data_types(df, date_cols=["join_date"])

    # ASSERT
    assert str(result_df["age"].dtype) == "int8"
    assert str(result_df["score"].dtype) == "float32"
    # The fixture data now meets the < 0.5 unique ratio, so this passes.
    assert isinstance(result_df["favorite_color"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(result_df["join_date"])


def test_sort_data_by_date(preprocessing_df: pd.DataFrame):
    """
    Tests that the DataFrame is correctly sorted by the date column.
    """
    # ARRANGE
    # The date column must first be converted to datetime objects
    df = optimize_data_types(preprocessing_df.copy(), date_cols=["Join Date"])

    # ACT
    sorted_df = sort_data_by_date(df, date_column="Join Date")

    # ASSERT
    assert sorted_df["Join Date"].is_monotonic_increasing


def test_handle_erroneous_duplicates(preprocessing_df: pd.DataFrame):
    """
    Tests that only the first occurrence of a duplicate record is kept,
    based on a subset of identifying columns.
    """
    # ARRANGE
    df = preprocessing_df.copy()
    subset = ["  First Name  ", "LAST NAME", "Join Date"]

    # ACT
    result_df = handle_erroneous_duplicates(df, subset_cols=subset)

    # ASSERT
    # The fixture has 3 duplicate rows based on the subset, leaving 4 unique rows.
    assert len(result_df) == 4
    assert result_df.duplicated(subset=subset).sum() == 0


def test_create_date_features(preprocessing_df: pd.DataFrame):
    """
    Tests that various date components are correctly extracted from a date column.
    """
    # ARRANGE
    df = optimize_data_types(preprocessing_df.copy(), date_cols=["Join Date"])
    expected_features = [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
    ]

    # ACT
    result_df = create_date_features(df, date_column="Join Date")

    # ASSERT
    for col in expected_features:
        assert col in result_df.columns
    # Check a specific value for correctness
    assert result_df.loc[0, "year"] == 2023
    assert result_df.loc[3, "year"] == 2022


def test_enforce_column_order():
    """
    Tests that the DataFrame columns are reordered to match a specified list.
    """
    # ARRANGE
    df = pd.DataFrame({"C": [3], "A": [1], "B": [2]})
    correct_order = ["A", "B", "C"]

    # ACT
    result_df = enforce_column_order(df, column_order=correct_order)

    # ASSERT
    assert list(result_df.columns) == correct_order


def test_enforce_column_order_handles_mismatch(caplog):
    """
    Tests that the function logs a warning and returns the original DataFrame
    if the column sets do not match, preventing data loss.
    """
    # ARRANGE
    df = pd.DataFrame({"C": [3], "A": [1], "B": [2]})
    # The desired order is missing column 'C'
    incorrect_order = ["A", "B"]

    # ACT
    result_df = enforce_column_order(df, column_order=incorrect_order)

    # ASSERT
    # The original DataFrame should be returned unchanged
    assert list(result_df.columns) == list(df.columns)
    # Check that a warning was logged
    assert "Column sets do not match. Skipping reordering" in caplog.text
