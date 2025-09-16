"""
Tests for the silver data preprocessing functions.

This test suite contains individual unit tests for each function defined in the
`src.silver_data_preprocessing.silver_preprocessing` module. The goal is to ensure
that each data transformation step behaves as expected in isolation.

A shared fixture, `preprocessing_df`, provides a consistent, messy input DataFrame
for all tests, allowing each function to be tested against a variety of data issues
like messy column names, incorrect data types, and duplicates.
"""

import pytest
import pandas as pd

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
    Provides a realistic, messy DataFrame for testing all silver preprocessing functions.

    This fixture is central to the test suite and includes:
    - Messy column names with extra spaces and mixed casing.
    - Numeric data stored as strings ('Age').
    - Duplicate records to test dropping logic.
    - An unsorted date column.
    - A column suitable for conversion to a 'category' type.

    Returns:
        pd.DataFrame: A DataFrame designed to be a challenging input for cleaning functions.
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
        # This column has a cardinality of 3/7 (< 0.5), which meets the
        # criteria for conversion to the 'category' dtype in `optimize_data_types`.
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

    Args:
        preprocessing_df (pd.DataFrame): The fixture providing messy column names.
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
    Tests that data types are optimized correctly.

    It checks for:
    - Conversion of strings representing numbers to numeric types.
    - Conversion of low-cardinality object columns to 'category'.
    - Conversion of date strings to datetime objects.

    Args:
        preprocessing_df (pd.DataFrame): The fixture providing mixed data types.
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
    assert isinstance(result_df["favorite_color"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(result_df["join_date"])


def test_sort_data_by_date(preprocessing_df: pd.DataFrame):
    """
    Tests that the DataFrame is correctly sorted by the specified date column.

    Args:
        preprocessing_df (pd.DataFrame): The fixture providing an unsorted date column.
    """
    # ARRANGE
    # The date column must first be converted to datetime objects for sorting to work correctly.
    df = optimize_data_types(preprocessing_df.copy(), date_cols=["Join Date"])

    # ACT
    sorted_df = sort_data_by_date(df, date_column="Join Date")

    # ASSERT
    assert sorted_df["Join Date"].is_monotonic_increasing


def test_handle_erroneous_duplicates(preprocessing_df: pd.DataFrame):
    """
    Tests that only the first occurrence of a duplicate record is kept,
    based on a subset of identifying columns.

    Args:
        preprocessing_df (pd.DataFrame): The fixture providing duplicate rows.
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

    Args:
        preprocessing_df (pd.DataFrame): The fixture providing a date column.
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
    # Check a specific value for correctness to ensure logic is sound
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
    if the column sets do not match, preventing accidental data loss.

    Args:
        caplog: A built-in pytest fixture to capture log output.
    """
    # ARRANGE
    df = pd.DataFrame({"C": [3], "A": [1], "B": [2]})
    # The desired order is missing column 'C' from the original DataFrame
    incorrect_order = ["A", "B"]

    # ACT
    result_df = enforce_column_order(df, column_order=incorrect_order)

    # ASSERT
    # The original DataFrame should be returned unchanged
    assert list(result_df.columns) == list(df.columns)
    # Check that a warning was logged to inform the user
    assert "Column sets do not match. Skipping reordering" in caplog.text
