# tests/test_data_preprocessing/test_silver_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Import all functions and the class from the script to be tested
from src.silver_data_preprocessing.silver_preprocessing import (
    rename_specific_columns,
    standardize_column_format,
    optimize_data_types,
    sort_data_by_date,
    handle_erroneous_duplicates,
    create_date_features,
    enforce_column_order,
    MissingValueHandler,
    ImputerNotFittedError,
    ImputerLoadError,
)

# --- Tests for Individual Preprocessing Functions ---


def test_rename_specific_columns():
    df = pd.DataFrame({"A": [1], "B": [2]})
    rename_map = {"A": "alpha", "B": "beta"}
    result_df = rename_specific_columns(df, rename_map)
    assert "alpha" in result_df.columns
    assert "beta" in result_df.columns
    assert "A" not in result_df.columns


def test_standardize_column_format(preprocessing_base_df):
    result_df = standardize_column_format(preprocessing_base_df)
    expected_cols = [
        "first_name",
        "last_name",
        "age",
        "score",
        "join_date",
        "favorite_color",
    ]
    assert list(result_df.columns) == expected_cols


def test_optimize_data_types():
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 128, 200, 300],
            "float_col": [1.0, 2.5, 3.5, 4.0, 5.5],
            "obj_col_cat": ["a", "b", "a", "a", "b"],
            "obj_col_high_card": ["x", "y", "z", "w", "v"],
            "date_str": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
        }
    )
    result_df = optimize_data_types(df, date_cols=["date_str"])

    # Assertions based on the corrected data and logic
    assert str(result_df["int_col"].dtype) == "int16"
    assert str(result_df["float_col"].dtype) == "float32"
    assert isinstance(result_df["obj_col_cat"].dtype, pd.CategoricalDtype)
    assert str(result_df["obj_col_high_card"].dtype) == "object"
    assert pd.api.types.is_datetime64_any_dtype(result_df["date_str"])


def test_sort_data_by_date(preprocessing_base_df):
    df_sorted = sort_data_by_date(preprocessing_base_df, date_column="Join Date")
    assert df_sorted["Join Date"].is_monotonic_increasing


def test_handle_erroneous_duplicates(preprocessing_base_df):
    result_df = handle_erroneous_duplicates(
        preprocessing_base_df, subset_cols=["First Name", "Last Name", "Age"]
    )
    assert len(result_df) == 4
    assert len(preprocessing_base_df) == 5


def test_create_date_features(preprocessing_base_df):
    result_df = create_date_features(preprocessing_base_df, date_column="Join Date")
    expected_features = [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
    ]
    for col in expected_features:
        assert col in result_df.columns
    assert result_df["year"].iloc[0] == 2023


def test_enforce_column_order():
    df = pd.DataFrame({"C": [3], "A": [1], "B": [2]})
    correct_order = ["A", "B", "C"]
    result_df = enforce_column_order(df, correct_order)
    assert list(result_df.columns) == correct_order


# --- Tests for the MissingValueHandler Class ---


class TestMissingValueHandler:
    def test_fit_transform_flow(self, imputer_train_df, imputer_test_df):
        # Arrange
        handler = MissingValueHandler()

        # Act
        handler.fit(imputer_train_df)
        transformed_df = handler.transform(imputer_test_df)

        # Assert
        # Expected: NaN in numeric_col is filled with median of train_df (30)
        # Expected: NaN in category_col is filled with mode of train_df ('A')
        assert transformed_df["numeric_col"].isnull().sum() == 0
        assert transformed_df["numeric_col"].iloc[1] == 30.0
        assert transformed_df["category_col"].isnull().sum() == 0
        assert transformed_df["category_col"].iloc[2] == "A"
        assert transformed_df["col_with_nan"].isnull().sum() == 0

    def test_column_specific_strategies(self, imputer_train_df, imputer_test_df):
        # Arrange
        strategies = {"numeric_col": "mean", "category_col": "MISSING"}
        handler = MissingValueHandler(column_strategies=strategies)

        # Act
        handler.fit(imputer_train_df)
        transformed_df = handler.transform(imputer_test_df)

        # Assert
        # Expected: numeric_col filled with mean of train_df (30.0)
        # Expected: category_col filled with the custom string 'MISSING'
        assert transformed_df["numeric_col"].iloc[1] == 30.0
        assert transformed_df["category_col"].iloc[2] == "MISSING"

    def test_exclude_columns(self, imputer_train_df, imputer_test_df):
        # Arrange
        # Add a NaN to the id_col in the test set to ensure it's ignored
        test_df_with_nan_id = imputer_test_df.copy()
        test_df_with_nan_id.loc[0, "id_col"] = np.nan
        handler = MissingValueHandler(exclude_columns=["id_col"])

        # Act
        handler.fit(imputer_train_df)
        transformed_df = handler.transform(test_df_with_nan_id)

        # Assert
        # The NaN in 'id_col' should remain because it was excluded
        assert transformed_df["id_col"].isnull().sum() == 1

    def test_save_and_load_flow(self, tmp_path, imputer_train_df, imputer_test_df):
        # Arrange: Fit and save an imputer
        save_path = tmp_path / "imputer.json"
        original_handler = MissingValueHandler()
        original_handler.fit(imputer_train_df)
        original_handler.save(save_path)

        # Act: Load the imputer and transform data
        loaded_handler = MissingValueHandler.load(save_path)
        transformed_df = loaded_handler.transform(imputer_test_df)
        assert transformed_df["numeric_col"].iloc[1] == 30.0
        assert transformed_df["category_col"].iloc[2] == "A"

    def test_transform_before_fit_raises_error(self):
        handler = MissingValueHandler()
        with pytest.raises(ImputerNotFittedError):
            handler.transform(pd.DataFrame({"A": [1, np.nan]}))

    def test_save_before_fit_raises_error(self, tmp_path):
        handler = MissingValueHandler()
        with pytest.raises(ImputerNotFittedError):
            handler.save(tmp_path / "imputer.json")

    def test_load_nonexistent_file_raises_error(self):
        with pytest.raises(ImputerLoadError):
            MissingValueHandler.load("non_existent_file.json")
