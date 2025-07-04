# tests/conftest.py

import pytest
import pandas as pd
from pathlib import Path
import zipfile
import numpy as np


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """
    A fixture that provides a sample DataFrame.
    The scope is 'module' so it's created only once per test module.
    """
    data = {
        "col_a": [1, 2, 3, 4],
        "col_b": [4.0, 5.0, 6.0, 7.0],
        "col_c": ["x", "y", "z", "w"],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def raw_data_for_splitting() -> pd.DataFrame:
    """
    Creates a DataFrame with 100 rows and a date range
    spanning several months, suitable for testing the split_data script.
    """
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = {"date": dates, "value": range(100)}
    return pd.DataFrame(data)


@pytest.fixture
def create_test_file(tmp_path: Path, sample_df: pd.DataFrame):
    """
    A factory fixture that creates data files of various formats
    in a temporary directory.

    It returns a function that the tests can call with the desired format.
    """

    # This is a "factory as a fixture". We return a function.
    def _create_file(file_format: str) -> Path:
        """The actual function that makes the file."""
        file_path = tmp_path / f"test_data.{file_format}"

        if file_format == "csv":
            sample_df.to_csv(file_path, index=False)
        elif file_format == "json":
            sample_df.to_json(file_path, orient="records", lines=True)
        elif file_format == "parquet":
            sample_df.to_parquet(file_path, index=False)
        elif file_format == "feather":
            sample_df.to_feather(file_path)
        elif file_format == "xlsx":
            sample_df.to_excel(file_path, index=False)
        elif file_format == "zip":
            # Create a zip file containing a CSV
            csv_path = tmp_path / "data_in_zip.csv"
            sample_df.to_csv(csv_path, index=False)
            with zipfile.ZipFile(file_path, "w") as zf:
                zf.write(csv_path, arcname="data_in_zip.csv")
        else:
            # Create an empty file for unsupported formats
            file_path.touch()

        return file_path

    return _create_file


@pytest.fixture(scope="module")
def preprocessing_base_df() -> pd.DataFrame:
    """
    Provides a versatile DataFrame for testing multiple preprocessing steps.
    It includes:
    - Mixed data types (int, float, object, datetime).
    - Messy column names for standardization.
    - An unsorted date column.
    - Duplicate rows.
    - Missing values.
    """
    data = {
        "First Name": ["Alvin", "Simon", "Theodore", "Alvin", "Dave"],
        "Last Name": ["Seville", "Seville", "Seville", "Seville", "Seville"],
        "Age": [10, 10, 10, 10, 40],
        "Score": [88.0, 92.5, np.nan, 88.0, 99.9],
        "Join Date": pd.to_datetime(
            ["2023-03-15", "2023-01-20", "2023-02-10", "2023-03-15", "2021-05-01"]
        ),
        "Favorite Color": ["Red", "Blue", "Green", "Red", np.nan],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def imputer_train_df() -> pd.DataFrame:
    """Provides a training DataFrame for fitting the MissingValueHandler."""
    data = {
        "numeric_col": [10, 20, 30, 40, 50],  # median=30
        "category_col": ["A", "B", "A", "A", "C"],  # mode='A'
        "col_with_nan": [1, 2, np.nan, 4, 5],
        "id_col": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def imputer_test_df() -> pd.DataFrame:
    """Provides a test DataFrame with missing values to transform."""
    data = {
        "numeric_col": [15, np.nan, 25, np.nan, 55],
        "category_col": ["B", "B", np.nan, "C", np.nan],
        "col_with_nan": [1, 2, 3, 4, 5],  # No NaNs here to test it's untouched
        "id_col": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)
