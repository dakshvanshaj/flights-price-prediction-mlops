# tests/conftest.py

import pytest
import pandas as pd
from pathlib import Path
import zipfile


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
