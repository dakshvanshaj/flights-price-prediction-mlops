import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.data_ingestion.data_loader import load_data

# List of formats we want to test
SUPPORTED_FORMATS = ["csv", "json", "parquet", "feather", "xlsx", "zip"]


@pytest.mark.parametrize("file_format", SUPPORTED_FORMATS)
def test_load_data_supported_formats(
    file_format: str, sample_df: pd.DataFrame, create_test_file
):
    """
    Tests that `load_data` can successfully load all supported file formats.
    This test is parameterized to run for each format in SUPPORTED_FORMATS.
    """
    # Arrange: Create the test file using our fixture factory
    test_file_path = create_test_file(file_format)

    # Act: Load the data from the created file
    loaded_df = load_data(str(test_file_path))

    # Assert: Check that the loaded data is correct
    assert loaded_df is not None, "The loaded DataFrame should not be None."
    assert isinstance(loaded_df, pd.DataFrame), (
        "The result should be a pandas DataFrame."
    )

    # For Excel, pandas reads back integers as floats if there are NaNs,
    # so we may need to adjust types before comparison.
    # In our case, the sample data is clean, so a direct comparison works.

    # Use pandas' testing utility for robust DataFrame comparison
    assert_frame_equal(sample_df, loaded_df, check_dtype=False)


def test_load_non_existent_file():
    """
    Tests that `load_data` returns None when the file does not exist.
    """
    # Act
    result = load_data("non_existent_file.csv")
    # Assert
    assert result is None


def test_load_unsupported_format(create_test_file):
    """
    Tests that `load_data` returns None for an unsupported file format.
    """
    # Arrange
    test_file_path = create_test_file("txt")  # Create a .txt file
    # Act
    result = load_data(str(test_file_path))
    # Assert
    assert result is None
