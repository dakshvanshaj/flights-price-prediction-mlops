import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Any
import zipfile
import io

# It's good practice to have a logger instance for each module.
logger = logging.getLogger(__name__)


def load_data(file_path: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Loads data from various file formats into a pandas DataFrame.

    This function intelligently detects the file format based on its
    extension and uses the appropriate pandas reader. It supports:
    - .csv
    - .xlsx (Excel)
    - .parquet
    - .json
    - .zip (by extracting and reading the first supported data file within)

    Args:
        file_path: The full path to the data file.
        **kwargs: Arbitrary keyword arguments to pass to the underlying
                  pandas read function (e.g., sep=';' for pd.read_csv).

    Returns:
        A pandas DataFrame containing the loaded data, or None if the
        file cannot be found or an error occurs during loading.
    """
    path = Path(file_path)

    # --- 1. Check if the file exists ---
    if not path.exists():
        logger.error(f"File not found at path: {file_path}")
        return None

    # --- 2. Determine the file format and load the data ---
    suffix = path.suffix.lower()
    logger.info(f"Attempting to load data from '{path.name}' with format '{suffix}'...")

    try:
        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)

        elif suffix == ".xlsx":
            # Note: Requires the 'openpyxl' library to be installed.
            return pd.read_excel(path, **kwargs)

        elif suffix == ".parquet":
            # Note: Requires the 'pyarrow' or 'fastparquet' library.
            return pd.read_parquet(path, **kwargs)

        elif suffix == ".json":
            # Often JSON data is stored as one record per line.
            # Defaulting to lines=True is a common, robust choice.
            return pd.read_json(path, lines=True, **kwargs)

        elif suffix == ".feather":
            # Note: Requires the 'pyarrow' library.
            return pd.read_feather(path, **kwargs)

        elif suffix == ".zip":
            # For zip files, we'll try to find the first data file inside.
            with zipfile.ZipFile(path, "r") as z:
                # Find the first file inside the zip that is a data file we can read.
                file_to_read = None
                for filename in z.namelist():
                    if filename.endswith((".csv", ".json", ".parquet", ".feather")):
                        file_to_read = filename
                        logger.info(f"Found '{file_to_read}' inside the zip archive.")
                        break

                if file_to_read:
                    # Read the file content into an in-memory buffer and pass to pandas.
                    with z.open(file_to_read) as f:
                        # We need to read the file into a buffer pandas can use
                        buffer = io.BytesIO(f.read())
                        # Now, recursively call this same function on the buffer,
                        # but we need to pass the file type information.
                        # For simplicity here, we'll just check the extension again.
                        if file_to_read.endswith(".csv"):
                            return pd.read_csv(buffer, **kwargs)
                        elif file_to_read.endswith(".json"):
                            return pd.read_json(buffer, lines=True, **kwargs)
                        elif file_to_read.endswith(".parquet"):
                            return pd.read_parquet(buffer, **kwargs)
                else:
                    logger.error(
                        f"No supported data file (.csv, .json, .parquet) found inside '{path.name}'"
                    )
                    return None

        else:
            logger.error(f"Unsupported file format: '{suffix}'")
            raise ValueError(f"Unsupported file format: '{suffix}'")

    except Exception as e:
        logger.error(f"Failed to load or parse the file '{path.name}'. Error: {e}")
        return None


if __name__ == "__main__":
    # This block allows you to test the script directly.
    # Create dummy files for testing purposes.

    # Configure basic logging for the test
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Test CSV ---
    pd.DataFrame({"a": [1], "b": [2]}).to_csv("test.csv", index=False)
    df_csv = load_data("test.csv")
    if df_csv is not None:
        print("\nSuccessfully loaded CSV:")
        print(df_csv.head())

    # --- Test Parquet ---
    pd.DataFrame({"c": [3], "d": [4]}).to_parquet("test.parquet")
    df_parquet = load_data("test.parquet")
    if df_parquet is not None:
        print("\nSuccessfully loaded Parquet:")
        print(df_parquet.head())

    # --- Test ZIP containing a CSV ---
    with zipfile.ZipFile("test_archive.zip", "w") as zf:
        zf.writestr("my_data.csv", "col1,col2\n5,6\n7,8")
    df_zip = load_data("test_archive.zip")
    if df_zip is not None:
        print("\nSuccessfully loaded from ZIP:")
        print(df_zip.head())

    # --- Test File Not Found ---
    print("\nTesting non-existent file:")
    load_data("non_existent_file.csv")
