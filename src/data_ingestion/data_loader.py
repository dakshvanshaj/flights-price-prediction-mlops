import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Any
import zipfile
import io

logger = logging.getLogger(__name__)


def _read_data_from_buffer(
    buffer: io.BytesIO, suffix: str, **kwargs: Any
) -> Optional[pd.DataFrame]:
    """
    Reads data from an in-memory buffer based on the file suffix.
    This is an internal helper function.

    Args:
        buffer: The in-memory BytesIO buffer containing the file data.
        suffix: The file extension (e.g., '.csv', '.parquet').
        **kwargs: Additional keyword arguments for the pandas reader.

    Returns:
        A pandas DataFrame or None if the format is unsupported.
    """
    if suffix == ".csv":
        return pd.read_csv(buffer, **kwargs)
    elif suffix == ".xlsx":
        return pd.read_excel(buffer, **kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(buffer, **kwargs)
    elif suffix == ".json":
        return pd.read_json(buffer, lines=True, **kwargs)
    elif suffix == ".feather":
        return pd.read_feather(buffer, **kwargs)

    logger.error(f"Attempted to read unsupported format '{suffix}' from buffer.")
    return None


def load_data(file_path: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Loads data from various file formats into a pandas DataFrame.

    This function intelligently detects the file format based on its
    extension and uses the appropriate pandas reader.

    Supported Formats & Dependencies:
    - .csv: Natively supported by pandas.
    - .xlsx: Requires the 'openpyxl' library.
    - .parquet: Requires the 'pyarrow' or 'fastparquet' library.
    - .json: Natively supported by pandas.
    - .feather: Requires 'pyarrow'.
    - .zip: Extracts and reads the first supported data file within.

    Args:
        file_path: The full path to the data file.
        **kwargs: Arbitrary keyword arguments to pass to the underlying
                  pandas read function (e.g., sep=';' for pd.read_csv).

    Returns:
        A pandas DataFrame containing the loaded data, or None if the
        file cannot be found or an error occurs during loading.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found at path: {file_path}")
        return None

    suffix = path.suffix.lower()
    logger.info(f"Attempting to load data from '{path.name}' with format '{suffix}'...")

    try:
        if suffix == ".zip":
            with zipfile.ZipFile(path, "r") as z:
                supported_files = [
                    f
                    for f in z.namelist()
                    if f.endswith((".csv", ".json", ".parquet", ".feather", ".xlsx"))
                ]
                if not supported_files:
                    logger.error(f"No supported data file found inside '{path.name}'")
                    return None

                file_to_read = supported_files[0]
                logger.info(f"Found '{file_to_read}' inside the zip archive.")

                with z.open(file_to_read) as f:
                    buffer = io.BytesIO(f.read())
                    file_suffix = Path(file_to_read).suffix.lower()
                    return _read_data_from_buffer(buffer, file_suffix, **kwargs)
        else:
            # For non-zip files, read directly from the file path.
            if suffix == ".csv":
                return pd.read_csv(path, **kwargs)
            elif suffix == ".xlsx":
                return pd.read_excel(path, **kwargs)
            elif suffix == ".parquet":
                return pd.read_parquet(path, **kwargs)
            elif suffix == ".json":
                return pd.read_json(path, lines=True, **kwargs)
            elif suffix == ".feather":
                return pd.read_feather(path, **kwargs)
            else:
                logger.error(f"Unsupported file format: '{suffix}'")
                return None

    except pd.errors.ParserError as e:
        logger.error(f"Pandas parsing error for '{path.name}': {e}")
        return None
    except zipfile.BadZipFile:
        logger.error(f"File '{path.name}' is not a valid zip archive or is corrupted.")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading '{path.name}': {e}",
            exc_info=True,
        )
        return None
