import logging
import logging.config
import yaml
from pathlib import Path
from typing import Union
import shutil
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging_from_yaml(
    log_path: Union[str, Path],
    default_yaml_path: str = "logging.yaml",
    default_level=logging.INFO,
):
    """
    Set up logging configuration from a YAML file, overriding the log file path.

    Args:
        log_path (Union[str, Path]): The desired path for the log file, read from main config.
        default_yaml_path (str): The path to the logging configuration file.
        default_level: The log level to use if the config file is not found.
    """
    config_path = Path(default_yaml_path)

    # Ensure the provided log_path is a Path object
    log_path = Path(log_path)

    if config_path.exists():
        try:
            with open(config_path, "rt") as f:
                # Load the YAML configuration into a Python dictionary
                config = yaml.safe_load(f.read())

            # --- OVERRIDE THE LOG FILE PATH ---
            # Modify the dictionary to set the filename for the 'file' handler
            if "file" in config.get("handlers", {}):
                config["handlers"]["file"]["filename"] = str(log_path)
            # --- END OF OVERRIDE ---

            # Ensure the parent directory for the new log path exists
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Configure the logging system using the modified dictionary
            logging.config.dictConfig(config)
            logging.info(
                "Logging configured from %s; outputting to %s",
                default_yaml_path,
                log_path,
            )

        except Exception as e:
            print(f"Error loading logging configuration: {e}")
            logging.basicConfig(level=default_level)
    else:
        print(f"Logging configuration file not found: {config_path}")
        logging.basicConfig(level=default_level)


def handle_file_based_on_validation(
    result, file_path: Path, success_dir: Path, failure_dir: Path
) -> bool:
    """
    Copies a file to a success or failure directory based on a validation result.

    Args:
        result: The validation result object (must have a .success attribute).
        file_path: The full path to the source file to be moved.
        success_dir: The destination directory for successful validations.
        failure_dir: The destination directory for failed validations.

    Returns:
        True if the original validation succeeded and the file operation was
        successful, False otherwise.
    """
    destination_dir = success_dir if result.success else failure_dir

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / file_path.name
        shutil.copy(src=file_path, dst=destination_path)
        logger.info(f"Copied '{file_path.name}' to '{destination_path}'")
        return True
    except (IOError, OSError) as e:
        logger.error(
            f"Failed to copy file '{file_path.name}' to '{destination_dir}'. Error: {e}"
        )
        return False


def save_dataframe_based_on_validation(
    result, df: pd.DataFrame, file_name: str, success_dir: Path, failure_dir: Path
) -> bool:
    """
    Saves a DataFrame to a success or failure directory based on a validation result.

    Args:
        result: The validation result object (must have a .success attribute).
        df: The pandas DataFrame to save.
        file_name: The original filename to use for the saved file.
        success_dir: The destination directory for successful validations.
        failure_dir: The destination directory for failed validations.

    Returns:
        True if the original validation succeeded and the save operation was
        successful, False otherwise.
    """
    destination_dir = success_dir if result.success else failure_dir

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        file_name = file_name + ".parquet"
        destination_path = destination_dir / file_name
        df.to_parquet(destination_path, engine="pyarrow", index=False)
        logger.info(f"Saved DataFrame for '{file_name}' to '{destination_path}'")
        return True
    except (IOError, OSError) as e:
        logger.error(
            f"Failed to save DataFrame for '{file_name}' to '{destination_dir}'. Error: {e}"
        )
        return False


def s_mape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The SMAPE value.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle cases where denominator might be zero to avoid division by zero
    # A common approach is to set the error to 0 if both y_true and y_pred are 0
    # or to a small epsilon if only one is 0.
    # For simplicity here, we'll use a small epsilon for the denominator if it's zero.
    # In a production setting, carefully consider edge cases for zero values.
    smape_values = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(smape_values) * 100


def median_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Median Absolute Percentage Error (MdAPE).

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: The Median Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate individual absolute percentage errors
    # Handle division by zero for y_true values close to zero by returning a large value
    # instead of infinity, similar to sklearn's MAPE handling.
    # A small epsilon is added to the denominator to prevent division by zero.
    epsilon = np.finfo(float).eps
    absolute_percentage_errors = np.abs((y_true - y_pred) / (y_true + epsilon)) * 100

    # Calculate the median of the absolute percentage errors
    mdape = np.median(absolute_percentage_errors)

    return mdape
