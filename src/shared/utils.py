import logging
import logging.config
import yaml
from pathlib import Path
from typing import Union
import shutil
import pandas as pd
from typing import Dict, Any

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


def flatten_params(params: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """
    Flattens a nested dictionary of parameters for clean logging to MLflow.

    Args:
        params: The dictionary of parameters to flatten.
        parent_key: The base key for nested dictionaries.

    Returns:
        A flattened dictionary.
    """
    items = []
    for key, value in params.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_params(value, new_key).items())
        else:
            # Convert lists to strings for better MLflow display
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            items.append((new_key, value))
    return dict(items)
