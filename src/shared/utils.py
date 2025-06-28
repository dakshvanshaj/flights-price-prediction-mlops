import logging
import logging.config
import yaml
from pathlib import Path
from typing import Union


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
