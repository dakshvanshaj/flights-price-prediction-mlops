"""
Utility functions for loading the production model and preprocessing artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
from dotenv import load_dotenv

from gold_data_preprocessing.categorical_encoder import CategoricalEncoder
from gold_data_preprocessing.missing_value_imputer import SimpleImputer
from gold_data_preprocessing.outlier_handling import OutlierTransformer
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.rare_category_grouper import RareCategoryGrouper
from gold_data_preprocessing.scaler import Scaler
from shared.config import config_gold

logger = logging.getLogger(__name__)

# Load environment variables from a .env file if it exists
load_dotenv()


def load_production_model(
    model_name: str, model_version_alias: str, model_flavor: str
) -> Any:
    """
    Loads a production-tagged model from the MLflow Model Registry.

    Args:
        model_name: The registered name of the MLflow model.
        model_version_alias: The alias of the model version (e.g., 'production').
        model_flavor: The MLflow flavor of the model (e.g., 'lightgbm', 'sklearn').

    Returns:
        The loaded production model object.

    Raises:
        ValueError: If the model flavor is not a valid MLflow flavor.
    """
    logger.info(
        f"Loading model '{model_name}@{model_version_alias}' from MLflow Registry..."
    )
    try:
        model_flavor_module = getattr(mlflow, model_flavor)
    except AttributeError:
        logger.error(f"Invalid MLflow model flavor specified: '{model_flavor}'")
        raise ValueError(f"'{model_flavor}' is not a valid model flavor.")

    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = model_flavor_module.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model


def load_preprocessing_artifacts() -> Dict[str, Any]:
    """
    Loads all preprocessing objects saved by the gold pipeline.

    This function checks for the existence of each artifact before loading to
    dynamically support different preprocessing pipelines (e.g., tree vs. non-tree models).

    Returns:
        A dictionary mapping artifact names to their loaded objects.
    """
    preprocessors = {}
    logger.info("Loading preprocessing artifacts...")

    # Define a mapping from artifact name to its path and loading class
    artifact_map = {
        "imputer": (config_gold.SIMPLE_IMPUTER_PATH, SimpleImputer),
        "grouper": (config_gold.RARE_CATEGORY_GROUPER_PATH, RareCategoryGrouper),
        "encoder": (config_gold.CATEGORICAL_ENCODER_PATH, CategoricalEncoder),
        "outlier_handler": (config_gold.OUTLIER_HANDLER_PATH, OutlierTransformer),
        "power_transformer": (config_gold.POWER_TRANSFORMER_PATH, PowerTransformer),
        "scaler": (config_gold.SCALER_PATH, Scaler),
    }

    for name, (path, loader_class) in artifact_map.items():
        if path.exists():
            logger.info(f"Loading artifact: '{name}' from {path}")
            preprocessors[name] = loader_class.load(path)
        else:
            logger.warning(f"Artifact '{name}' not found at {path}. Skipping.")

    # Load the final columns list, which is essential
    if config_gold.GOLD_FINAL_COLS_PATH.exists():
        logger.info(f"Loading final columns from {config_gold.GOLD_FINAL_COLS_PATH}")
        with open(config_gold.GOLD_FINAL_COLS_PATH, "r") as f:
            preprocessors["final_columns"] = json.load(f)
    else:
        logger.error(
            "Critical artifact 'final_columns' not found. This may cause errors."
        )

    logger.info("Preprocessing artifacts loading complete.")
    return preprocessors
