import mlflow
import logging
import json
from dotenv import load_dotenv
from prediction_server import config_app
from shared.config import config_gold
from gold_data_preprocessing.missing_value_imputer import SimpleImputer
from gold_data_preprocessing.rare_category_grouper import RareCategoryGrouper
from gold_data_preprocessing.categorical_encoder import CategoricalEncoder
from gold_data_preprocessing.outlier_handling import OutlierTransformer
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler

logger = logging.getLogger(__name__)

load_dotenv()
model_name = config_app.MODEL_NAME
model_version_alias = config_app.MODEL_VERSION_ALIAS


def load_production_model(model_name: str, model_version_alias: str, model_flavor: str):
    """
    Loads the production model from Mlflow model registry.

    Args:
        model_name (str): The name of the MLflow model.
        model_version_alias(str): The alias of registered MLflow model.
    Returns:
        The loaded production model.
    """
    try:
        model_flavor_module = getattr(mlflow, model_flavor)
    except AttributeError:
        raise ValueError(f"{model_flavor} is not a valid model flavor")

    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = model_flavor_module.load_model(model_uri)

    return model


def load_preprocessing_artifacts():
    """
    Load's all preprocessor objects saved by gold pipeline.

    Returns:
        A dictionary of loaded preprocessor objects
    """

    preprocessors = {}

    if config_gold.SIMPLE_IMPUTER_PATH.exists():
        preprocessors["imputer"] = SimpleImputer.load(config_gold.SIMPLE_IMPUTER_PATH)
    if config_gold.RARE_CATEGORY_GROUPER_PATH.exists():
        preprocessors["grouper"] = RareCategoryGrouper.load(
            config_gold.RARE_CATEGORY_GROUPER_PATH
        )
    if config_gold.CATEGORICAL_ENCODER_PATH.exists():
        preprocessors["encoder"] = CategoricalEncoder.load(
            config_gold.CATEGORICAL_ENCODER_PATH
        )
    if config_gold.OUTLIER_HANDLER_PATH.exists():
        preprocessors["outlier_handler"] = OutlierTransformer.load(
            config_gold.OUTLIER_HANDLER_PATH
        )
    if config_gold.POWER_TRANSFORMER_PATH.exists():
        preprocessors["power_transformer"] = PowerTransformer.load(
            config_gold.POWER_TRANSFORMER_PATH
        )
    if config_gold.SCALER_PATH.exists():
        preprocessors["scaler"] = Scaler.load(config_gold.SCALER_PATH)

    if config_gold.GOLD_FINAL_COLS_PATH.exists():
        with open(config_gold.GOLD_FINAL_COLS_PATH, "r") as f:
            preprocessors["final_columns"] = json.load(f)

    return preprocessors
