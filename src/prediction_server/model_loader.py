import mlflow
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()
model_name = "LGBMR_Flights_Price"
model_version_alias = "champion"


def load_production_model(model_name: str, model_version_alias: str, model_flavor: str):
    """
    Loads the production model from Mlflow model registry.

    Args:
        model_name (str): The name of the MLflow model.
        model_version_alias(str): The alias of registered MLflow model.
    Returns:
        The loaded model.
    """
    try:
        model_flavor_module = getattr(mlflow, model_flavor)
    except AttributeError:
        raise ValueError(f"{model_flavor} is not a valid model flavor")

    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = model_flavor_module.load_model(model_uri)

    return model
