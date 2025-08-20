import logging
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from typing import Dict, Any

logger = logging.getLogger(__name__)


# A mapping from model names to their respective classes for easy instantiation.
MODEL_CLASS_MAP: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "SGDRegressor": SGDRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "SVR": SVR,
    "XGBRegressor": XGBRegressor,
    "LGBMRegressor": LGBMRegressor,
}


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    model_to_train: str,
    model_params: Dict[str, Any],
):
    """
    Instantiates and trains a specified regression model.

    Args:
        train_x: The training features.
        train_y: The training target.
        model_to_train: The name of the model to train.
        model_params: Hyperparameters for the model.

    Returns:
        A trained regressor model.

    Raises:
        ValueError: If the specified model_to_train is not supported.
    """
    model_class = MODEL_CLASS_MAP.get(model_to_train)
    if not model_class:
        raise ValueError(
            f"Unsupported model type: '{model_to_train}'. "
            f"Supported models are: {list(MODEL_CLASS_MAP.keys())}"
        )

    logger.info(f"Instantiating model: {model_to_train} with params: {model_params}")
    model = model_class(**model_params)

    logger.info(f"Training model: {model_to_train}...")
    model.fit(train_x, train_y)
    logger.info("Model training complete.")

    return model
