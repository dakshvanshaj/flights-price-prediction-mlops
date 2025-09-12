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

from typing import List, Optional

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


def get_model(
    model_to_train: str,
    model_params: Dict[str, Any],
) -> Any:
    """
    Instantiates a regression model from its string name and parameters.

    Args:
        model_to_train: The name of the model to instantiate. Must be a key
                        in the MODEL_CLASS_MAP.
        model_params: Hyperparameters for the model.

    Returns:
        An unfitted regressor model instance.

    Raises:
        ValueError: If the specified model_to_train is not supported.
    """
    if not (model_class := MODEL_CLASS_MAP.get(model_to_train)):
        raise ValueError(
            f"Unsupported model: '{model_to_train}'. Supported models are: "
            f"{list(MODEL_CLASS_MAP.keys())}"
        )

    logger.info(f"Instantiating model: {model_to_train} with params: {model_params}")
    return model_class(**model_params)


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    model: Any,
    categorical_features: Optional[List[str]] = None,
) -> Any:
    """
    Trains a given regression model instance.

    Args:
        train_x: The training features.
        train_y: The training target.
        model: An unfitted, instantiated regressor model instance.
        categorical_features: Optional list of column names to be treated as
                              categorical by the model (e.g., for LightGBM).

    Returns:
        A trained regressor model.
    """
    logger.info(f"Training model: {model.__class__.__name__}...")

    # Check if the model is a LightGBM model and if categorical features are provided
    if isinstance(model, LGBMRegressor) and categorical_features:
        logger.info(f"Passing categorical features to LightGBM: {categorical_features}")
        model.fit(train_x, train_y, categorical_feature=categorical_features)
    else:
        model.fit(train_x, train_y)

    logger.info("Model training complete.")
    return model
