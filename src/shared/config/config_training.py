# shared/config/config_training.py

import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

TARGET_COLUMN = "price"

# --- Model Library Groupings ---
SKLEARN_MODELS = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SGDRegressor",
    "RandomForestRegressor",
    "SVR",
]
XGBOOST_MODELS = ["XGBRegressor"]
LIGHTGBM_MODELS = ["LGBMRegressor"]

# --- MLflow Mappings for Scalability ---

# Maps model library names to their corresponding MLflow autolog function.
# This avoids long if/elif chains in the training pipeline.
AUTOLOG_MAPPING = {
    "sklearn": mlflow.sklearn.autolog,
    "xgboost": mlflow.xgboost.autolog,
    "lightgbm": mlflow.lightgbm.autolog,
}

# Maps model library names to their corresponding MLflow log_model function.
LOG_MODEL_MAPPING = {
    "sklearn": mlflow.sklearn.log_model,
    "xgboost": mlflow.xgboost.log_model,
    "lightgbm": mlflow.lightgbm.log_model,
}

# Maps model library to the expected keyword argument for the model object.
# e.g., mlflow.sklearn.log_model(sk_model=...) vs mlflow.xgboost.log_model(xgb_model=...)
LOG_MODEL_ARG_NAME = {
    "sklearn": "sk_model",
    "xgboost": "xgb_model",
    "lightgbm": "lgb_model",
}
