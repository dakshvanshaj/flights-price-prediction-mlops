import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import logging
import argparse
import yaml
import sys
from typing import Dict, Any
from data_ingestion.data_loader import load_data
from model_training.train import train_model
from shared.config import config_logging, config_gold, config_training, core_paths
from shared.utils import (
    setup_logging_from_yaml,
    s_mape,
    median_absolute_percentage_error,
    flatten_params,
)

# from sklearn.metrics import (
#     max_error,
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     median_absolute_error,
#     mean_squared_error,
#     root_mean_squared_error,
#     r2_score,
# )
from mlflow.exceptions import MlflowException

# Create a logger object for this module
logger = logging.getLogger(__name__)


def _evaluate_and_log_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculates custom regression metrics and logs them to MLflow.
    Note: Autologging handles standard metrics, this is for any additional custom ones.

    Args:
        y_true: The true target values.
        y_pred: The predicted target values.

    Returns:
        A dictionary of calculated metrics.
    """
    logger.info("Calculating and logging custom validation metrics...")
    metrics = {
        "symmetric_mean_absolute_percentage_error": s_mape(y_true, y_pred),
        "median_absolute_percentage_error": median_absolute_percentage_error(
            y_true, y_pred
        ),
        # Add any other custom metrics here. Standard metrics like MSE, MAE, R2
        # are already captured by autologging.
    }

    for name, value in metrics.items():
        logger.info(f"  - {name}: {value:.4f}")

    mlflow.log_metrics(metrics)
    logger.info("Successfully logged custom metrics to MLflow.")
    return metrics


def training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    training_params: Dict[str, Any],
    mlflow_params: Dict[str, Any],
    gold_pipeline_params: Dict[str, Any],
):
    """Orchestrates the model training, evaluation, and MLflow logging."""
    logger.info("--- Starting Training Pipeline ---")

    # === STAGE 1: MLFLOW SETUP ===
    logger.info("=" * 25 + " STAGE 1/4: MLFLOW SETUP " + "=" * 25)
    experiment_name = mlflow_params["experiment_name"]
    run_name = mlflow_params["run_name"]
    model_to_train = training_params["model_to_train"]

    # Set the experiment. MLflow will use the URI from the environment variable.
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: '{experiment_name}'")
    logger.info(f"MLflow run name: '{run_name}'")

    # Enable Autologging for the corresponding framework
    # This will automatically log params, metrics, and the model artifact.
    logger.info("Enabling MLflow autologging...")
    if model_to_train in config_training.SKLEARN_MODELS:
        mlflow.sklearn.autolog(log_models=False, log_input_examples=True, silent=False)
    elif model_to_train in config_training.XGBOOST_MODELS:
        mlflow.xgboost.autolog(log_models=False, log_input_examples=True, silent=False)
    elif model_to_train in config_training.LIGHTGBM_MODELS:
        mlflow.lightgbm.autolog(log_models=False, log_input_examples=True, silent=False)
    else:
        raise ValueError(f"Unsupported model type for autologging: {model_to_train}")
    logger.info("MLflow autologging enabled.")

    # === STAGE 2: DATA PREPARATION ===
    logger.info("=" * 25 + " STAGE 2/4: DATA PREPARATION " + "=" * 25)
    train_x = train_df.drop(columns=config_training.TARGET_COLUMN)
    train_y = train_df[config_training.TARGET_COLUMN]
    val_x = val_df.drop(columns=config_training.TARGET_COLUMN)
    val_y = val_df[config_training.TARGET_COLUMN]
    logger.info("Successfully separated features and target.")

    # --- Handle Multicollinearity by Dropping Specified Columns ---
    cols_to_drop = training_params.get("drop_multicollinear_cols")
    if cols_to_drop:
        logger.info(
            f"Dropping multicollinear columns as per params.yaml: {cols_to_drop}"
        )
        train_x = train_x.drop(columns=cols_to_drop, errors="ignore")
        val_x = val_x.drop(columns=cols_to_drop, errors="ignore")
        logger.info("Columns dropped successfully from training and validation sets.")
    else:
        logger.info(
            "No multicollinear columns specified to drop. Keeping all features."
        )

    with mlflow.start_run(run_name=run_name) as run:
        # Log preprocessing parameters from gold_pipeline
        logger.info("Logging preprocessing parameters from gold_pipeline...")
        flat_gold_params = flatten_params(gold_pipeline_params)
        mlflow.log_params(flat_gold_params)
        logger.info("Successfully logged preprocessing parameters.")

        # Log any extra parameters not captured by autologging
        mlflow.log_param(
            "dropped_multicollinear_cols", cols_to_drop if cols_to_drop else "None"
        )
        logger.info(f"Starting MLflow Run ID: {run.info.run_id}")

        # === STAGE 3: MODEL TRAINING ===
        logger.info("=" * 25 + " STAGE 3/4: MODEL TRAINING " + "=" * 25)
        model = train_model(
            train_x, train_y, model_to_train, training_params["training_params"]
        )

        # === STAGE 4: MODEL EVALUATION ===
        logger.info("=" * 25 + " STAGE 4/4: MODEL EVALUATION " + "=" * 25)
        # Autolog will capture standard metrics on prediction.
        # We call our custom function for any additional metrics.
        y_pred = model.predict(val_x)
        _evaluate_and_log_metrics(val_y, y_pred)

    logger.info(
        f"--- Training Pipeline: COMPLETED for model '{training_params['name']}' ---"
    )
    return model


def main():
    # --- SETUP LOGGING ---
    setup_logging_from_yaml(
        log_path=config_logging.TRAINING_PIPELINE_LOGS_PATH,
        default_level=logging.DEBUG,
        default_yaml_path=config_logging.LOGGING_YAML,
    )

    logger.info(">>> ORCHESTRATOR: Starting Training Pipeline execution.")
    parser = argparse.ArgumentParser(description="Run the Training Pipeline.")
    parser.add_argument(
        "train_file_name",
        type=str,
        help="The name of the gold training data file.",
    )
    parser.add_argument(
        "validation_file_name",
        type=str,
        help="The name of the gold validation data file.",
    )
    args = parser.parse_args()

    train_data_path = config_gold.GOLD_PROCESSED_DIR / args.train_file_name
    validation_data_path = config_gold.GOLD_PROCESSED_DIR / args.validation_file_name

    # Load Datasets
    logger.info(f"Loading training data from: {train_data_path}")
    train_df = load_data(train_data_path)
    if train_df is None:
        logger.critical(f"Failed to load data from {train_data_path}. Aborting.")
        sys.exit(1)

    logger.info(f"Loading validation data from: {validation_data_path}")
    validation_df = load_data(validation_data_path)
    if validation_df is None:
        logger.critical(f"Failed to load data from {validation_data_path}. Aborting.")
        sys.exit(1)

    logger.info("Loading pipeline parameters from params.yaml")
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    training_params = params["training_pipeline"]
    mlflow_params = params["mlflow_params"]

    try:
        training_pipeline(
            train_df=train_df,
            val_df=validation_df,
            training_params=training_params,
            mlflow_params=mlflow_params,
            gold_pipeline_params=params["gold_pipeline"],
        )
        logger.info(
            ">>> ORCHESTRATOR: Training Pipeline execution finished successfully."
        )
    except (MlflowException, ValueError, FileNotFoundError) as e:
        logger.critical(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
