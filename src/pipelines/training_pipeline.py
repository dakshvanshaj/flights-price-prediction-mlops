import mlflow
import pandas as pd
import logging
import argparse
import yaml
import sys
from typing import Dict, Any, Optional

from data_ingestion.data_loader import load_data
from model_evaluation.evaluation import (
    calculate_all_regression_metrics,
    time_based_cross_validation,
    unscale_predictions,
)
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler
from model_training.train import get_model, train_model
from shared.config import config_logging, config_gold, config_training
from shared.utils import setup_logging_from_yaml, flatten_params

logger = logging.getLogger(__name__)


def get_model_library(model_class_name: str) -> Optional[str]:
    """
    Determines the machine learning library for a given model class name.
    """
    if model_class_name in config_training.SKLEARN_MODELS:
        return "sklearn"
    if model_class_name in config_training.XGBOOST_MODELS:
        return "xgboost"
    if model_class_name in config_training.LIGHTGBM_MODELS:
        return "lightgbm"
    logger.warning(f"Could not determine library for model '{model_class_name}'.")
    return None


def _calculate_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, log_prefix: str
) -> None:
    """
    Calculates and logs a comprehensive set of regression metrics to MLflow.
    """
    logger.info(f"Calculating and logging metrics with prefix: '{log_prefix}'...")
    raw_metrics = calculate_all_regression_metrics(y_true, y_pred)
    metrics = {f"{log_prefix}/{k}": v for k, v in raw_metrics.items()}
    mlflow.log_metrics(metrics)
    for name, value in metrics.items():
        logger.info(f"  - {name}: {value:.4f}")


def _evaluate_and_log_set(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    log_prefix: str,
    scaler: Optional[Scaler],
    power_transformer: Optional[PowerTransformer],
    log_scaled_metrics: bool = True,
    log_predictions: bool = False,  # <-- New parameter
) -> None:
    """
    Evaluates a model and logs metrics and optionally prediction artifacts.

    Args:
        ...
        log_predictions: If True, logs true vs. predicted values as a CSV artifact.
    """
    logger.info(f"--- Evaluating model on '{log_prefix}' data ---")
    y_pred = model.predict(X)

    if log_scaled_metrics:
        _calculate_and_log_metrics(y_true, y_pred, log_prefix=f"{log_prefix}/scaled")
    else:
        logger.info(
            "Skipping manual logging of scaled metrics to avoid autolog duplication."
        )

    # --- Conditionally log prediction artifacts ---
    if log_predictions:
        logger.info(f"Logging predictions for '{log_prefix}' to a CSV artifact...")
        predictions_df = pd.DataFrame(
            {"y_true_scaled": y_true, "y_pred_scaled": y_pred}
        )
        csv_path = f"{log_prefix}_predictions.csv"
        predictions_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, "predictions")
        logger.info(f"Successfully logged '{csv_path}'.")

    if scaler and power_transformer:
        y_true_unscaled, y_pred_unscaled = unscale_predictions(
            y_true, y_pred, scaler, power_transformer
        )
        _calculate_and_log_metrics(
            y_true_unscaled, y_pred_unscaled, log_prefix=f"{log_prefix}/unscaled"
        )


def _run_simple_validation(
    model_instance: Any,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    scaler: Optional[Scaler],
    power_transformer: Optional[PowerTransformer],
    model_config: Dict[str, Any],  # <-- Pass model_config
) -> Any:
    """
    Trains a model and evaluates it on training and validation sets.
    """
    logger.info("=" * 25 + " SIMPLE VALIDATION " + "=" * 25)
    model = train_model(train_x, train_y, model_instance)

    _evaluate_and_log_set(
        model,
        train_x,
        train_y,
        "training",
        scaler,
        power_transformer,
        log_scaled_metrics=False,
    )

    # Read the flag from the config for the validation set
    should_log_preds = model_config.get("log_predictions", False)
    _evaluate_and_log_set(
        model,
        val_x,
        val_y,
        "validation",
        scaler,
        power_transformer,
        log_predictions=should_log_preds,
    )

    return model


def _run_cross_validation(
    model_instance: Any,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    cv_config: Dict[str, Any],
    scaler: Optional[Scaler],
    power_transformer: Optional[PowerTransformer],
    model_config: Dict[str, Any],  # <-- Pass model_config
) -> Any:
    """
    Runs time-based cross-validation and retrains a final model on all data.
    """
    logger.info("=" * 25 + " CROSS VALIDATION " + "=" * 25)
    mlflow.log_params(
        {
            f"cv/{k}": v
            for k, v in flatten_params({"cross_validation": cv_config}).items()
        }
    )

    combined_x = pd.concat([train_x, val_x], ignore_index=True)
    combined_y = pd.concat([train_y, val_y], ignore_index=True)

    # ... (Existing CV logic remains the same)
    cv_params = cv_config.copy()
    cv_params.pop("enabled", None)
    cv_results = time_based_cross_validation(
        model=model_instance,
        X=combined_x,
        y=combined_y,
        scaler=scaler,
        power_transformer=power_transformer,
        **cv_params,
    )
    for scale_type, df in cv_results.items():
        if not df.empty:
            logger.info(f"Logging mean/std of {scale_type} CV metrics...")
            mlflow.log_metrics(
                {f"cv/{scale_type}/mean/{k}": v for k, v in df.mean().items()}
            )
            mlflow.log_metrics(
                {f"cv/{scale_type}/std/{k}": v for k, v in df.std().items()}
            )
            csv_path = f"cv_results_{scale_type}.csv"
            df.to_csv(csv_path)
            mlflow.log_artifact(csv_path, "cv_results")

    logger.info("Retraining model on the full dataset (train + validation)...")
    final_model = train_model(combined_x, combined_y, model_instance)

    should_log_preds = model_config.get("log_predictions", False)
    _evaluate_and_log_set(
        final_model,
        combined_x,
        combined_y,
        "final_model_on_all_data",
        scaler,
        power_transformer,
        log_scaled_metrics=False,
        log_predictions=should_log_preds,
    )

    return final_model


def training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    training_params: Dict[str, Any],
    mlflow_params: Dict[str, Any],
    gold_pipeline_params: Dict[str, Any],
):
    """
    Orchestrates the model training, evaluation, and MLflow logging.
    """
    logger.info("--- Starting Training Pipeline ---")

    # === STAGE 1: CONFIG & MLFLOW SETUP ===
    model_config_key = training_params["model_config_to_run"]
    model_config = training_params["models"][model_config_key]
    # ... (rest of the config setup is the same)
    model_class_name = model_config["model_class"]
    model_name = model_config["name"]
    run_name = model_config["run_name"]

    mlflow.set_experiment(mlflow_params["experiment_name"])

    # ... (autolog setup is the same)
    model_library = get_model_library(model_class_name)
    if model_library and model_library in config_training.AUTOLOG_MAPPING:
        config_training.AUTOLOG_MAPPING[model_library](
            log_models=False, log_input_examples=True, silent=True
        )

    # === STAGE 2: DATA PREPARATION ===
    # ... (data prep is the same)
    train_x = train_df.drop(columns=config_training.TARGET_COLUMN)
    train_y = train_df[config_training.TARGET_COLUMN]
    val_x = val_df.drop(columns=config_training.TARGET_COLUMN)
    val_y = val_df[config_training.TARGET_COLUMN]
    if cols_to_drop := model_config.get("drop_multicollinear_cols"):
        train_x = train_x.drop(columns=cols_to_drop, errors="ignore")
        val_x = val_x.drop(columns=cols_to_drop, errors="ignore")
    try:
        scaler = Scaler.load(config_gold.SCALER_PATH)
        power_transformer = PowerTransformer.load(config_gold.POWER_TRANSFORMER_PATH)
    except FileNotFoundError as e:
        scaler, power_transformer = None, None
        logger.warning(
            f"Could not load preprocessors: {e}. Unscaled metrics unavailable."
        )

    # === STAGE 3: MODEL TRAINING & EVALUATION ===
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Starting MLflow Run: '{run_name}' (ID: {run.info.run_id})")
        mlflow.log_params(flatten_params({"gold_pipeline": gold_pipeline_params}))
        mlflow.log_params(flatten_params({model_config_key: model_config}))
        mlflow.set_tags({"model_name": model_name, "model_class": model_class_name})

        model_instance = get_model(model_class_name, model_config["training_params"])

        cv_config = model_config.get("cross_validation", {})
        is_cv_enabled = cv_config.get("enabled", False)
        mlflow.set_tag(
            "run_type", "cross_validation" if is_cv_enabled else "simple_validation"
        )

        if is_cv_enabled:
            final_model = _run_cross_validation(
                model_instance,
                train_x,
                train_y,
                val_x,
                val_y,
                cv_config,
                scaler,
                power_transformer,
                model_config,  # Pass config
            )
        else:
            final_model = _run_simple_validation(
                model_instance,
                train_x,
                train_y,
                val_x,
                val_y,
                scaler,
                power_transformer,
                model_config,  # Pass config
            )

        # ... (log_model_artifact logic is the same)
        if model_config.get("log_model_artifact", False):
            if model_library and model_library in config_training.LOG_MODEL_MAPPING:
                log_model_func = config_training.LOG_MODEL_MAPPING[model_library]
                model_arg_name = config_training.LOG_MODEL_ARG_NAME[model_library]
                log_model_func(
                    **{model_arg_name: final_model},
                    artifact_path="model",
                    registered_model_name=model_name,
                )

    logger.info(f"--- Training Pipeline: COMPLETED for run '{run_name}' ---")


def main():
    """Main function to execute the training pipeline."""
    # ... (main function remains exactly the same)
    setup_logging_from_yaml(
        log_path=config_logging.TRAINING_PIPELINE_LOGS_PATH,
        default_yaml_path=config_logging.LOGGING_YAML,
    )
    logger.info(">>> ORCHESTRATOR: Starting Training Pipeline execution.")
    parser = argparse.ArgumentParser(description="Run the Training Pipeline.")
    parser.add_argument("train_file_name", help="Name of the gold training data file.")
    parser.add_argument(
        "validation_file_name", help="Name of the gold validation data file."
    )
    args = parser.parse_args()
    train_data_path = config_gold.GOLD_PROCESSED_DIR / args.train_file_name
    validation_data_path = config_gold.GOLD_PROCESSED_DIR / args.validation_file_name
    train_df = load_data(train_data_path)
    validation_df = load_data(validation_data_path)
    if train_df is None or validation_df is None:
        sys.exit(1)
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    try:
        training_pipeline(
            train_df=train_df,
            val_df=validation_df,
            training_params=params["training_pipeline"],
            mlflow_params=params["mlflow_params"],
            gold_pipeline_params=params["gold_pipeline"],
        )
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
