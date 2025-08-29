# -*- coding: utf-8 -*-
"""
Orchestration script for the model training pipeline.

This script manages the end-to-end process of training, evaluating, and logging
a machine learning model, including a final evaluation on a hold-out test set.
"""

import argparse
import logging
import sys
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

from data_ingestion.data_loader import load_data
from model_evaluation.evaluation import (
    calculate_all_regression_metrics,
    time_based_cross_validation,
    unscale_predictions,
)
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler
from model_training.train import get_model, train_model
from shared.config import config_gold, config_logging, config_training
from shared.utils import flatten_params, setup_logging_from_yaml

# Initialize logger for this module
logger = logging.getLogger(__name__)


def get_model_library(model_class_name: str) -> Optional[str]:
    """
    Determines the ML library for a given model class name.

    Args:
        model_class_name: The class name of the model (e.g., "LinearRegression").

    Returns:
        A string representing the library name (e.g., "sklearn") or None if not found.
    """
    if model_class_name in config_training.SKLEARN_MODELS:
        return "sklearn"
    if model_class_name in config_training.XGBOOST_MODELS:
        return "xgboost"
    if model_class_name in config_training.LIGHTGBM_MODELS:
        return "lightgbm"

    logger.warning(f"Could not determine ML library for model '{model_class_name}'.")
    return None


def _calculate_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, log_prefix: str
) -> None:
    """
    Calculates and logs a set of regression metrics to MLflow.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        log_prefix: A string to prepend to metric names in MLflow.
    """
    logger.info(f"Calculating and logging metrics with prefix: '{log_prefix}'...")
    raw_metrics = calculate_all_regression_metrics(y_true, y_pred)
    metrics_to_log = {f"{log_prefix}/{k}": v for k, v in raw_metrics.items()}

    mlflow.log_metrics(metrics_to_log)
    for name, value in metrics_to_log.items():
        logger.info(f"  - {name}: {value:.4f}")


def _evaluate_and_log_set(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    log_prefix: str,
    scaler: Optional[Scaler],
    power_transformer: Optional[PowerTransformer],
    log_predictions: bool = False,
) -> None:
    """
    Evaluates a model on a given dataset and logs all relevant information.

    Args:
        model: The trained model instance.
        X: The feature DataFrame for evaluation.
        y_true: The ground truth Series for evaluation.
        log_prefix: Prefix for naming metrics and artifacts (e.g., "validation").
        scaler: The fitted Scaler object for inverse transformation.
        power_transformer: The fitted PowerTransformer for inverse transformation.
        log_predictions: If True, logs predictions as a JSON artifact.
    """
    logger.info(f"--- Evaluating model on '{log_prefix}' data ---")
    y_pred = model.predict(X)

    _calculate_and_log_metrics(y_true, y_pred, log_prefix=f"{log_prefix}/scaled")

    if scaler and power_transformer:
        y_true_unscaled, y_pred_unscaled = unscale_predictions(
            y_true, y_pred, scaler, power_transformer
        )
        _calculate_and_log_metrics(
            y_true_unscaled, y_pred_unscaled, log_prefix=f"{log_prefix}/unscaled"
        )

    if log_predictions:
        logger.info(f"Logging prediction artifacts for '{log_prefix}'...")
        scaled_preds_dict = {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}
        mlflow.log_dict(
            scaled_preds_dict, f"predictions/{log_prefix}_scaled_predictions.json"
        )
        if scaler and power_transformer:
            unscaled_preds_dict = {
                "y_true": y_true_unscaled.tolist(),
                "y_pred": y_pred_unscaled.tolist(),
            }
            mlflow.log_dict(
                unscaled_preds_dict,
                f"predictions/{log_prefix}_unscaled_predictions.json",
            )


def _run_simple_validation(
    model_instance: Any,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    scaler: Optional[Scaler],
    power_transformer: Optional[PowerTransformer],
    model_config: Dict[str, Any],
) -> Any:
    """
    Trains a model and evaluates it on training and validation sets.

    Returns:
        The trained model instance.
    """
    logger.info("=" * 25 + " SIMPLE VALIDATION " + "=" * 25)
    model = train_model(train_x, train_y, model_instance)

    _evaluate_and_log_set(
        model, train_x, train_y, "training", scaler, power_transformer
    )

    should_log_preds = model_config.get("log_predictions", False)
    _evaluate_and_log_set(
        model, val_x, val_y, "validation", scaler, power_transformer, should_log_preds
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
    model_config: Dict[str, Any],
) -> Any:
    """
    Performs time-based cross-validation and retrains a final model on all data.

    Returns:
        The final model instance trained on the combined dataset.
    """
    logger.info("=" * 25 + " CROSS VALIDATION " + "=" * 25)

    # --- 1. Run Time-Based Cross-Validation ---
    combined_x = pd.concat([train_x, val_x], ignore_index=True)
    combined_y = pd.concat([train_y, val_y], ignore_index=True)

    mlflow.log_params(
        {
            f"cv/{k}": v
            for k, v in flatten_params({"cross_validation": cv_config}).items()
        }
    )

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

    # --- 2. Log Aggregated CV Results ---
    for scale_type, df in cv_results.items():
        if not df.empty:
            logger.info(f"Logging mean/std of {scale_type} CV metrics...")

            mean_metrics = {
                f"cv/{scale_type}/mean/{k}": v for k, v in df.mean().items()
            }
            std_metrics = {f"cv/{scale_type}/std/{k}": v for k, v in df.std().items()}
            all_cv_metrics = {**mean_metrics, **std_metrics}

            for key, value in all_cv_metrics.items():
                mlflow.log_metric(key, value)

            csv_path = f"cv_results_{scale_type}.csv"
            df.to_csv(csv_path, index=True)
            mlflow.log_artifact(csv_path, "cv_results")

    # --- 3. Retrain Final Model on All Data ---
    logger.info("Retraining final model on the full dataset (train + validation)...")
    final_model = train_model(combined_x, combined_y, model_instance)

    # --- 4. Evaluate and Log Final Model Performance on Training Data---
    _evaluate_and_log_set(
        final_model,
        combined_x,
        combined_y,
        "final_model_on_all_data",
        scaler,
        power_transformer,
        model_config.get("log_predictions", False),
    )

    return final_model


def training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    training_params: Dict[str, Any],
    mlflow_params: Dict[str, Any],
    gold_pipeline_params: Dict[str, Any],
):
    """
    Main orchestration function for the training pipeline.
    """
    logger.info("--- Starting Model Training Pipeline ---")

    # === STAGE 1: LOAD CONFIGURATION ===
    logger.info("Loading configurations from params.yaml...")
    model_config_key = training_params["model_config_to_run"]
    model_config = training_params["models"][model_config_key]
    model_class_name = model_config["model_class"]
    model_name = model_config["name"]
    run_name = model_config["run_name"]

    mlflow.set_experiment(mlflow_params["experiment_name"])

    # === STAGE 2: PREPARE DATA AND PREPROCESSORS ===
    logger.info("Preparing data splits and loading preprocessors...")
    train_x = train_df.drop(columns=config_training.TARGET_COLUMN)
    train_y = train_df[config_training.TARGET_COLUMN]
    val_x = val_df.drop(columns=config_training.TARGET_COLUMN)
    val_y = val_df[config_training.TARGET_COLUMN]

    cols_to_drop = model_config.get("drop_multicollinear_cols")
    if cols_to_drop:
        train_x = train_x.drop(columns=cols_to_drop, errors="ignore")
        val_x = val_x.drop(columns=cols_to_drop, errors="ignore")

    try:
        scaler = Scaler.load(config_gold.SCALER_PATH)
        power_transformer = PowerTransformer.load(config_gold.POWER_TRANSFORMER_PATH)
    except FileNotFoundError:
        scaler, power_transformer = None, None
        logger.warning(
            "Scaler or PowerTransformer not found. Unscaled metrics will be unavailable."
        )

    # === STAGE 3: EXECUTE MLFLOW RUN ===
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow Run: '{run_name}' (ID: {run_id})")

        # --- Autologging and Manual Parameter Logging ---
        model_library = get_model_library(model_class_name)
        if model_library and model_library in config_training.LOG_MODEL_MAPPING:
            try:
                auto_log_func = config_training.AUTOLOG_MAPPING[model_library]
                model_info = auto_log_func(
                    log_models=model_config.get("log_model_artifact", False),
                    log_input_examples=True,
                    silent=True,
                    disable=False,  # Ensure autologging is enabled
                )
            except Exception as e:
                logger.error(f"Autologging failed: {e}", exc_info=True)

        mlflow.log_params(flatten_params({"gold_pipeline": gold_pipeline_params}))
        mlflow.log_params(flatten_params({model_config_key: model_config}))
        mlflow.set_tags({"model_name": model_name, "model_class": model_class_name})

        # --- Model Instantiation and Training ---
        model_instance = get_model(model_class_name, model_config["training_params"])

        if model_config.get("cross_validation", {}).get("enabled", False):
            mlflow.set_tag("run_type", "cross_validation")
            final_model = _run_cross_validation(
                model_instance,
                train_x,
                train_y,
                val_x,
                val_y,
                model_config["cross_validation"],
                scaler,
                power_transformer,
                model_config,
            )
        else:
            mlflow.set_tag("run_type", "simple_validation")
            final_model = _run_simple_validation(
                model_instance,
                train_x,
                train_y,
                val_x,
                val_y,
                scaler,
                power_transformer,
                model_config,
            )

        # === STAGE 4: FINAL EVALUATION ON TEST SET ===
        if model_config.get("evaluate_on_test_set", False) and test_df is not None:
            logger.info("--- Performing final evaluation on the hold-out test set ---")
            test_x = test_df.drop(columns=config_training.TARGET_COLUMN)
            test_y = test_df[config_training.TARGET_COLUMN]

            if cols_to_drop:
                test_x = test_x.drop(columns=cols_to_drop, errors="ignore")

            _evaluate_and_log_set(
                model=final_model,
                X=test_x,
                y_true=test_y,
                log_prefix="test",
                scaler=scaler,
                power_transformer=power_transformer,
                log_predictions=True,
            )
        elif test_df is None:
            logger.warning("No test dataset provided. Skipping final evaluation.")
        else:
            logger.info(
                "'evaluate_on_test_set' is false in config. Skipping final test set evaluation to prevent bias."
            )

        # === STAGE 5: MODEL REGISTRATION (OPTIONAL) ===
        if model_config.get("register_model", False):
            logger.info(f"Registering model '{model_name}' to MLflow Model Registry...")
            model_library = get_model_library(model_class_name)
            if model_library and model_library in config_training.LOG_MODEL_MAPPING:
                try:
                    log_model_func = config_training.LOG_MODEL_MAPPING[model_library]
                    model_arg_name = config_training.LOG_MODEL_ARG_NAME[model_library]
                    model_info = log_model_func(
                        **{model_arg_name: final_model},
                        artifact_path="model",
                        registered_model_name=model_name,
                    )
                    logger.info(
                        f"Model registered successfully. URI: {model_info.model_uri}"
                    )
                except Exception as e:
                    logger.error(f"Model registration failed: {e}", exc_info=True)
            else:
                logger.warning(f"No registration mapping for '{model_class_name}'")

    logger.info(f"--- Model Training Pipeline COMPLETED for run '{run_name}' ---")


def main():
    """
    Main entry point for the script.
    """
    # === SETUP ===
    load_dotenv()
    setup_logging_from_yaml(
        log_path=config_logging.TRAINING_PIPELINE_LOGS_PATH,
        default_yaml_path=config_logging.LOGGING_YAML,
    )
    logger.info(">>> ORCHESTRATOR: Starting Training Pipeline Execution <<<")

    parser = argparse.ArgumentParser(description="Run the Model Training Pipeline.")
    parser.add_argument("train_file_name", help="Name of the gold training data file.")
    parser.add_argument(
        "validation_file_name", help="Name of the gold validation data file."
    )
    parser.add_argument("test_file_name", help="Name of the gold test data file.")
    args = parser.parse_args()

    # === EXECUTION ===
    try:
        # --- Load Data and Parameters ---
        logger.info("Loading data and params.yaml...")
        train_df = load_data(config_gold.GOLD_PROCESSED_DIR / args.train_file_name)
        validation_df = load_data(
            config_gold.GOLD_PROCESSED_DIR / args.validation_file_name
        )
        test_df = load_data(config_gold.GOLD_PROCESSED_DIR / args.test_file_name)

        if train_df is None or validation_df is None or test_df is None:
            logger.critical("Failed to load one or more data files. Aborting.")
            sys.exit(1)

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        # --- Run Pipeline ---
        training_pipeline(
            train_df=train_df,
            val_df=validation_df,
            test_df=test_df,
            training_params=params["training_pipeline"],
            mlflow_params=params["mlflow_params"],
            gold_pipeline_params=params["gold_pipeline"],
        )

    except FileNotFoundError as e:
        logger.critical(f"A required file was not found: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred in the pipeline: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info(">>> ORCHESTRATOR: Pipeline Execution Finished Successfully <<<")


if __name__ == "__main__":
    main()
