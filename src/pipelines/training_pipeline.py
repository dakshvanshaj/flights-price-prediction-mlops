import argparse
import logging
import pickle
import sys
from typing import Any, Dict, Optional, List

import mlflow
import pandas as pd
import shap
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
from shared.config import core_paths, config_gold, config_logging, config_training
from shared.utils import flatten_params, setup_logging_from_yaml
from model_evaluation.evaluation_plots import (
    scatter_plot,
    residual_plot,
    qq_plot_residuals,
    plot_feature_importance,
)
from model_explainability.shap_explain import shap_plots

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


def _create_and_log_shap_explainer(
    model: Any, X_train: pd.DataFrame, artifact_path: str = "shap_explainer"
):
    """
    Creates a SHAP explainer, saves it as a pickle file, and logs it to MLflow.

    Args:
        model: The trained model for which to create the explainer.
        X_train: The background dataset to use for the explainer.
        artifact_path: The directory path within MLflow artifacts to save the explainer.

    """
    logger.info(
        f"Creating and logging SHAP explainer for model: {type(model).__name__}..."
    )
    try:
        # Create the SHAP explainer
        explainer = shap.Explainer(model, X_train)

        # Save the explainer to models folder as pickle file(tracked by both dvc and git)
        explainer_loc = core_paths.MODELS_DIR / "shap_explainer.pkl"
        with open(explainer_loc, "wb") as f:
            pickle.dump(explainer, f)

        # Log the file as an artifact to he current MLflow run
        mlflow.log_artifact(explainer_loc, artifact_path=artifact_path)
        logger.info(
            f"SHAP explainer successfully logged to MLflow at artifact path: '{artifact_path}'."
        )

    except Exception as e:
        logger.info(
            f"An unexpected error occurred while creating or logging the SHAP explainer: {e}",
            exc_info=True,
        )


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
    model_config: Dict[str, Any],
    log_predictions: bool = False,
    log_interpretability_artifacts: bool = True,
    log_plots: bool = True,
    log_shap_plots: bool = True,
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
        model_config: The configuration dictionary for the current model.
        log_predictions: If True, logs predictions as a JSON artifact.
        log_interpretability_artifacts: If True, logs coefficients or feature importances.
        log_plots: If True, generates and logs evaluation plots.
    """
    logger.info(f"--- Evaluating model on '{log_prefix}' data ---")
    y_pred = model.predict(X)

    # --- Metric Calculation & Logging ---
    is_scaled_run = scaler and power_transformer
    y_true_unscaled, y_pred_unscaled = None, None

    if is_scaled_run:
        # Log metrics on the data as seen by the model (scaled)
        _calculate_and_log_metrics(y_true, y_pred, log_prefix=f"{log_prefix}/scaled")

        # Unscale for interpretable metrics and plots
        y_true_unscaled, y_pred_unscaled = unscale_predictions(
            y_true, y_pred, scaler, power_transformer
        )
        _calculate_and_log_metrics(
            y_true_unscaled, y_pred_unscaled, log_prefix=f"{log_prefix}/unscaled"
        )
    else:
        # For tree models, data is already on the original scale. Log directly.
        _calculate_and_log_metrics(y_true, y_pred, log_prefix=log_prefix)
        # Assign original data to the 'unscaled' variables for consistent plotting
        y_true_unscaled, y_pred_unscaled = y_true, y_pred

    # --- Artifact Logging ---

    # Log predictions as a JSON artifact if enabled
    if log_predictions:
        logger.info(f"Logging prediction artifacts for '{log_prefix}'...")
        # Always log the model's direct output
        model_output_dict = {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}
        mlflow.log_dict(
            model_output_dict, f"predictions/{log_prefix}_model_output.json"
        )
        # If unscaled values are different, log them too
        if is_scaled_run:
            unscaled_preds_dict = {
                "y_true": y_true_unscaled.tolist(),
                "y_pred": y_pred_unscaled.tolist(),
            }
            mlflow.log_dict(
                unscaled_preds_dict,
                f"predictions/{log_prefix}_unscaled_predictions.json",
            )

    # Log interpretability artifacts if enabled
    if log_interpretability_artifacts:
        if hasattr(model, "coef_"):
            logger.info(f"Logging model coefficients for '{log_prefix}'...")
            coefficients_dict = dict(zip(X.columns, model.coef_))
            mlflow.log_dict(
                coefficients_dict, f"coefficients/{log_prefix}_coefficients.json"
            )
        if hasattr(model, "feature_importances_"):
            logger.info(f"Logging feature importances for '{log_prefix}'...")
            feature_importances_dict = dict(zip(X.columns, model.feature_importances_))
            mlflow.log_dict(
                feature_importances_dict,
                f"feature_importances/{log_prefix}_feature_importances.json",
            )
            # Also plot the feature importances if plots are enabled
            if log_plots:
                plot_feature_importance(
                    list(X.columns), model, title=f"[{log_prefix}] Feature Importance"
                )

    # Generate and log plots if enabled (now uses the consistently populated unscaled vars)
    if log_plots and y_true_unscaled is not None and y_pred_unscaled is not None:
        logger.info(f"Generating and logging plots for '{log_prefix}'...")
        scatter_plot(
            x=y_true_unscaled,
            y=y_pred_unscaled,
            x_label="Actual Values",
            y_label="Predicted Values",
            title=f"[{log_prefix}] Actual vs. Predicted Values",
        )
        residual_plot(
            y_true=y_true_unscaled,
            y_pred=y_pred_unscaled,
            xlabel="Predicted Values",
            ylabel="Residuals",
            title=f"[{log_prefix}] Residuals vs. Predicted Values",
        )
        qq_plot_residuals(
            y_true=y_true_unscaled,
            y_pred=y_pred_unscaled,
            title=f"[{log_prefix}] Q-Q Plot of Residuals",
        )
        logger.info(f"Successfully logged all plots for '{log_prefix}'.")

    if log_shap_plots:
        n_local_plots = model_config.get("n_shap_local_plots", 3)
        shap_plots(
            model=model,
            X=X,
            log_prefix=log_prefix,
            n_local_plots=n_local_plots,
        )


def _run_simple_training(
    model_instance: Any,
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
) -> Any:
    """
    Trains a model on combination of training and validation sets.

    Returns:
        The trained model instance.
    """
    logger.info("=" * 25 + " SIMPLE  MODEL TRAINING " + "=" * 25)

    # Just train the model and return the model No evaluation will be done
    model = train_model(
        train_x, train_y, model_instance, categorical_features=categorical_features
    )
    logger.info("Model training completed...")
    return model


def _run_simple_validation(
    model_instance: Any,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    scaler: Optional[Scaler],
    power_transformer: Optional[PowerTransformer],
    model_config: Dict[str, Any],
    categorical_features: Optional[List[str]] = None,
) -> Any:
    """
    Trains a model and evaluates it on training and validation sets.

    Returns:
        The trained model instance.
    """
    logger.info("Running Simple Validation (Train/Validation Split)...")
    model = train_model(
        train_x, train_y, model_instance, categorical_features=categorical_features
    )

    should_log_plots = model_config.get("log_plots", True)
    should_log_preds = model_config.get("log_predictions", False)
    should_log_interpretability = model_config.get(
        "log_interpretability_artifacts", True
    )

    _create_and_log_shap_explainer(model, train_x)

    should_log_shap_plots = model_config.get("log_shap_plots", True)
    _evaluate_and_log_set(
        model,
        train_x,
        train_y,
        "training",
        scaler,
        power_transformer,
        model_config=model_config,
        log_plots=should_log_plots,
        log_predictions=should_log_preds,
        log_interpretability_artifacts=should_log_interpretability,
        log_shap_plots=should_log_shap_plots,
    )

    _evaluate_and_log_set(
        model,
        val_x,
        val_y,
        "validation",
        scaler,
        power_transformer,
        model_config=model_config,
        log_predictions=should_log_preds,
        log_interpretability_artifacts=should_log_interpretability,
        log_plots=should_log_plots,
        log_shap_plots=should_log_shap_plots,
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
    categorical_features: Optional[List[str]] = None,
    is_tree_model: bool = False,
) -> Any:
    """
    Performs time-based cross-validation and retrains a final model on all data.

    Returns:
        The final model instance trained on the combined dataset.
    """
    logger.info("Running Time-Based Cross-Validation...")

    # --- 1. Run Time-Based Cross-Validation ---
    logger.info("Combining train and validation sets for CV...")
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
        is_tree_model=is_tree_model,
    )

    # --- 2. Log Aggregated CV Results ---
    logger.info("Logging aggregated CV results...")
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
    final_model = train_model(
        combined_x,
        combined_y,
        model_instance,
        categorical_features=categorical_features,
    )

    # Save model explainer using shap
    _create_and_log_shap_explainer(final_model, combined_x)

    # --- 4. Evaluate and Log Final Model Performance on Training Data---
    _evaluate_and_log_set(
        final_model,
        combined_x,
        combined_y,
        "final_model_on_all_data",
        scaler,
        power_transformer,
        model_config=model_config,
        log_predictions=model_config.get("log_predictions", False),
        log_interpretability_artifacts=model_config.get(
            "log_interpretability_artifacts", True
        ),
        log_plots=model_config.get("log_plots", True),
        log_shap_plots=model_config.get("log_shap_plots", True),
    )

    return final_model


def training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    training_params: Dict[str, Any],
    mlflow_params: Dict[str, Any],
    gold_pipeline_params: Dict[str, Any],
    is_tree_model: bool = False,
):
    """
    Main orchestration function for the training pipeline.
    """
    logger.info("--- Starting Model Training Pipeline ---")

    # === STAGE 1: CONFIGURATION & SETUP ===
    logger.info("=" * 25 + " STAGE 1/5: CONFIGURATION & SETUP " + "=" * 25)
    logger.info("Loading configurations from params.yaml...")
    model_config_key = training_params["model_config_to_run"]
    model_config = training_params["models"][model_config_key]
    model_class_name = model_config["model_class"]
    model_name = model_config.get("name", model_config_key)  # Use key as fallback name
    run_name = model_config["run_name"]

    logger.info(f"Configuration selected: '{model_config_key}'")

    mlflow.set_experiment(mlflow_params["experiment_name"])
    logger.info(f"MLflow experiment set to: '{mlflow_params['experiment_name']}'")

    # === STAGE 2: DATA PREPARATION ===
    logger.info("=" * 25 + " STAGE 2/5: DATA PREPARATION " + "=" * 25)
    logger.info("Preparing data splits (X, y)...")
    train_x = train_df.drop(columns=config_training.TARGET_COLUMN)
    train_y = train_df[config_training.TARGET_COLUMN]

    val_x, val_y = None, None
    if val_df is not None:
        val_x = val_df.drop(columns=config_training.TARGET_COLUMN)
        val_y = val_df[config_training.TARGET_COLUMN]

    cols_to_drop = model_config.get("drop_multicollinear_cols")
    if cols_to_drop:
        logger.info(f"Dropping specified multicollinear columns: {cols_to_drop}")
        train_x = train_x.drop(columns=cols_to_drop, errors="ignore")
        if val_x is not None:
            val_x = val_x.drop(columns=cols_to_drop, errors="ignore")

    logger.info("Loading Scaler and PowerTransformer objects...")
    try:
        scaler = Scaler.load(config_gold.SCALER_PATH)
        power_transformer = PowerTransformer.load(config_gold.POWER_TRANSFORMER_PATH)
        logger.info("Successfully loaded Scaler and PowerTransformer objects.")
    except FileNotFoundError:
        scaler, power_transformer = None, None
        logger.warning(
            "Scaler or PowerTransformer not found. Unscaled metrics will be unavailable."
        )

    # === STAGE 3: MODEL TRAINING & VALIDATION ===
    logger.info("=" * 25 + " STAGE 3/5: MODEL TRAINING & VALIDATION " + "=" * 25)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow Run: '{run_name}' (ID: {run_id})")

        # --- Autologging and Manual Parameter Logging ---
        logger.info("Setting up MLflow autologging and logging parameters...")
        model_library = get_model_library(model_class_name)
        if model_library and model_library in config_training.LOG_MODEL_MAPPING:
            try:
                auto_log_func = config_training.AUTOLOG_MAPPING[model_library]
                auto_log_func(
                    log_models=False,
                    log_input_examples=True,
                    silent=True,
                    disable=False,
                )
                logger.info(f"Enabled autologging for '{model_library}'.")
            except Exception as e:
                logger.error(f"Autologging failed: {e}", exc_info=True)

        mlflow.log_params(flatten_params({"gold_pipeline": gold_pipeline_params}))
        mlflow.log_params(flatten_params({model_config_key: model_config}))
        mlflow.set_tags({"model_name": model_name, "model_class": model_class_name})
        logger.info("Logged pipeline parameters and model tags.")

        # --- Define Categorical Features for LightGBM ---
        categorical_features = config_gold.ENCODING_CONFIG.get("ordinal_cols", [])
        if categorical_features:
            # Sanitize names to match DataFrame columns (lowercase, underscores)
            categorical_features = [
                c.replace(" ", "_").lower() for c in categorical_features
            ]
            logger.info(
                f"Identified categorical features for LightGBM: {categorical_features}"
            )

        # --- Model Instantiation and Training ---
        logger.info(f"Instantiating model: {model_class_name}")
        model_instance = get_model(model_class_name, model_config["training_params"])

        if model_config.get("train_model_only", False):
            mlflow.set_tag("run_type", "model_training_only")
            final_model = _run_simple_training(
                model_instance,
                train_x,
                train_y,
                categorical_features=categorical_features,
            )

        elif model_config.get("cross_validation", {}).get("enabled", False):
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
                categorical_features=categorical_features,
                is_tree_model=is_tree_model,
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
                categorical_features=categorical_features,
            )

        # === STAGE 4: FINAL EVALUATION ON TEST SET ===
        logger.info("=" * 25 + " STAGE 4/5: FINAL EVALUATION ON TEST SET " + "=" * 25)
        if model_config.get("evaluate_on_test_set", False) and test_df is not None:
            logger.info("Performing final evaluation on the hold-out test set...")
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
                model_config=model_config,
                log_predictions=model_config.get("log_predictions", False),
                log_plots=model_config.get("log_plots", True),
                log_shap_plots=model_config.get("log_shap_plots", True),
            )
        elif test_df is None:
            logger.warning("No test dataset provided. Skipping final evaluation.")
        else:
            logger.info(
                "'evaluate_on_test_set' is false in config. Skipping final test set evaluation."
            )

        # === STAGE 5: MODEL LOGGING & REGISTRATION ===
        logger.info("=" * 25 + " STAGE 5/5: MODEL LOGGING & REGISTRATION " + "=" * 25)
        if model_config.get("log_model_artifact", False):
            logger.info("Logging final model artifact to MLflow...")
            model_library = get_model_library(model_class_name)

            registered_model_name = None
            if model_config.get("register_model", False):
                registered_model_name = model_name
                logger.info(
                    f"Model will also be registered as '{registered_model_name}'."
                )

            if model_library and model_library in config_training.LOG_MODEL_MAPPING:
                try:
                    log_model_func = config_training.LOG_MODEL_MAPPING[model_library]
                    model_arg_name = config_training.LOG_MODEL_ARG_NAME[model_library]

                    model_info = log_model_func(
                        **{model_arg_name: final_model},
                        name=model_name,
                        registered_model_name=registered_model_name,
                        input_example=train_x.iloc[:5],
                    )
                    logger.info(
                        f"Model logged successfully. URI: {model_info.model_uri}"
                    )
                except Exception as e:
                    logger.error(
                        f"Model logging/registration failed: {e}", exc_info=True
                    )
            else:
                logger.warning(
                    f"No logging/registration mapping for '{model_class_name}'"
                )
        else:
            logger.info(
                "Skipping final model logging as 'log_model_artifact' is false."
            )

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
    parser.add_argument(
        "--test_file_name",
        default=None,
        help="Optional:Name of the gold test data file.",
    )
    args = parser.parse_args()

    # === EXECUTION ===
    try:
        # ---------------------------------------------------------------------------- #
        #                           Read the params.yaml file                          #
        # ---------------------------------------------------------------------------- #
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        model_config_key = params["training_pipeline"]["model_config_to_run"]
        model_config = params["training_pipeline"]["models"][model_config_key]
        is_train_only_run = model_config.get("train_model_only", False)
        # ---------------------------------------------------------------------------- #
        #                                   Load Data                                  #
        # ---------------------------------------------------------------------------- #
        test_df = None
        if is_train_only_run:
            logger.info("Training only mode (no evaluation) is enabled")
            logger.info("Loading Data...")
            train_df_part = load_data(
                config_gold.GOLD_PROCESSED_DIR / args.train_file_name
            )
            val_df_part = load_data(
                config_gold.GOLD_PROCESSED_DIR / args.validation_file_name
            )
            train_df = pd.concat([train_df_part, val_df_part], ignore_index=True)
            validation_df = None
        else:
            logger.info("Loading Data...")
            train_df = load_data(config_gold.GOLD_PROCESSED_DIR / args.train_file_name)
            validation_df = load_data(
                config_gold.GOLD_PROCESSED_DIR / args.validation_file_name
            )

        # ---------------------------------------------------------------------------- #
        #               Load Test Data If Test Set Evaluation Is Enabled               #
        # ---------------------------------------------------------------------------- #
        if model_config.get("evaluate_on_test_set", False):
            logger.info("Loading Test Data...")
            if args.test_file_name:
                logger.info(f"Loading test data from '{args.test_file_name}'...")
                test_df = load_data(
                    config_gold.GOLD_PROCESSED_DIR / args.test_file_name
                )
            else:
                logger.critical(
                    "No test data provided. Cannot perform test set evaluation..."
                )
                sys.exit(1)

        # ---------------------------------------------------------------------------- #
        #                             Run Training Pipeline                            #
        # ---------------------------------------------------------------------------- #
        training_pipeline(
            train_df=train_df,
            val_df=validation_df,
            test_df=test_df,
            training_params=params["training_pipeline"],
            mlflow_params=params["mlflow_params"],
            gold_pipeline_params=params["gold_pipeline"],
            is_tree_model=params.get("is_tree_model", False),
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
