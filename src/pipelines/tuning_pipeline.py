import argparse
import logging
import sys
from typing import Any, Callable, Dict

import mlflow
import optuna
import pandas as pd
import yaml
from dotenv import load_dotenv

# --- Local Module Imports ---
from data_ingestion.data_loader import load_data
from hyperparameter_tuning.tuning import (
    grid_search,
    halving_grid_search,
    halving_random_search,
    optuna_search,
    random_search,
)
from model_training.train import get_model
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler
from shared.config import config_gold, config_logging, config_training
from shared.utils import flatten_params, setup_logging_from_yaml

# Initialize logger for this module
logger = logging.getLogger(__name__)


def create_optuna_param_definer(
    param_space: Dict[str, Any], fixed_params: Dict[str, Any]
) -> Callable[[optuna.trial.Trial], Dict[str, Any]]:
    """
    Dynamically creates a parameter definer function for Optuna from a dictionary
    defined in the YAML configuration. This allows defining search spaces directly
    in tuning.yaml instead of hardcoding them in Python functions.

    Args:
        param_space: A dictionary defining the hyperparameter search space.
                     The keys are parameter names, and values are dicts specifying
                     the type ('int', 'float', 'categorical'), range, and other
                     details for Optuna's trial.suggest_* methods.
        fixed_params: A dictionary of parameters with fixed values that should
                      be included with every trial (e.g., 'random_state', 'n_jobs').

    Returns:
        A function that can be passed to an Optuna study, which takes a 'trial'
        object and returns a dictionary of hyperparameters for that trial.
    """

    def param_definer(trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Generates a set of hyperparameters for a single Optuna trial."""
        params = {}
        for name, space in param_space.items():
            suggest_type = space.get("type")
            if suggest_type == "int":
                params[name] = trial.suggest_int(
                    name, space["low"], space["high"], step=space.get("step", 1)
                )
            elif suggest_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    space["low"],
                    space["high"],
                    step=space.get("step"),
                    log=space.get("log", False),
                )
            elif suggest_type == "categorical":
                params[name] = trial.suggest_categorical(name, space["choices"])
            else:
                raise ValueError(
                    f"Unsupported Optuna suggestion type: '{suggest_type}' for param '{name}'"
                )

        params.update(fixed_params)
        return params

    return param_definer


def tuning_pipeline(
    train_df: pd.DataFrame,
    tuning_config: Dict[str, Any],
    mlflow_params: Dict[str, Any],
) -> None:
    """
    Main orchestration function for the hyperparameter tuning pipeline.

    Args:
        train_df: The DataFrame containing training data.
        tuning_config: Dictionary of the tuning configuration from tuning.yaml.
        mlflow_params: Dictionary of parameters for MLflow from tuning.yaml.
    """
    logger.info("--- Starting Hyperparameter Tuning Pipeline ---")

    # === STAGE 1: LOAD CONFIGURATION ===
    logger.info("Loading configurations from tuning.yaml...")
    run_name = tuning_config["run_name"]
    model_class_name = tuning_config["model_class"]
    tuner_type = tuning_config["tuner_type"]
    tuner_kwargs = tuning_config.get("tuner_params", {})

    mlflow.set_experiment(mlflow_params["experiment_name"])

    # === STAGE 2: PREPARE DATA ===
    logger.info("Preparing data splits...")
    X_train = train_df.drop(columns=config_training.TARGET_COLUMN)
    y_train = train_df[config_training.TARGET_COLUMN]
    logger.info(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")

    # === STAGE 3: EXECUTE MLFLOW RUN ===
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow Run: '{run_name}' (ID: {run_id})")

        # --- Log Initial Configuration ---
        logger.info("Logging parameters from tuning.yaml to MLflow...")
        mlflow.log_params(flatten_params({"tuning_config": tuning_config}))
        mlflow.set_tags(
            {
                "model_class": model_class_name,
                "tuner_type": tuner_type,
                "run_type": "hyperparameter_tuning",
            }
        )

        # --- Select and Execute Tuner ---
        tuner_map = {
            "grid": grid_search,
            "random": random_search,
            "halving_grid": halving_grid_search,
            "halving_random": halving_random_search,
            "optuna": optuna_search,
        }

        if tuner_type not in tuner_map:
            raise ValueError(f"Unsupported tuner_type: '{tuner_type}'")

        tuner_func = tuner_map[tuner_type]

        # --- Prepare arguments for the selected tuner ---
        if tuner_type == "optuna":
            estimator_class = get_model(model_class_name, model_params={}).__class__

            param_space = tuning_config.get("param_space")
            if not param_space:
                raise ValueError(
                    "Optuna tuner requires 'param_space' in tuning_config."
                )

            base_params = tuning_config.get("base_params", {})
            fixed_params = {
                k: v for k, v in base_params.items() if k in ["random_state", "n_jobs"]
            }

            param_definer_func = create_optuna_param_definer(param_space, fixed_params)

            try:
                scaler = Scaler.load(config_gold.SCALER_PATH)
                power_transformer = PowerTransformer.load(
                    config_gold.POWER_TRANSFORMER_PATH
                )
            except FileNotFoundError:
                scaler, power_transformer = None, None
                logger.warning(
                    "Scaler or PowerTransformer not found. Unscaled metrics will be unavailable."
                )

            best_estimator, best_params, best_score = tuner_func(
                estimator_class=estimator_class,
                param_definer=param_definer_func,
                X_train=X_train,
                y_train=y_train,
                scaler=scaler,
                power_transformer=power_transformer,
                **tuner_kwargs,
            )
        else:
            # For scikit-learn tuners
            estimator = get_model(model_class_name, model_params={})
            param_grid = tuning_config.get("param_grid")
            if not param_grid:
                raise ValueError(
                    f"{tuner_type} tuner requires 'param_grid' in tuning_config."
                )

            tuner_kwargs["param_grid"] = param_grid

            best_estimator, best_params, best_score = tuner_func(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                **tuner_kwargs,
            )

        # === STAGE 4: LOG RESULTS TO MLFLOW ===
        logger.info("Logging best results to MLflow...")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        scoring_metric = tuner_kwargs.get("scoring", "neg_mean_squared_error")
        mlflow.log_metric(f"best_cv_{scoring_metric}", best_score)

        # --- Conditionally log the model artifact ---
        if tuning_config.get("log_model_artifact", False):
            should_register = tuning_config.get("register_model", False)
            registered_model_name = (
                f"{run_name}-best-model" if should_register else None
            )

            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                artifact_path="best-model",
                registered_model_name=registered_model_name,
            )
            log_msg = "Logged best model artifact"
            if registered_model_name:
                log_msg += f" and registered it as '{registered_model_name}'"
            log_msg += " in MLflow."
            logger.info(log_msg)
        else:
            logger.info("Skipping model artifact logging as per configuration.")

    logger.info(
        f"--- Hyperparameter Tuning Pipeline COMPLETED for run '{run_name}' ---"
    )


def main():
    """Main entry point for the script."""
    # === SETUP ===
    load_dotenv()  # Load environment variables from .env file
    setup_logging_from_yaml(
        log_path=config_logging.TUNING_PIPELINE_LOGS_PATH,
        default_yaml_path=config_logging.LOGGING_YAML,
    )
    logger.info(">>> ORCHESTRATOR: Starting Tuning Pipeline Execution <<<")

    parser = argparse.ArgumentParser(
        description="Run the Hyperparameter Tuning Pipeline."
    )
    parser.add_argument("train_file_name", help="Name of the gold training data file.")
    args = parser.parse_args()

    # === EXECUTION ===
    try:
        # --- Load Data and Parameters ---
        logger.info("Loading data and tuning.yaml configuration file...")
        train_df = load_data(config_gold.GOLD_PROCESSED_DIR / args.train_file_name)

        if train_df is None:
            logger.critical("Failed to load training data file. Aborting.")
            sys.exit(1)

        with open("tuning.yaml", "r") as f:
            all_configs = yaml.safe_load(f)

        # --- Get Model and Tuning Config ---
        mlflow_params = all_configs["mlflow_params"]
        model_key_to_tune = all_configs["model_to_tune"]
        tuning_config = all_configs["tuning_configs"][model_key_to_tune]
        logger.info(f"Loaded tuning configuration for model: '{model_key_to_tune}'")

        if not tuning_config.get("enabled", False):
            logger.warning(
                f"Tuning configuration for '{model_key_to_tune}' is disabled in tuning.yaml. "
                "Skipping pipeline."
            )
            sys.exit(0)

        # --- Run Pipeline ---
        tuning_pipeline(
            train_df=train_df,
            tuning_config=tuning_config,
            mlflow_params=mlflow_params,
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
