import logging
import math
import mlflow
from typing import Any, Callable, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from model_evaluation.evaluation import time_based_cross_validation
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler
from model_evaluation.evaluation import (
    calculate_all_regression_metrics,
    unscale_predictions,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


def grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Grid Search CV to find the best hyperparameters.

    Args:
        estimator: The model instance for which to tune hyperparameters.
        param_grid: Dictionary with parameter names as keys and lists of settings to try.
        X_train: Training feature data.
        y_train: Training target data.
        cv: Number of cross-validation folds.
        scoring: Scoring metric to evaluate predictions.
        n_jobs: Number of jobs to run in parallel (-1 means using all processors).
        verbose: Controls the verbosity of the output.

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
    """
    logger.info(f"Starting GridSearchCV for {estimator.__class__.__name__}...")
    logger.info(f"Parameter grid: {param_grid}")

    time_series_cv = TimeSeriesSplit(n_splits=cv)

    grid_search_cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=time_series_cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid_search_cv.fit(X_train, y_train)

    best_estimator = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_
    best_score = grid_search_cv.best_score_

    logger.info(f"GridSearchCV complete. Best score ({scoring}): {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")

    return best_estimator, best_params, best_score


def random_search(
    estimator: BaseEstimator,
    param_distributions: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 100,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Randomized Search CV to find the best hyperparameters.

    Args:
        estimator: The model instance for which to tune hyperparameters.
        param_distributions: Dict with parameter names as keys and distributions or lists to sample from.
        X_train: Training feature data.
        y_train: Training target data.
        n_iter: Number of parameter settings that are sampled.
        cv: Number of cross-validation folds.
        scoring: Scoring metric to evaluate predictions.
        n_jobs: Number of jobs to run in parallel.
        verbose: Controls the verbosity of the output.
        random_state: Seed for the random number generator.

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
    """
    logger.info(f"Starting RandomizedSearchCV for {estimator.__class__.__name__}...")
    logger.info(f"Parameter distributions: {param_distributions}")

    time_series_cv = TimeSeriesSplit(n_splits=cv)

    random_search_cv = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=time_series_cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
    )
    random_search_cv.fit(X_train, y_train)

    best_estimator = random_search_cv.best_estimator_
    best_params = random_search_cv.best_params_
    best_score = random_search_cv.best_score_

    logger.info(
        f"RandomizedSearchCV complete. Best score ({scoring}): {best_score:.4f}"
    )
    logger.info(f"Best parameters found: {best_params}")

    return best_estimator, best_params, best_score


def halving_grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
    factor: int = 3,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Halving Grid Search CV, a resource-efficient tuning method.

    Args:
        estimator: The model instance for which to tune hyperparameters.
        param_grid: Dictionary with parameter names as keys and lists of settings to try.
        X_train: Training feature data.
        y_train: Training target data.
        cv: Number of cross-validation folds.
        scoring: Scoring metric to evaluate predictions.
        n_jobs: Number of jobs to run in parallel.
        verbose: Controls the verbosity of the output.
        factor: The 'halving' parameter that controls resource reduction.

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
    """
    logger.info(f"Starting HalvingGridSearchCV for {estimator.__class__.__name__}...")
    logger.info(f"Parameter grid: {param_grid}")

    time_series_cv = TimeSeriesSplit(n_splits=cv)

    halving_search = HalvingGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=time_series_cv,
        n_jobs=n_jobs,
        verbose=verbose,
        factor=factor,
    )
    halving_search.fit(X_train, y_train)

    best_estimator = halving_search.best_estimator_
    best_params = halving_search.best_params_
    best_score = halving_search.best_score_

    logger.info(
        f"HalvingGridSearchCV complete. Best score ({scoring}): {best_score:.4f}"
    )
    logger.info(f"Best parameters found: {best_params}")

    return best_estimator, best_params, best_score


def halving_random_search(
    estimator: BaseEstimator,
    param_distributions: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42,
    factor: int = 3,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Halving Randomized Search CV.

    Args:
        estimator: The model instance for which to tune hyperparameters.
        param_distributions: Dictionary with parameter names as keys and distributions to sample from.
        X_train: Training feature data.
        y_train: Training target data.
        cv: Number of cross-validation folds.
        scoring: Scoring metric to evaluate predictions.
        n_jobs: Number of jobs to run in parallel.
        verbose: Controls the verbosity of the output.
        random_state: Seed for the random number generator.
        factor: The 'halving' parameter that controls resource reduction.

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
    """
    logger.info(f"Starting HalvingRandomSearchCV for {estimator.__class__.__name__}...")
    logger.info(f"Parameter distributions: {param_distributions}")

    time_series_cv = TimeSeriesSplit(n_splits=cv)

    halving_search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=time_series_cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        factor=factor,
    )
    halving_search.fit(X_train, y_train)

    best_estimator = halving_search.best_estimator_
    best_params = halving_search.best_params_
    best_score = halving_search.best_score_

    logger.info(
        f"HalvingRandomSearchCV complete. Best score ({scoring}): {best_score:.4f}"
    )
    logger.info(f"Best parameters found: {best_params}")

    return best_estimator, best_params, best_score


def optuna_search(
    estimator_class: Callable[..., BaseEstimator],
    param_definer: Callable[[optuna.trial.Trial], Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    direction: str = "maximize",
    scaler: Scaler = None,
    power_transformer: PowerTransformer = None,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs hyperparameter optimization using Optuna.

    Args:
        estimator_class: The class of the model to be instantiated (e.g., RandomForestRegressor).
        param_definer: A function that takes an optuna `trial` object and returns a dictionary of parameters.
        X_train: Training feature data.
        y_train: Training target data.
        n_trials: Number of optimization trials to run.
        cv: Number of cross-validation folds.
        scoring: Scoring metric to evaluate predictions.
        direction: Direction of optimization ('minimize' or 'maximize').
        scaler: Fitted scaler for unscaling predictions.
        power_transformer: Fitted transformer for unscaling predictions.

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
    """
    import os

    def _objective(trial: optuna.trial.Trial) -> float:
        """Internal objective function for Optuna to optimize."""
        params = param_definer(trial)
        estimator = estimator_class(**params)

        # Log parameters for each trial for better traceability
        mlflow.log_params(
            {f"trial_{trial.number}_param_{k}": v for k, v in params.items()}
        )

        try:
            cv_results = time_based_cross_validation(
                estimator,
                X_train,
                y_train,
                n_splits=cv,
                power_transformer=power_transformer,
                scaler=scaler,
            )
        except Exception as e:
            logger.warning(
                f"Trial {trial.number} failed during CV with error: {e}. Pruning trial."
            )
            raise optuna.exceptions.TrialPruned()

        # Log metrics and artifacts for each trial
        for scale_type, df in cv_results.items():
            if not df.empty:
                mean_metrics = {
                    f"trial/{trial.number}/cv/{scale_type}/mean/{k}": v
                    for k, v in df.mean().items()
                }
                std_metrics = {
                    f"trial/{trial.number}/cv/{scale_type}/std/{k}": v
                    for k, v in df.std().items()
                }
                mlflow.log_metrics({**mean_metrics, **std_metrics})

                # FIX: Log artifact to a unique path per trial to avoid overwriting
                csv_path = f"cv_results_{scale_type}_trial_{trial.number}.csv"
                df.to_csv(csv_path, index=True)
                mlflow.log_artifact(csv_path, f"cv_results/trial_{trial.number}")
                os.remove(csv_path)  # Clean up the temporary file

        # Determine the score for Optuna to optimize
        score_df = cv_results.get("unscaled")
        score_metric = (
            "mean_squared_error"  # This is hardcoded as in your original implementation
        )

        if score_df is None or score_df.empty:
            logger.info(
                "Unscaled results not found, using scaled results for objective score."
            )
            score_df = cv_results.get("scaled")

        if score_df is None or score_df.empty:
            logger.error("No results available to score objective. Pruning trial.")
            raise optuna.exceptions.TrialPruned()

        # FIX: The objective function must return a single float value (e.g., the mean of the scores)
        final_score = score_df[score_metric].mean()
        mlflow.log_metric(f"trial/{trial.number}/final_score", final_score)
        return final_score

    logger.info(f"Starting Optuna search for {estimator_class.__name__}...")
    study = optuna.create_study(direction=direction)

    study.optimize(
        _objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    best_score = study.best_trial.value

    logger.info(f"Optuna search complete. Best score ({scoring}): {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")

    # Log best trial info for easy access in MLflow
    mlflow.log_params({f"best_param_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_trial_score", best_score)
    mlflow.log_metric("best_trial_number", study.best_trial.number)

    logger.info("Refitting the best model on the entire training dataset...")
    best_estimator = estimator_class(**best_params).fit(X_train, y_train)

    return best_estimator, best_params, best_score
