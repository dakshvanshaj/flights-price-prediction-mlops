import logging
from typing import Any, Callable, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
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

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
    """

    def _objective(trial: optuna.trial.Trial) -> float:
        """Internal objective function for Optuna to optimize."""
        params = param_definer(trial)
        estimator = estimator_class(**params)
        
        time_series_cv = TimeSeriesSplit(n_splits=cv)

        scores = cross_val_score(
            estimator, X_train, y_train, cv=time_series_cv, scoring=scoring, n_jobs=-1
        )
        return np.mean(scores)

    logger.info(f"Starting Optuna search for {estimator_class.__name__}...")
    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_score = study.best_trial.value

    logger.info(f"Optuna search complete. Best score ({scoring}): {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")

    logger.info("Refitting the best model on the entire training dataset...")
    best_estimator = estimator_class(**best_params).fit(X_train, y_train)

    return best_estimator, best_params, best_score