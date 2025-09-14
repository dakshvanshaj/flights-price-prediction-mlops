import logging
import mlflow
from typing import Any, Callable, Dict, Tuple
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
)
from model_evaluation.evaluation import time_based_cross_validation
from gold_data_preprocessing.power_transformer import PowerTransformer
from gold_data_preprocessing.scaler import Scaler

# Initialize logger for this module
logger = logging.getLogger(__name__)


def _log_sklearn_cv_results(cv_results: Dict[str, Any]):
    """Logs results from a scikit-learn CV search as nested MLflow runs."""
    logger.info("Logging CV results to MLflow as nested runs...")

    # Number of candidate parameter settings
    num_candidates = len(cv_results["params"])

    for i in range(num_candidates):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True) as child_run:  # noqa
            # Log parameters for this trial
            params = cv_results["params"][i]
            mlflow.log_params(params)

            # Log metrics for this trial
            metrics = {}
            for key, value in cv_results.items():
                if key.startswith(("mean_", "std_", "rank_")):
                    metrics[key] = value[i]
                # Also log individual split scores
                elif key.startswith("split") and "_test_score" in key:
                    metrics[key] = value[i]

            mlflow.log_metrics(metrics)
            mlflow.set_tag("trial_number", i)


def grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 1,
    log_model_artifact: bool = False,
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
        log_model_artifact: If True, refit the best model and return it.

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
        refit=log_model_artifact,  # Conditionally refit
    )
    grid_search_cv.fit(X_train, y_train)

    # Log all trials as nested runs
    _log_sklearn_cv_results(grid_search_cv.cv_results_)

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
    log_model_artifact: bool = False,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Randomized Search CV and logs each trial as a nested MLflow run.

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
        log_model_artifact: If True, refit the best model and return it.

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
        refit=log_model_artifact,  # Conditionally refit
    )
    random_search_cv.fit(X_train, y_train)

    # Log all trials as nested runs
    _log_sklearn_cv_results(random_search_cv.cv_results_)

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
    log_model_artifact: bool = False,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Halving Grid Search CV and logs each trial as a nested MLflow run.

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
        log_model_artifact: If True, refit the best model and return it.

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
        refit=log_model_artifact,  # Conditionally refit
    )
    halving_search.fit(X_train, y_train)

    # Log all trials as nested runs
    _log_sklearn_cv_results(halving_search.cv_results_)

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
    log_model_artifact: bool = False,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Performs Halving Randomized Search CV and logs each trial as a nested MLflow run.

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
        log_model_artifact: If True, refit the best model and return it.

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
        refit=log_model_artifact,  # Conditionally refit
    )
    halving_search.fit(X_train, y_train)

    # Log all trials as nested runs
    _log_sklearn_cv_results(halving_search.cv_results_)

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
    scoring: str = "mean_squared_error",
    direction: str = "minimize",
    scaler: Scaler = None,
    power_transformer: PowerTransformer = None,
    log_model_artifact: bool = False,
    is_tree_model: bool = False,
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
        log_model_artifact: If True, refit the best model and return it.

    Returns:
        A tuple containing (best_estimator, best_params, best_score).
        The best_estimator will be None if log_model_artifact is False.
    """
    import os

    def _objective(trial: optuna.trial.Trial) -> float:
        """Internal objective function for Optuna to optimize."""
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            params = param_definer(trial)
            estimator = estimator_class(**params)

            mlflow.log_params(params)
            mlflow.set_tag("trial_number", trial.number)

            try:
                cv_results = time_based_cross_validation(
                    estimator,
                    X_train,
                    y_train,
                    n_splits=cv,
                    power_transformer=power_transformer,
                    scaler=scaler,
                    is_tree_model=is_tree_model,
                )
            except Exception as e:
                logger.warning(
                    f"Trial {trial.number} failed during CV with error: {e}. Pruning trial."
                )
                mlflow.set_tag("status", "FAILED")
                raise optuna.exceptions.TrialPruned()

            if is_tree_model:
                score_df = cv_results.get("score_df")
                if score_df is not None and not score_df.empty:
                    mean_metrics = {
                        f"cv/mean/{k}": v for k, v in score_df.mean().items()
                    }
                    std_metrics = {
                        f"cv/std/{k}": v for k, v in score_df.std().items()
                    }
                    mlflow.log_metrics({**mean_metrics, **std_metrics})
                    csv_path = "cv_results_scores.csv"
                    score_df.to_csv(csv_path, index=True)
                    mlflow.log_artifact(csv_path, "cv_results")
                    os.remove(csv_path)
            else:
                for scale_type, df in cv_results.items():
                    if not df.empty:
                        mean_metrics = {
                            f"cv/{scale_type}/mean/{k}": v
                            for k, v in df.mean().items()
                        }
                        std_metrics = {
                            f"cv/{scale_type}/std/{k}": v for k, v in df.std().items()
                        }
                        mlflow.log_metrics({**mean_metrics, **std_metrics})

                        csv_path = f"cv_results_{scale_type}.csv"
                        df.to_csv(csv_path, index=True)
                        mlflow.log_artifact(csv_path, "cv_results")
                        os.remove(csv_path)

            score_df = cv_results.get("unscaled", cv_results.get("score_df"))
            score_metric = "mean_squared_error"

            if score_df is None or score_df.empty:
                logger.error("No results available to score objective. Pruning trial.")
                raise optuna.exceptions.TrialPruned()

            final_score = score_df[score_metric].mean()
            mlflow.log_metric("final_score", final_score)
            mlflow.set_tag("status", "COMPLETED")
            return final_score

    logger.info(f"Starting Optuna search for {estimator_class.__name__}...")
    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_score = study.best_trial.value

    logger.info(f"Optuna search complete. Best score ({scoring}): {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")

    mlflow.log_params({f"best_param_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_trial_score", best_score)
    mlflow.log_metric("best_trial_number", study.best_trial.number)

    best_estimator = None
    if log_model_artifact:
        logger.info("Refitting the best model on the entire training dataset...")
        best_estimator = estimator_class(**best_params).fit(X_train, y_train)
    else:
        logger.info("Skipping final model refit as log_model_artifact is false.")

    return best_estimator, best_params, best_score
