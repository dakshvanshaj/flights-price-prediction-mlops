"""
Tests for the hyperparameter tuning functions in `src.hyperparameter_tuning.tuning`.

These tests are designed to verify the logic of our wrapper functions, not the
underlying optimization libraries (like Scikit-learn or Optuna). The primary
goal is to ensure that our functions correctly initialize, call, and process the
results from these libraries.

We use mocking extensively to:
- Isolate our code from external dependencies.
- Prevent slow, actual model training or hyperparameter searches.
- Create predictable return values to test our logic.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import optuna

# Functions to be tested
from hyperparameter_tuning import tuning
from sklearn.linear_model import LinearRegression, Ridge

# --- Fixtures ---


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a simple, small dataset for testing function calls."""
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    return X, y


# --- Tests for Scikit-learn CV Wrappers ---


@patch("hyperparameter_tuning.tuning.mlflow")
def test_log_sklearn_cv_results(mock_mlflow):
    """
    Tests the private helper `_log_sklearn_cv_results`.

    Asserts that the function correctly iterates through fake CV results and calls
    the relevant MLflow logging functions.
    """
    # ARRANGE: Create a fake cv_results dictionary
    cv_results = {
        "params": [{"C": 1.0}, {"C": 0.1}],
        "mean_test_score": [0.9, 0.8],
        "std_test_score": [0.05, 0.08],
        "rank_test_score": [1, 2],
    }

    # ACT
    tuning._log_sklearn_cv_results(cv_results)

    # ASSERT
    # Check that a nested run was started for each set of params
    assert mock_mlflow.start_run.call_count == 2

    # Check that parameters and metrics were logged
    assert mock_mlflow.log_params.call_count == 2
    assert mock_mlflow.log_metrics.call_count == 2

    # Check one of the calls to ensure the correct data is being logged
    mock_mlflow.log_params.assert_called_with({"C": 0.1})


@patch("hyperparameter_tuning.tuning.GridSearchCV")
@patch("hyperparameter_tuning.tuning._log_sklearn_cv_results")
def test_grid_search_orchestration(mock_log_results, mock_grid_search_cv, sample_data):
    """
    Tests the `grid_search` function's orchestration logic.

    This test mocks the `GridSearchCV` object itself to confirm that our wrapper
    function correctly initializes it, calls its `fit` method, and passes the
    results to the logging helper.
    """
    # ARRANGE
    X, y = sample_data
    estimator = LinearRegression()
    param_grid = {"C": [1.0, 0.1]}

    # Configure the mock GridSearchCV instance
    mock_cv_instance = MagicMock()
    mock_cv_instance.best_estimator_ = LinearRegression()  # A dummy object
    mock_cv_instance.best_params_ = {"C": 1.0}
    mock_cv_instance.best_score_ = 0.95
    mock_cv_instance.cv_results_ = {"params": [], "mean_test_score": []}
    mock_grid_search_cv.return_value = mock_cv_instance

    # ACT
    best_estimator, best_params, best_score = tuning.grid_search(
        estimator, param_grid, X, y
    )

    # ASSERT
    # 1. Was GridSearchCV initialized correctly?
    mock_grid_search_cv.assert_called_once()
    # 2. Was the `fit` method called on the instance?
    mock_cv_instance.fit.assert_called_once_with(X, y)
    # 3. Was the logging helper called with the results?
    mock_log_results.assert_called_once_with(mock_cv_instance.cv_results_)
    # 4. Does the function return the correct values from the CV object?
    assert best_score == 0.95
    assert best_params == {"C": 1.0}


# --- Tests for Optuna Wrapper ---


@patch("hyperparameter_tuning.tuning.optuna.create_study")
@patch("hyperparameter_tuning.tuning.mlflow")
@patch("hyperparameter_tuning.tuning.time_based_cross_validation")
def test_optuna_search_success_case(
    mock_time_based_cv, mock_mlflow, mock_create_study, sample_data
):
    """
    Tests the `optuna_search` function by mocking the study and CV process.

    This test ensures that when the cross-validation is successful, the objective
    function correctly processes the results and logs them.
    """
    # ARRANGE
    X, y = sample_data
    mock_time_based_cv.return_value = {
        "unscaled": pd.DataFrame({"mean_squared_error": [0.5, 0.6]})
    }

    # Mock the Optuna study and trial objects
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_study.best_trial.params = {"alpha": 0.5}
    mock_study.best_trial.value = 0.55
    mock_study.best_trial.number = 0
    mock_create_study.return_value = mock_study

    # Use side_effect to run the objective function just once with our mock trial
    def run_objective_once(objective, n_trials, show_progress_bar):
        objective(mock_trial)

    mock_study.optimize.side_effect = run_objective_once

    def param_definer(trial):
        return {"alpha": 0.5}

    # ACT
    tuning.optuna_search(
        estimator_class=Ridge,  # Use Ridge since it accepts 'alpha'
        param_definer=param_definer,
        X_train=X,
        y_train=y,
        n_trials=1,  # We only run one trial in the test
    )

    # ASSERT
    mock_time_based_cv.assert_called_once()
    mock_mlflow.log_metric.assert_any_call("final_score", 0.55)
    mock_mlflow.log_params.assert_any_call({"best_param_alpha": 0.5})


@patch("hyperparameter_tuning.tuning.optuna.create_study")
@patch("hyperparameter_tuning.tuning.time_based_cross_validation")
def test_optuna_search_failure_case(mock_time_based_cv, mock_create_study, sample_data):
    """
    Tests that `optuna_search` correctly handles a CV failure by pruning the trial.
    """
    # ARRANGE
    X, y = sample_data
    mock_time_based_cv.side_effect = ValueError("CV Failed")

    mock_study = MagicMock()
    mock_trial = MagicMock()
    # FIX: Ensure best_trial.value is a real number to prevent logging format error
    mock_study.best_trial.params = {}
    mock_study.best_trial.value = 0.0
    mock_study.best_trial.number = 0
    mock_create_study.return_value = mock_study

    # This is a simplified way to test that TrialPruned is raised inside the objective
    def run_objective_once(objective, n_trials, show_progress_bar):
        with pytest.raises(optuna.exceptions.TrialPruned):
            objective(mock_trial)

    mock_study.optimize.side_effect = run_objective_once

    def param_definer(trial):
        return {}

    # ACT & ASSERT
    # The test passes if the TrialPruned exception is caught by the side_effect function
    tuning.optuna_search(
        estimator_class=LinearRegression,
        param_definer=param_definer,
        X_train=X,
        y_train=y,
        n_trials=1,
    )
