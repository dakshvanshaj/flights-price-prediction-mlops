import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
from typing import Union, Dict, Optional, Any, Tuple
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from gold_data_preprocessing.scaler import Scaler
from gold_data_preprocessing.power_transformer import PowerTransformer
from shared.config import config_training

logger = logging.getLogger(__name__)


def unscale_predictions(
    y_true_scaled: pd.Series,
    y_pred_scaled: np.ndarray,
    scaler: Scaler,
    power_transformer: PowerTransformer,
) -> Tuple[pd.Series, pd.Series]:
    """
    Inverse transforms scaled true values and predictions to their original scale.

    Args:
        y_true_scaled: The scaled ground truth target values.
        y_pred_scaled: The scaled predicted target values as a NumPy array.
        scaler: The fitted Scaler object.
        power_transformer: The fitted PowerTransformer object.

    Returns:
        A tuple containing the unscaled true values and unscaled predicted values.
    """
    y_true_unscaled = power_transformer.inverse_transform(
        scaler.inverse_transform(pd.DataFrame(y_true_scaled))
    )[config_training.TARGET_COLUMN]

    y_pred_unscaled = power_transformer.inverse_transform(
        scaler.inverse_transform(
            pd.DataFrame(
                y_pred_scaled,
                index=y_true_scaled.index,
                columns=[config_training.TARGET_COLUMN],
            )
        )
    )[config_training.TARGET_COLUMN]

    return y_true_unscaled, y_pred_unscaled


def calculate_all_regression_metrics(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Calculates a comprehensive set of regression metrics.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.

    Returns:
        A dictionary of metric names and their calculated values.
    """
    return {
        "r2_score": r2_score(y_true, y_pred),
        "root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }


def time_based_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    max_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    gap: int = 0,
    scaler: Optional[Scaler] = None,
    power_transformer: Optional[PowerTransformer] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Performs time-based cross-validation on pre-sorted data.

    Args:
        model: The model instance to evaluate.
        X: The feature matrix, sorted chronologically.
        y: The target vector, sorted chronologically.
        n_splits: The number of splits for cross-validation.
        max_train_size: Maximum size for a single training set.
        test_size: Size of the test set.
        gap: Number of samples to exclude between train and test sets.
        scaler: Fitted scaler for inverse transforming predictions.
        power_transformer: Fitted transformer for inverse transforming.

    Returns:
        A dictionary containing 'scaled' and 'unscaled' DataFrames with
        evaluation metrics for each fold.
    """
    logger.info(f"--- Starting Time-Based CV with {n_splits} splits ---")
    tscv = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap
    )
    all_scaled_scores, all_unscaled_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"--- Fold {fold}/{n_splits} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"Training on {len(X_train)}, validating on {len(X_val)}.")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        scaled_scores = calculate_all_regression_metrics(y_val, y_pred)
        scaled_scores["fold"] = fold
        all_scaled_scores.append(scaled_scores)

        if scaler and power_transformer:
            y_val_unscaled, y_pred_unscaled = unscale_predictions(
                y_val, y_pred, scaler, power_transformer
            )
            unscaled_scores = calculate_all_regression_metrics(
                y_val_unscaled, y_pred_unscaled
            )
            unscaled_scores["fold"] = fold
            all_unscaled_scores.append(unscaled_scores)

    logger.info("--- Time-Based Cross-Validation Complete ---")
    scaled_df = pd.DataFrame(all_scaled_scores).set_index("fold")
    unscaled_df = pd.DataFrame(all_unscaled_scores).set_index("fold")

    logger.info(f"Scaled CV summary:\n{scaled_df.describe().T}")
    if not unscaled_df.empty:
        logger.info(f"Unscaled CV summary:\n{unscaled_df.describe().T}")

    return {"scaled": scaled_df, "unscaled": unscaled_df}
