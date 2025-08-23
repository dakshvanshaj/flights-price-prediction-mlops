import numpy as np
import pandas as pd
from typing import Union, Dict
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


def calculate_all_regression_metrics(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """Calculates a comprehensive set of regression metrics."""
    metrics = {
        "max_error": max_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
        # "smape": s_mape(y_true, y_pred),
        # "mdape": median_absolute_percentage_error(y_true, y_pred),
    }
    return metrics


def s_mape(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The SMAPE value.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle cases where denominator might be zero to avoid division by zero
    # A common approach is to set the error to 0 if both y_true and y_pred are 0
    # or to a small epsilon if only one is 0.
    # For simplicity here, we'll use a small epsilon for the denominator if it's zero.
    # In a production setting, carefully consider edge cases for zero values.
    smape_values = np.where(denominator == 0, 0, numerator / denominator)
    return float(np.mean(smape_values) * 100)


def median_absolute_percentage_error(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculates the Median Absolute Percentage Error (MdAPE).

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: The Median Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate individual absolute percentage errors
    # Handle division by zero for y_true values close to zero by returning a large value
    # instead of infinity, similar to sklearn's MAPE handling.
    # A small epsilon is added to the denominator to prevent division by zero.
    epsilon = np.finfo(float).eps
    absolute_percentage_errors = np.abs((y_true - y_pred) / (y_true + epsilon)) * 100

    # Calculate the median of the absolute percentage errors
    mdape = np.median(absolute_percentage_errors)

    return float(mdape)
