import pandas as pd
from typing import Tuple, Dict


def chronological_split(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    train_end: str,
    val_end: str,
    test_end: str,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Chronologically splits a DataFrame into train, validation, test, and Holdout(Production simulation) sets.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column.
        target_column: Name of the target column.
        train_end: End date (inclusive) for training set.
        val_end: End date (inclusive) for validation set.
        test_end: End date (inclusive) for test set.

    Returns:
        Dictionary with keys 'train', 'validation', 'test', 'simulation', each mapping to (X, y).
    """
    # Ensure datetime and sort
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    df = df.sort_values(date_column).reset_index(drop=True)

    # Convert split dates
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    test_end = pd.to_datetime(test_end)

    # Masks
    train_mask = df[date_column] <= train_end
    val_mask = (df[date_column] > train_end) & (df[date_column] <= val_end)
    test_mask = (df[date_column] > val_end) & (df[date_column] <= test_end)
    sim_mask = df[date_column] > test_end

    splits = {
        "train": df[train_mask],
        "validation": df[val_mask],
        "test": df[test_mask],
        "holdout": df[sim_mask],
    }

    # Prepare (X, y) tuples
    result = {}
    for name, subset in splits.items():
        X = subset.drop(columns=[target_column])
        y = subset[target_column]
        result[name] = (X, y)
        if subset.empty:
            print(f"Warning: '{name}' set is empty for the given date range.")

    return result
