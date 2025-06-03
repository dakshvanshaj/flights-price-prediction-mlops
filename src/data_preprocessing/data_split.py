import pandas as pd
from typing import Tuple, Dict

SplitSet = Tuple[pd.DataFrame, pd.Series]


def chronological_percentage_split(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    train_pct: float,
    val_pct: float,
    test_pct: float,
) -> Dict[str, SplitSet]:
    """
    Chronologically splits a DataFrame into train, validation, test, and holdout sets based on given percentages.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column.
        target_column: Name of the target column.
        train_pct: Fraction of data for training (between 0 and 1).
        val_pct: Fraction of data for validation (between 0 and 1).
        test_pct: Fraction of data for testing (between 0 and 1).

    Returns:
        Dictionary with keys 'train', 'validation', 'test', 'holdout', each mapping to (X, y).

    Raises:
        ValueError: If columns are missing or percentages are invalid.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    if not (0 <= train_pct <= 1 and 0 <= val_pct <= 1 and 0 <= test_pct <= 1):
        raise ValueError("Percentages must be between 0 and 1.")
    if train_pct + val_pct + test_pct > 1:
        raise ValueError("Sum of train_pct, val_pct, and test_pct must be <= 1.")

    try:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    except Exception as e:
        raise ValueError(f"Error converting '{date_column}' to datetime: {e}")

    if df[date_column].isnull().any():
        raise ValueError(f"Null values found in '{date_column}' after conversion.")

    df = df.sort_values(date_column).reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    test_end = val_end + int(n * test_pct)

    splits = {
        "train": df.iloc[:train_end],
        "validation": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:test_end],
        "holdout": df.iloc[test_end:],
    }

    result = {}
    for name, subset in splits.items():
        if subset.empty:
            print(f"Warning: '{name}' set is empty for the given percentage split.")
        X = subset.drop(columns=[target_column])
        y = subset[target_column]
        result[name] = (X, y)

    return result


def chronological_split(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    train_end: str,
    val_end: str,
    test_end: str,
) -> Dict[str, SplitSet]:
    """
    Chronologically splits a DataFrame into train, validation, test, and holdout sets.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column.
        target_column: Name of the target column.
        train_end: End date (inclusive) for training set.
        val_end: End date (inclusive) for validation set.
        test_end: End date (inclusive) for test set.

    Returns:
        Dictionary with keys 'train', 'validation', 'test', 'holdout', each mapping to (X, y).

    Raises:
        ValueError: If columns are missing or date ranges are invalid.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    try:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    except Exception as e:
        raise ValueError(f"Error converting '{date_column}' to datetime: {e}")

    if df[date_column].isnull().any():
        raise ValueError(f"Null values found in '{date_column}' after conversion.")

    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    test_end = pd.to_datetime(test_end)

    if not (train_end <= val_end <= test_end):
        raise ValueError("Date ranges must satisfy train_end <= val_end <= test_end.")

    df = df.sort_values(date_column).reset_index(drop=True)

    train_mask = df[date_column] <= train_end
    val_mask = (df[date_column] > train_end) & (df[date_column] <= val_end)
    test_mask = (df[date_column] > val_end) & (df[date_column] <= test_end)
    holdout_mask = df[date_column] > test_end

    splits = {
        "train": df[train_mask],
        "validation": df[val_mask],
        "test": df[test_mask],
        "holdout": df[holdout_mask],
    }

    result = {}
    for name, subset in splits.items():
        if subset.empty:
            print(f"Warning: '{name}' set is empty for the given date range.")
        X = subset.drop(columns=[target_column])
        y = subset[target_column]
        result[name] = (X, y)

    return result
