import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# Checking duplicate values
def check_duplicates(df):
    """
    Check for duplicate rows in the DataFrame and return the number of duplicates.
    """
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = duplicate_rows / len(df) * 100
    print(f"Percentage of rows involved in duplication: {duplicate_percentage:.2f}%")
    return


# Automated eda generator
def generate_eda_report(
    df,
    report_title="EDA Report",
    save_path="eda_report.html",
    minimal=True,
    explorative=False,
):
    """
    Generate an EDA report for the given DataFrame and save it to an HTML file.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    report_title (str): The title of the report.
    save_path (str): The path where the report will be saved.
    """

    # Generate the report
    try:
        profile = ProfileReport(
            df, title=report_title, explorative=explorative, minimal=minimal
        )

    except Exception as e:
        print(f"Error generating EDA report: {e}")

    # Save the report to an HTML file
    try:
        # remove unnecessary multiple progress bars
        Settings().progress_bar = False
        profile.to_file(save_path)
        print(f"EDA report saved to {save_path}")
    except Exception as e:
        print(f"Error saving EDA report: {e}")
    return


def get_date_stats(date_series, series_name="Date Column"):
    """
    Prints and returns basic statistics for a date column.
    Args:
        date_series (pd.Series): The pandas Series containing date information.
        series_name (str): Name of the series for print statements.
    Returns:
        dict: A dictionary containing basic date statistics.
    """
    # Ensure input is a pandas Series and convert to datetime
    dates = pd.to_datetime(date_series, errors="coerce").dropna()
    if dates.empty:
        print(f"No valid dates found in {series_name}.")
        return {}

    stats = {
        "min_date": dates.min(),
        "max_date": dates.max(),
        "time_span": dates.max() - dates.min(),
        "unique_days": dates.nunique(),
        "year_counts": dates.dt.year.value_counts().sort_index(),
        "month_counts": dates.dt.month.value_counts().sort_index(),
        "unique_year_months": dates.dt.to_period("M").nunique(),
    }

    print(f"--- Date Stats for: {series_name} ---")
    print(f"Min date: {stats['min_date'].date()}")
    print(f"Max date: {stats['max_date'].date()}")
    print(f"Time span: {stats['time_span']}")
    print(f"Unique days: {stats['unique_days']}")
    print("\nYear counts:\n", stats["year_counts"])
    print("\nMonth counts:\n", stats["month_counts"])
    print(f"\nUnique year-months: {stats['unique_year_months']}")
    print("--- End of Stats ---")
    return


def count_rows_between_dates(df, date_col, start_date, end_date):
    """
    Counts rows in df[date_col] between start_date and end_date (inclusive).
    Also prints the percentage of total rows in this interval.
    """
    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    mask = (df[date_col] >= pd.to_datetime(start_date)) & (
        df[date_col] <= pd.to_datetime(end_date)
    )
    count = mask.sum()
    total = len(df)
    percent = (count / total) * 100 if total > 0 else 0
    print(
        f"Rows between {start_date} and {end_date}: {count} ({percent:.2f}% of total)"
    )
    return count


def check_missing(df):
    """
    Check for missing values in the DataFrame and return a DataFrame with the results.
    """
    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Values": missing, "Percentage": missing_percentage}
    )
    return missing_df


import warnings


def optimize_dtypes(df, category_threshold=0.5, datetime_threshold=0.8):
    """
    Optimize the data types of DataFrame columns to reduce memory usage,
    including conversion of integers, floats, objects (to category/date), and datetimes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to optimize
    category_threshold : float
        Max ratio of unique/total values to convert object to category
    datetime_threshold : float
        Min ratio of valid datetimes to convert object to datetime

    Returns:
    --------
    pandas.DataFrame
        A copy of the DataFrame with optimized data types
    """
    df_optimized = df.copy()

    # Optimize integer columns
    int_columns = df_optimized.select_dtypes(include=[np.integer]).columns
    for col in int_columns:
        if df_optimized[col].isnull().any():
            continue
        col_min, col_max = df_optimized[col].min(), df_optimized[col].max()
        if col_min >= 0:
            if col_max <= np.iinfo(np.uint8).max:
                df_optimized[col] = df_optimized[col].astype(np.uint8)
            elif col_max <= np.iinfo(np.uint16).max:
                df_optimized[col] = df_optimized[col].astype(np.uint16)
            elif col_max <= np.iinfo(np.uint32).max:
                df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.uint64)
        else:
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif (
                col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max
            ):
                df_optimized[col] = df_optimized[col].astype(np.int32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.int64)

    # Optimize float columns
    float_columns = df_optimized.select_dtypes(include=[np.floating]).columns
    for col in float_columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

    # Optimize object columns (categorical + datetime)
    object_columns = df_optimized.select_dtypes(include=["object"]).columns
    for col in object_columns:
        num_unique = df_optimized[col].nunique(dropna=False)
        num_total = len(df_optimized[col])

        # Attempt datetime conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                converted = pd.to_datetime(df_optimized[col], errors="coerce")
                valid_count = converted.notna().sum()
                if valid_count / num_total >= datetime_threshold:
                    df_optimized[col] = converted
                    continue
            except Exception:
                pass

        # Convert to category if unique/total ratio < threshold
        if num_unique / num_total < category_threshold:
            try:
                df_optimized[col] = df_optimized[col].astype("category")
            except Exception:
                continue

    return df_optimized


def skewness(df):
    """
    Calculate the skewness of each column in the DataFrame.
    """
    skewness = df.skew().reset_index().rename(columns={"index": "Column", 0: "Skew"})
    return skewness
