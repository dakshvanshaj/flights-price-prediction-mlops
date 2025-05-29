import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_flights_per_year(date_series, series_name="Date Column"):
    dates = pd.to_datetime(date_series, errors="coerce").dropna()
    if dates.empty:
        print(f"No valid dates in {series_name}.")
        return
    year_counts = dates.dt.year.value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    year_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Flights per Year - {series_name}", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Number of Flights")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_flights_per_month(date_series, series_name="Date Column"):
    dates = pd.to_datetime(date_series, errors="coerce").dropna()
    if dates.empty:
        print(f"No valid dates in {series_name}.")
        return
    month_counts = dates.dt.month.value_counts().sort_index()
    # Map month numbers to names for better x-axis labels
    month_names = pd.to_datetime(month_counts.index, format="%m").month_name().str[:3]
    plt.figure(figsize=(10, 5))
    plt.bar(month_names, month_counts.values, color="coral", edgecolor="black")
    plt.title(f"Flights per Month (All Years) - {series_name}", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Number of Flights")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def flights_distribution(df, cols, color="#0060ff", save_dir=None):
    """
    Displays (and optionally saves) distribution histograms with KDE for specified numerical columns.

    Parameters:
        df (pd.DataFrame): The flights dataframe.
        cols (list): List of columns to plot.
        color (str): Histogram color.
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for col in cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Column '{col}' is not numeric. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(
            df[col].dropna(),
            bins=40,
            kde=True,
            color=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
        )
        plt.title(f"Distribution of {col}", fontsize=16, fontweight="bold")
        plt.xlabel(col, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            filename = os.path.join(save_dir, f"{col}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot for '{col}' to {filename}")

        plt.show()
        plt.close()
