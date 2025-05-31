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


def boxplots(df, cols, color="#0060ff", save_dir=None):
    """
    Plots box plots for numerical columns in flights data, with major statistical annotations.
    Optionally saves each plot in the specified directory and displays them.

    Parameters:
        df (pd.DataFrame): The flights dataframe.
        cols (list): List of numerical columns to plot.
        color (str, optional): Box color.
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
    """
    sns.set_theme(style="darkgrid", font_scale=1.2)
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "flights_boxplots")
        os.makedirs(save_dir, exist_ok=True)

    for col in cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Column '{col}' is not numeric. Skipping.")
            continue

        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        median = data.median()
        iqr = q3 - q1
        min_val, max_val = data.min(), data.max()
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            x=data,
            color=color,
            linewidth=1.5,
            fliersize=5,
            boxprops=dict(alpha=0.7, edgecolor="black"),
            medianprops=dict(color="#ff6600", linewidth=2),
        )
        plt.title(f"Box Plot of {col}", fontsize=16, fontweight="bold")
        plt.xlabel(col, fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        y = 0  # y-position for annotation (horizontal boxplot)

        # Annotate statistics
        stats = [
            (median, y + 0.15, f"Median: {median:.2f}", "#ff6600", 13, "bold"),
            (q1, y - 0.18, f"Q1: {q1:.2f}", "#ff6600", 12, "normal"),
            (q3, y - 0.18, f"Q3: {q3:.2f}", "#ff6600", 12, "normal"),
            ((q1 + q3) / 2, y + 0.22, f"IQR: {iqr:.2f}", "#6D597A", 12, "normal"),
            (min_val, y + 0.12, f"Min: {min_val:.2f}", "#43AA8B", 11, "normal"),
            (max_val, y + 0.12, f"Max: {max_val:.2f}", "#F94144", 11, "normal"),
            (
                lower_bound,
                y - 0.32,
                f"Lower Outlier Bound\n(Q1 - 1.5×IQR): {lower_bound:.2f}",
                "#4361EE",
                11,
                "normal",
            ),
            (
                upper_bound,
                y - 0.32,
                f"Upper Outlier Bound\n(Q3 + 1.5×IQR): {upper_bound:.2f}",
                "#F3722C",
                11,
                "normal",
            ),
        ]
        for x, y_pos, text, color, size, weight in stats:
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(x, y_pos),
                textcoords="data",
                ha="center",
                color=color,
                fontsize=size,
                fontweight=weight,
                arrowprops=dict(arrowstyle="->", color=color)
                if "Bound" not in text and "IQR" not in text
                else None,
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="#e0e0e0", ec="#bbbbbb", alpha=0.7
                )
                if "IQR" in text
                else None,
            )

        if save_dir is not None:
            filename = os.path.join(save_dir, f"{col}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved box plot for '{col}' to {filename}")

        plt.show()
        plt.close()


def barplot_univariate(
    df,
    cols,
    top_n=None,
    color="#0060ff",
    sort=None,
    save_dir=None,
):
    """
    Plots bar graphs of value counts for multiple categorical columns in flights data, with frequency annotations.
    Optionally saves each plot in a 'flights_barplots' subfolder.

    Parameters:
        df (pd.DataFrame): The flights dataframe.
        cols (list): List of columns to plot.
        top_n (int, optional): Show only the top N categories.
        color (str, optional): Bar color.
        sort (str, optional): 'asc' for ascending, 'desc' for descending, None for index order.
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
    """
    sns.set_theme(style="darkgrid", font_scale=1.2)
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "flights_barplots")
        os.makedirs(save_dir, exist_ok=True)

    for col in cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
        if not pd.api.types.is_categorical_dtype(
            df[col]
        ) and not pd.api.types.is_object_dtype(df[col]):
            print(f"Column '{col}' is not categorical. Skipping.")
            continue

        counts = df[col].value_counts(dropna=False)
        if sort == "asc":
            counts = counts.sort_values(ascending=True)
        elif sort == "desc":
            counts = counts.sort_values(ascending=False)
        else:
            counts = counts.sort_index()
        if top_n:
            counts = counts.head(top_n)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color=color)
        plt.title(f"Frequency of {col}", fontsize=18, fontweight="bold")
        plt.xlabel(col, fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(rotation=45, ha="right")

        # Add annotations
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )
        plt.tight_layout()

        if save_dir is not None:
            filename = os.path.join(save_dir, f"barplot_flights_{col}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot for '{col}' to {filename}")

        plt.show()
        plt.close()


def pairplots(df, color="#0060ff", save_dir=None):
    """
    Generates a pair plot (scatter matrix) for all numerical columns in the dataframe.
    Optionally saves the plot in the specified directory and displays it.

    Parameters:
        df (pd.DataFrame): The flights dataframe.
        color (str, optional): Color for the plots.
        save_dir (str or None): Directory to save the plot. If None, plot is not saved.
    """
    # Select only numerical columns
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        print("Not enough numerical columns for a pairplot. Skipping.")
        return

    sns.set_theme(style="ticks")
    pair_grid = sns.pairplot(
        num_df,
        diag_kind="kde",
        plot_kws={"color": color, "edgecolor": "black", "alpha": 0.6},
    )
    pair_grid.figure.suptitle(
        "Pair Plot (Scatter Matrix) of Numerical Features",
        fontsize=20,
        fontweight="bold",
        color="#333333",
        y=1.02,
    )
    plt.tight_layout()

    if save_dir is not None:
        save_dir = os.path.join(save_dir, "flights_pairplot")
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, "pairplot.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved pair plot to {filename}")

    plt.show()
    plt.close()


def correlation_heatmap(
    df,
    cols=None,
    annot=True,
    cmap="Blues",
    save_dir=None,
    filename="correlation_heatmap.png",
    figsize=(8, 6),
):
    """
    Plots a heatmap of the correlation matrix for the specified columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame.
        cols (list or None): List of columns to include. If None, uses all numeric columns.
        annot (bool): Whether to annotate the heatmap with correlation values.
        cmap (str): Colormap for the heatmap.
        save_dir (str or None): Directory to save the plot. If None, plot is not saved.
        filename (str): Name of the saved file.
        figsize (tuple): Figure size.
    """
    if cols is None:
        data = df.select_dtypes(include="number")
    else:
        data = df[cols]

    if data.shape[1] < 2:
        print("Not enough columns for a correlation heatmap. Skipping.")
        return

    corr = data.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        square=True,
        annot_kws={"size": 12, "weight": "bold", "color": "#333333"},
    )
    plt.title("Correlation Heatmap", fontsize=16, fontweight="bold", color="#333333")
    plt.xticks(fontsize=12, color="#555555")
    plt.yticks(fontsize=12, color="#555555")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation heatmap to {path}")

    plt.show()
    plt.close()


def boxplot_bivariate(
    df, cat_cols, num_cols, color="#0060ff", save_dir=None, rotation=30
):
    """
    Plots box plots for each combination of categorical and numerical columns in flights data.
    Annotates each box with its median value.
    Optionally saves each plot in the specified directory and displays them.

    Parameters:
        df (pd.DataFrame): The flights dataframe.
        cat_cols (list): List of categorical columns for x-axis.
        num_cols (list): List of numerical columns for y-axis.
        color (str, optional): Box color.
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
        rotation (int, optional): Degrees to rotate x-axis tick labels.
    """
    sns.set_theme(style="darkgrid", font_scale=1.2)
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "flights_boxplots_cat_num")
        os.makedirs(save_dir, exist_ok=True)

    for cat_col in cat_cols:
        if cat_col not in df.columns or not (
            pd.api.types.is_categorical_dtype(df[cat_col])
            or pd.api.types.is_object_dtype(df[cat_col])
        ):
            print(f"Column '{cat_col}' is not categorical or not found. Skipping.")
            continue
        for num_col in num_cols:
            if num_col not in df.columns or not pd.api.types.is_numeric_dtype(
                df[num_col]
            ):
                print(f"Column '{num_col}' is not numeric or not found. Skipping.")
                continue

            plt.figure(figsize=(14, 8))
            ax = sns.boxplot(
                x=df[cat_col],
                y=df[num_col],
                color=color,
                linewidth=1.5,
                fliersize=5,
                boxprops=dict(alpha=0.7, edgecolor="black"),
                medianprops=dict(color="#ff6600", linewidth=2),
            )
            plt.title(
                f"Box Plot of {num_col} by {cat_col}",
                fontsize=20,
                fontweight="bold",
                color="#333333",
            )
            plt.xlabel(cat_col, fontsize=16, color="#555555")
            plt.ylabel(num_col, fontsize=16, color="#555555")
            plt.tick_params(axis="both", which="major", labelsize=14, colors="#777777")
            plt.xticks(rotation=rotation)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Add median annotations
            medians = df.groupby(cat_col, observed=False)[num_col].median()
            for i, category in enumerate(medians.index):
                median_val = medians[category]
                ax.annotate(
                    f"{median_val:.2f}",
                    xy=(i, median_val),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                    color="#ff6600",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#ff6600",
                        lw=1,
                        alpha=0.7,
                    ),
                )

            if save_dir is not None:
                filename = os.path.join(save_dir, f"{num_col}_by_{cat_col}.png")
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved box plot for {num_col} by {cat_col} to {filename}")

            plt.show()
            plt.close()


def barplot_bivariate(
    df,
    cat_cols,
    num_cols,
    aggfunc="mean",
    palette=None,
    color="#0060ff",
    save_dir=None,
    rotation=30,
):
    """
    Plots bar plots for each combination of categorical and numerical columns in flights data.
    Annotates each bar with its aggregated value (mean, median, mode, sum, etc.).
    Optionally saves each plot in the specified directory and displays them.

    Parameters:
        df (pd.DataFrame): The flights dataframe.
        cat_cols (list): List of categorical columns for x-axis.
        num_cols (list): List of numerical columns for y-axis.
        aggfunc (str, optional): Aggregation function ('mean', 'median', 'mode', 'sum', etc.).
        palette (str or None, optional): Color palette for the bars. If None, all bars are the same color.
        color (str, optional): Single color for all bars if palette is None.
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
        rotation (int, optional): Degrees to rotate x-axis tick labels.
    """
    sns.set_theme(style="darkgrid", font_scale=1.2)
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "flights_barplots_cat_num")
        os.makedirs(save_dir, exist_ok=True)

    for cat_col in cat_cols:
        if cat_col not in df.columns or not (
            pd.api.types.is_categorical_dtype(df[cat_col])
            or pd.api.types.is_object_dtype(df[cat_col])
        ):
            print(f"Column '{cat_col}' is not categorical or not found. Skipping.")
            continue
        for num_col in num_cols:
            if num_col not in df.columns or not pd.api.types.is_numeric_dtype(
                df[num_col]
            ):
                print(f"Column '{num_col}' is not numeric or not found. Skipping.")
                continue

            # Handle 'mode' aggregation
            if aggfunc == "mode":
                agg_df = (
                    df.groupby(cat_col, observed=False)[num_col]
                    .agg(
                        lambda x: x.mode().iloc[0]
                        if not x.mode().empty
                        else float("nan")
                    )
                    .reset_index()
                )
            else:
                agg_df = (
                    df.groupby(cat_col, observed=False)[num_col]
                    .agg(aggfunc)
                    .reset_index()
                )

            plt.figure(figsize=(14, 8))
            barplot_kwargs = dict(
                data=agg_df,
                x=cat_col,
                y=num_col,
                order=agg_df[cat_col],
            )
            if palette is not None:
                barplot_kwargs.update(dict(hue=cat_col, palette=palette, legend=False))
            else:
                barplot_kwargs.update(dict(color=color))

            ax = sns.barplot(**barplot_kwargs)

            plt.title(
                f"{aggfunc.capitalize()} of {num_col} by {cat_col}",
                fontsize=20,
                fontweight="bold",
                color="#333333",
            )
            plt.xlabel(cat_col, fontsize=16, color="#555555")
            plt.ylabel(
                f"{aggfunc.capitalize()} of {num_col}", fontsize=16, color="#555555"
            )
            plt.tick_params(axis="both", which="major", labelsize=14, colors="#777777")
            plt.xticks(rotation=rotation)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Annotate bars with values
            for bar, value in zip(ax.patches, agg_df[num_col]):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                    color="#ff6600",
                    xytext=(0, 8),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#ff6600",
                        lw=1,
                        alpha=0.7,
                    ),
                )

            if save_dir is not None:
                filename = os.path.join(
                    save_dir, f"{aggfunc}_{num_col}_by_{cat_col}.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(
                    f"Saved bar plot for {aggfunc} of {num_col} by {cat_col} to {filename}"
                )

            plt.show()
            plt.close()


from matplotlib.ticker import MaxNLocator


def lineplots(
    df,
    date_parts=["year", "month", "day_name"],
    numeric_cols=None,
    color_palette=None,
    save_dir=None,
    figsize=(14, 8),
):
    """
    Generates line plots between date parts (year, month, day_name) and specified numerical columns.
    Optionally saves each plot in the specified directory and displays them.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        date_parts (list): List of date part columns to use (default: ['year', 'month', 'day_name']).
        numeric_cols (list): List of numerical columns to plot against date parts.
        color_palette (list or None): List of colors for different numeric columns.
        save_dir (str or None): Directory to save plots. If None, plots are not saved.
        figsize (tuple): Figure size for the plots.
    """
    sns.set_theme(style="darkgrid", font_scale=1.2)
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if color_palette is None:
        color_palette = sns.color_palette("husl", len(numeric_cols))

    if save_dir is not None:
        save_dir = os.path.join(save_dir, "date_numeric_lineplots")
        os.makedirs(save_dir, exist_ok=True)

    for part in date_parts:
        if part not in df.columns:
            print(f"Column '{part}' not found in DataFrame. Skipping.")
            continue

        plt.figure(figsize=figsize)
        if part == "day_name":
            weekday_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            grouped = (
                df.groupby(part, observed=False)[numeric_cols]
                .mean()
                .reindex(weekday_order)
            )
        else:
            grouped = df.groupby(part, observed=False)[numeric_cols].mean().sort_index()

        for i, col in enumerate(numeric_cols):
            plt.plot(
                grouped.index,
                grouped[col],
                marker="o",
                linewidth=2.5,
                markersize=8,
                label=col,
                color=color_palette[i],
            )
            for x, y in zip(grouped.index, grouped[col]):
                plt.annotate(
                    f"{y:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontweight="bold",
                    fontsize=9,
                    color=color_palette[i],
                )

        plt.title(
            f"Average {', '.join(numeric_cols)} by {part.capitalize()}",
            fontsize=18,
            fontweight="bold",
        )
        plt.xlabel(part.capitalize(), fontsize=14)
        plt.ylabel("Average Value", fontsize=14)
        if part != "day_name":
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            plt.xticks(rotation=45, ha="right")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            filename = os.path.join(
                save_dir, f"lineplot_{part}_vs_{'_'.join(numeric_cols)}.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {filename}")

        plt.show()
        plt.close()
