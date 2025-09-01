import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import logging
from scipy import stats
from typing import List, Any

logger = logging.getLogger(__name__)


def scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str, title: str):
    """Generates and logs a scatter plot to MLflow."""
    logger.info(f"Generating and logging plot: {title}")
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        mlflow.log_figure(fig, f"plots/{title}.png")
        logger.info(f"Successfully logged plot: {title}")
    except Exception as e:
        logger.warning(f"Could not generate or log plot for '{title}': {e}")
    finally:
        plt.close(fig)


def residual_plot(
    y_true: pd.Series, y_pred: pd.Series, xlabel: str, ylabel: str, title: str
):
    """Generates and logs a residual plot to MLflow."""
    logger.info(f"Generating and logging plot: {title}")
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        sns.residplot(
            x=y_true,
            y=y_pred,
            ax=ax,
            lowess=True,
            line_kws={"color": "red", "lw": 2, "alpha": 0.8},
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        mlflow.log_figure(fig, f"plots/{title}.png")
        logger.info(f"Successfully logged plot: {title}")
    except Exception as e:
        logger.warning(f"Could not generate or log plot for '{title}': {e}")
    finally:
        plt.close(fig)


def qq_plot_residuals(y_true: pd.Series, y_pred: pd.Series, title: str):
    """Generates and logs a Q-Q plot of residuals to MLflow."""
    logger.info(f"Generating and logging plot: {title}")
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(title)
        mlflow.log_figure(fig, f"plots/{title}.png")
        logger.info(f"Successfully logged plot: {title}")
    except Exception as e:
        logger.warning(f"Could not generate or log plot for '{title}': {e}")
    finally:
        plt.close(fig)


def plot_feature_importance(feature_names: List[str], model: Any, title: str):
    """Generates and logs a feature importance or coefficient plot to MLflow."""
    logger.info(f"Generating and logging plot: {title}")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Importance"
    elif hasattr(model, "coef_"):
        importances = model.coef_
        importance_label = "Coefficient"
    else:
        logger.warning(
            f"Model of type {type(model).__name__} has neither 'feature_importances_' nor 'coef_'. Skipping plot."
        )
        return

    importance_df = (
        pd.DataFrame({"Feature": feature_names, importance_label: importances})
        .sort_values(by=importance_label, ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        sns.barplot(
            x=importance_label,
            y="Feature",
            data=importance_df,
            ax=ax,
        )
        ax.set_title(title)
        mlflow.log_figure(fig, f"plots/{title}.png")
        logger.info(f"Successfully logged plot: {title}")
    except Exception as e:
        logger.warning(f"Could not generate or log plot for '{title}': {e}")
    finally:
        plt.close(fig)
