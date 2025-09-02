import shap
import logging
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional

# Set matplotlib backend to Agg to avoid GUI issues in non-interactive environments
import matplotlib

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def get_shap_explanation(
    model: Any, X: pd.DataFrame, algorithm: str = "tree"
) -> Optional[shap.Explanation]:
    """
    Calculates SHAP explanation object for a given model and dataset.

    This function uses `shap.Explainer` which automatically selects an
    appropriate explainer for the given model type (e.g., TreeExplainer for
    tree-based models, KernelExplainer for others).

    Args:
        model: The trained model instance.
        X: The DataFrame of features for which to generate explanations.

    Returns:
        A SHAP Explanation object, or None if the calculation fails.
    """
    try:
        logger.info(f"Creating SHAP explainer for model type: {type(model).__name__}")
        explainer = shap.Explainer(model, X, algorithm)
        logger.info("Calculating SHAP values...")
        explanation = explainer(X)
        logger.info("SHAP values calculated successfully.")
        return explanation
    except Exception as e:
        logger.error(f"Failed to calculate SHAP values: {e}", exc_info=True)
        return None


def log_shap_summary_plot(explanation: shap.Explanation, title: str):
    """
    Generates and logs a SHAP summary plot (beeswarm) to MLflow.

    Args:
        explanation: The SHAP Explanation object.
        title: The title for the plot and the artifact name.
    """
    logger.info(f"Generating and logging SHAP summary plot: {title}")
    try:
        shap.summary_plot(explanation, show=False)
        fig = plt.gcf()  # summary_plot uses the current figure
        fig.suptitle(title, y=1.0)  # Adjust y to prevent title overlap
        fig.tight_layout()
        mlflow.log_figure(fig, f"shap_plots/{title}.png")
        logger.info(f"Successfully logged SHAP summary plot: {title}")
    except Exception as e:
        logger.warning(
            f"Could not generate or log SHAP summary plot for '{title}': {e}"
        )
    finally:
        if "fig" in locals():
            plt.close(fig)


def log_shap_bar_plot(explanation: shap.Explanation, title: str):
    """
    Generates and logs a SHAP bar plot (global feature importance) to MLflow.

    Args:
        explanation: The SHAP Explanation object.
        title: The title for the plot and the artifact name.
    """
    logger.info(f"Generating and logging SHAP bar plot: {title}")
    try:
        shap.plots.bar(explanation, show=False)
        fig = plt.gcf()
        fig.suptitle(title, y=1.0)
        fig.tight_layout()
        mlflow.log_figure(fig, f"shap_plots/{title}.png")
        logger.info(f"Successfully logged SHAP bar plot: {title}")
    except Exception as e:
        logger.warning(f"Could not generate or log SHAP bar plot for '{title}': {e}")
    finally:
        if "fig" in locals():
            plt.close(fig)


def log_shap_waterfall_plot(
    explanation: shap.Explanation, instance_index: int, title: str
):
    """
    Generates and logs a SHAP waterfall plot for a single local prediction.

    Args:
        explanation: The SHAP Explanation object.
        instance_index: The index of the instance in the explanation object to plot.
        title: The title for the plot and the artifact name.
    """
    logger.info(f"Generating and logging SHAP waterfall plot: {title}")
    try:
        shap.plots.waterfall(explanation[instance_index], show=False)
        fig = plt.gcf()
        fig.suptitle(title, y=1.0)
        fig.tight_layout()
        mlflow.log_figure(fig, f"shap_plots/{title}.png")
        logger.info(f"Successfully logged SHAP waterfall plot: {title}")
    except Exception as e:
        logger.warning(
            f"Could not generate or log SHAP waterfall plot for '{title}': {e}"
        )
    finally:
        if "fig" in locals():
            plt.close(fig)


def shap_plots(
    model: Any, X: pd.DataFrame, log_prefix: str, n_local_plots: int = 3
) -> None:
    """
    Calculates and logs a standard set of SHAP explanation plots.

    This function orchestrates the generation of global (summary, bar) and
    local (waterfall) SHAP plots and logs them to MLflow. It includes
    subsampling for performance on large datasets.

    Args:
        model: The trained model instance.
        X: The feature DataFrame for explanation.
        log_prefix: Prefix for naming artifacts (e.g., "validation", "test").
        n_local_plots: The number of local (instance-level) waterfall plots to generate.
    """
    logger.info(f"--- Generating SHAP explanations for '{log_prefix}' data ---")

    # SHAP can be slow. We'll sample the data for the explainer if it's large.
    X_for_shap = X
    if len(X) > 2000:
        logger.info(f"Subsampling data for SHAP from {len(X)} to 2000 instances.")
        X_for_shap = X.sample(n=2000, random_state=42)

    explanation = get_shap_explanation(model, X_for_shap)

    if explanation is None:
        logger.warning(
            f"Skipping SHAP plots for '{log_prefix}' due to calculation failure."
        )
        return

    log_shap_summary_plot(explanation, title=f"[{log_prefix}] SHAP Summary Plot")
    log_shap_bar_plot(
        explanation, title=f"[{log_prefix}] SHAP Feature Importance (Bar)"
    )

    for i in range(min(n_local_plots, len(X_for_shap))):
        log_shap_waterfall_plot(
            explanation,
            instance_index=i,
            title=f"[{log_prefix}] SHAP Waterfall Plot for Instance {i}",
        )
    logger.info(f"Successfully logged all SHAP plots for '{log_prefix}'.")
