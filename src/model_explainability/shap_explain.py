import logging
import os
import tempfile
from collections import defaultdict
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

# Set matplotlib backend to Agg to avoid GUI issues in non-interactive environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _get_feature_group(feature_name: str) -> str:
    """
    Determines the group for a given feature name, which is useful for
    aggregating one-hot encoded or cyclical features.

    Args:
        feature_name: The name of the feature column.

    Returns:
        The name of the group the feature belongs to.
    """
    # 1. Group cyclical (sin/cos) features
    if feature_name.endswith("_sin") or feature_name.endswith("_cos"):
        # e.g., "month_sin" -> "month"
        return feature_name.rsplit("_", 1)[0]

    # 2. Group one-hot encoded categorical features
    # These prefixes should match the column names BEFORE one-hot encoding.
    ohe_prefixes = [
        "from_location_",
        "to_location_",
        "agency_",
        "route_",
        "agency_flight_type_",
        "route_agency_",
    ]
    for prefix in ohe_prefixes:
        if feature_name.startswith(prefix):
            return prefix.strip("_")

    # 3. If no group is found, it's a standalone feature
    return feature_name


def _aggregate_shap_values(explanation: shap.Explanation) -> shap.Explanation:
    """
    Aggregates SHAP values for features that were originally one-hot encoded.

    This function groups related columns (e.g., `from_location_A`, `from_location_B`)
    and sums their SHAP values to represent the importance of the original
    categorical feature.

    Args:
        explanation: The original SHAP Explanation object with disaggregated features.

    Returns:
        A new SHAP Explanation object with aggregated feature importances.
    """
    feature_names = explanation.feature_names

    # 1. Group features by their original categorical name
    groups = defaultdict(list)
    for feature in feature_names:
        group_name = _get_feature_group(feature)
        groups[group_name].append(feature)

    # 2. Separate single features from those that need aggregation
    feature_groups = {
        group: members for group, members in groups.items() if len(members) > 1
    }
    single_features = [
        members[0] for group, members in groups.items() if len(members) == 1
    ]

    if not feature_groups:
        logger.info(
            "No feature groups found for aggregation. Returning original SHAP explanation."
        )
        return explanation

    logger.info(
        f"Aggregating SHAP values for feature groups: {list(feature_groups.keys())}"
    )

    original_shap_values = explanation.values
    original_data = explanation.data

    # 3. Build new lists for aggregated values, data, and names
    new_shap_values_list, new_data_list, new_feature_names = [], [], []

    # Add aggregated groups
    for group_name, member_cols in feature_groups.items():
        indices = [feature_names.index(c) for c in member_cols]
        group_shap_sum = original_shap_values[:, indices].sum(axis=1)

        # For the beeswarm plot's color axis, we need a representative value.
        # Summing is incorrect for cyclical features.
        is_cyclical = any(c.endswith("_sin") or c.endswith("_cos") for c in member_cols)

        if is_cyclical:
            # For cyclical features, use the 'cos' part for coloring, as it
            # often represents the main phase of the cycle (1 at start/end, -1 at mid).
            try:
                cos_feature = [c for c in member_cols if c.endswith("_cos")][0]
                cos_idx = feature_names.index(cos_feature)
                group_data_val = original_data[:, cos_idx]
            except IndexError:
                # Fallback if no 'cos' feature is found, just take the first member.
                group_data_val = original_data[:, indices[0]]
        else:
            # For OHE features, summing the data values works because only one is 1.
            group_data_val = original_data[:, indices].sum(axis=1)

        new_shap_values_list.append(group_shap_sum)
        new_data_list.append(group_data_val)
        new_feature_names.append(group_name)

    # Add single, non-grouped features
    for feature_name in single_features:
        idx = feature_names.index(feature_name)
        new_shap_values_list.append(original_shap_values[:, idx])
        new_data_list.append(original_data[:, idx])
        new_feature_names.append(feature_name)

    # 4. Create the final aggregated Explanation object
    final_shap_values = np.array(new_shap_values_list).T
    final_data = np.array(new_data_list).T

    # Sort features by their global mean SHAP value for better plot readability
    mean_abs_shap = np.abs(final_shap_values).mean(0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    return shap.Explanation(
        values=final_shap_values[:, sorted_indices],
        base_values=explanation.base_values,
        data=final_data[:, sorted_indices],
        feature_names=[new_feature_names[i] for i in sorted_indices],
    )


def get_shap_explanation(
    model: Any, X: pd.DataFrame, algorithm: str = "auto"
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
        explainer = shap.Explainer(model, algorithm=algorithm)
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
    explanation: shap.Explanation,
    instance_index: int,
    title: str,
    max_display: int = 20,
):
    """
    Generates and logs a SHAP waterfall plot for a single local prediction.

    Args:
        explanation: The SHAP Explanation object.
        instance_index: The index of the instance in the explanation object to plot.
        title: The title for the plot and the artifact name.
        max_display: The maximum number of features to display in the plot.
    """
    logger.info(f"Generating and logging SHAP waterfall plot: {title}")
    try:
        shap.plots.waterfall(
            explanation[instance_index], max_display=max_display, show=False
        )
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


def log_shap_dependence_plot(explanation: shap.Explanation, feature: str, title: str):
    """
    Generates and logs a SHAP dependence plot to MLflow.

    Args:
        explanation: The SHAP Explanation object.
        feature: The name of the feature to plot.
        title: The title for the plot and artifact name.
    """
    logger.info(f"Generating and logging SHAP dependence plot for feature: {feature}")
    try:
        # Create the plot
        shap.dependence_plot(
            feature,
            explanation.values,
            features=explanation.data,
            feature_names=explanation.feature_names,
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle(title, y=1.0)
        fig.tight_layout()
        mlflow.log_figure(fig, f"shap_plots/dependence_plots/{title}.png")
        logger.info(f"Successfully logged SHAP dependence plot: {title}")
    except Exception as e:
        logger.warning(
            f"Could not generate or log SHAP dependence plot for '{title}': {e}"
        )
    finally:
        if "fig" in locals():
            plt.close(fig)


def log_shap_force_plot_html(explanation: shap.Explanation, title: str):
    """
    Generates and logs an interactive SHAP force plot as an HTML artifact.

    Args:
        explanation: The SHAP Explanation object.
        title: The artifact name.
    """
    logger.info(f"Generating and logging SHAP force plot: {title}")
    try:
        # Generate the plot object
        force_plot = shap.force_plot(
            base_value=explanation.base_values[0],
            shap_values=explanation.values,
            features=explanation.data,
            feature_names=explanation.feature_names,
        )
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".html", encoding="utf-8"
        ) as tmp:
            shap.save_html(tmp.name, force_plot)
            tmp_path = tmp.name

        # Log the HTML file to MLflow
        mlflow.log_artifact(tmp_path, f"shap_plots/{title}.html")
        logger.info(f"Successfully logged SHAP force plot: {title}.html")

    except Exception as e:
        logger.warning(f"Could not generate or log SHAP force plot for '{title}': {e}")
    finally:
        # Clean up the temporary file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


def shap_plots(
    model: Any,
    X: pd.DataFrame,
    log_prefix: str,
    n_local_plots: int = 3,
    n_dependence_plots: int = 7,
    aggregate_ohe_features: bool = True,
) -> None:
    """
    Calculates and logs a standard set of SHAP explanation plots.

    This function orchestrates the generation of global (summary, bar) and
    local (waterfall) SHAP plots and logs them to MLflow. It includes
    subsampling for performance on large datasets and can aggregate OHE features.

    Args:
        model: The trained model instance.
        X: The feature DataFrame for explanation.
        log_prefix: Prefix for naming artifacts (e.g., "validation", "test").
        n_local_plots: The number of local (instance-level) waterfall plots to generate.
        n_dependence_plots: The number of dependence plots to generate for top features.
        aggregate_ohe_features: If True, groups OHE features for clearer plots.
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

    # Aggregate features if requested, otherwise use the original explanation
    if aggregate_ohe_features:
        plot_explanation = _aggregate_shap_values(explanation)
    else:
        plot_explanation = explanation

    # --- Log Standard Plots ---
    log_shap_summary_plot(plot_explanation, title=f"[{log_prefix}] SHAP Summary Plot")
    log_shap_bar_plot(
        plot_explanation, title=f"[{log_prefix}] SHAP Feature Importance (Bar)"
    )

    for i in range(min(n_local_plots, len(X_for_shap))):
        log_shap_waterfall_plot(
            plot_explanation,
            instance_index=i,
            title=f"[{log_prefix}] SHAP Waterfall Plot for Instance {i}",
        )

    # --- Log Advanced Plots ---
    top_features = plot_explanation.feature_names[:n_dependence_plots]
    for feature in top_features:
        log_shap_dependence_plot(
            plot_explanation,
            feature=feature,
            title=f"[{log_prefix}] Dependence Plot - {feature}",
        )

    log_shap_force_plot_html(
        plot_explanation, title=f"[{log_prefix}] Global Force Plot"
    )

    logger.info(f"Successfully logged all SHAP plots for '{log_prefix}'.")
