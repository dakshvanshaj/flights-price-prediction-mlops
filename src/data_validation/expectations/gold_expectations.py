import logging
from typing import List, Optional
from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)


def build_gold_expectations(
    expected_cols_ordered: List[str],
    target_col: str,
    scaler_strategy: str,
    scaled_cols: Optional[List[str]] = None,
) -> List:
    """
    Builds a list of Great Expectations for the Gold data layer.

    These expectations adapt based on the scaler used, ensuring the
    validation logic is always correct.
    """
    logger.info("Building expectations for the Gold data quality gate...")
    logger.info(
        f"Applying scaler-specific expectations for strategy: '{scaler_strategy}'"
    )

    # --- 1. Final Schema and Column Order Check ---
    expect_columns_ordered = gxe.ExpectTableColumnsToMatchOrderedList(
        column_list=expected_cols_ordered,
        meta={"notes": "Verify the final model-ready schema and column order."},
    )

    # --- 2. Data Type Conformance (Post-Processing) ---
    type_expectations = []
    for col in expected_cols_ordered:
        type_expectations.append(
            gxe.ExpectColumnValuesToBeInTypeList(
                column=col,
                type_list=["int", "int64", "float", "float64"],
                meta={"notes": f"Verify column {col} is a numeric type for modeling."},
            )
        )

    # --- 3. Check for Any Missing Values ---
    not_null_expectations = []
    for col in expected_cols_ordered:
        not_null_expectations.append(
            gxe.ExpectColumnValuesToNotBeNull(
                column=col,
                meta={
                    "notes": f"Verify no missing values in final model-ready column: {col}."
                },
            )
        )

    # --- 4. SCALER-SPECIFIC CHECKS ---
    scaled_value_expectations = []
    if scaled_cols:
        if scaler_strategy == "standard":
            for col in scaled_cols:
                scaled_value_expectations.append(
                    gxe.ExpectColumnMeanToBeBetween(
                        column=col,
                        min_value=-0.2,
                        max_value=0.2,
                        meta={
                            "notes": f"Verify standardized column {col} has a mean near 0."
                        },
                    )
                )
                scaled_value_expectations.append(
                    gxe.ExpectColumnStdevToBeBetween(
                        column=col,
                        min_value=0.8,
                        max_value=1.2,
                        meta={
                            "notes": f"Verify standardized column {col} has a stdev near 1."
                        },
                    )
                )
        elif scaler_strategy == "minmax":
            for col in scaled_cols:
                scaled_value_expectations.append(
                    gxe.ExpectColumnValuesToBeBetween(
                        column=col,
                        min_value=0,
                        max_value=1.01,
                        meta={
                            "notes": f"Verify min-max scaled column {col} is within the [0, 1] range."
                        },
                    )
                )
        elif scaler_strategy == "robust":
            for col in scaled_cols:
                scaled_value_expectations.append(
                    gxe.ExpectColumnMedianToBeBetween(
                        column=col,
                        min_value=-0.2,
                        max_value=0.2,
                        meta={
                            "notes": f"Verify robust scaled column {col} has a median near 0."
                        },
                    )
                )
        else:
            logger.warning(
                f"No specific scaler expectations defined for strategy: '{scaler_strategy}'. Skipping scaler checks."
            )

    # --- Assemble the final list of all expectations ---
    all_expectations = [
        expect_columns_ordered,
    ]

    # --- 5. Conditional Target Variable Sanity Check ---
    # Only check if the target is non-negative if it has NOT been scaled.
    if target_col not in (scaled_cols or []):
        expect_target_range = gxe.ExpectColumnValuesToBeBetween(
            column=target_col,
            min_value=0,
            meta={
                "notes": "Sanity check for the target variable's range (pre-scaling)."
            },
        )
        all_expectations.append(expect_target_range)
    else:
        logger.warning(
            f"Target column '{target_col}' is part of the scaled columns. Skipping non-negative check."
        )

    # --- 6. Data Volume Check ---
    expect_number_of_rows = gxe.ExpectTableRowCountToBeBetween(
        min_value=10,
        meta={"notes": "Ensures the gold table is not empty after processing."},
    )
    all_expectations.append(expect_number_of_rows)

    # --- Add remaining expectations to the final list ---
    all_expectations.extend(type_expectations)
    all_expectations.extend(not_null_expectations)
    all_expectations.extend(scaled_value_expectations)

    logger.info(
        f"Successfully built {len(all_expectations)} expectations for the Gold gate."
    )
    return all_expectations
