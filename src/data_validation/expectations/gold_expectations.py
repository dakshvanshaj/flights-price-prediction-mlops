import logging
from typing import List
from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)


def build_gold_expectations(
    expected_cols_ordered: List[str],
    scaled_cols: List[str],
    target_col: str,
) -> List:
    """
    Builds a list of Great Expectations for the Gold data layer.

    These expectations verify that the final feature-engineered and
    preprocessed data is ready for model training.
    """
    logger.info("Building expectations for the Gold data quality gate...")

    # --- 1. Final Schema and Column Order Check ---
    # This is the most important check. It ensures the data has the exact
    # structure the model was trained on.
    expect_columns_ordered = gxe.ExpectTableColumnsToMatchOrderedList(
        column_list=expected_cols_ordered,
        meta={"notes": "Verify the final model-ready schema and column order."},
    )

    # --- 2. Data Type Conformance (Post-Processing) ---
    # After all transformations, all columns should be numeric (int or float).
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
    # After imputation, no columns should have missing values.
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

    # --- 4. Scaled Feature Range Check ---
    # If using a Min-Max Scaler, all scaled features should be between 0 and 1.
    scaled_value_expectations = []
    for col in scaled_cols:
        scaled_value_expectations.append(
            gxe.ExpectColumnValuesToBeBetween(
                column=col,
                min_value=0,
                max_value=1.01,  # Add a small buffer for floating point inaccuracies
                meta={
                    "notes": f"Verify scaled feature {col} is within the [0, 1] range."
                },
            )
        )

    # --- 5. Target Variable Sanity Check ---
    # A simple sanity check on the target variable's range.
    # This should be adapted based on your specific target (e.g., 'price').
    expect_target_range = gxe.ExpectColumnValuesToBeBetween(
        column=target_col,
        min_value=0,  # Price cannot be negative
        meta={"notes": "Sanity check for the target variable's range."},
    )

    # --- 6. Data Volume Check ---
    # check if volume of data is in the valid range
    expect_number_of_rows = gxe.ExpectTableRowCountToBeBetween(
        min_value=10,
        meta={"notes": "Ensures the gold table is not empty after processing."},
    )

    # --- Assemble the final list of all expectations ---
    all_expectations = [
        expect_columns_ordered,
        expect_target_range,
        expect_number_of_rows,
    ]
    all_expectations.extend(type_expectations)
    all_expectations.extend(not_null_expectations)
    all_expectations.extend(scaled_value_expectations)

    logger.info(
        f"Successfully built {len(all_expectations)} expectations for the Gold gate."
    )
    return all_expectations
