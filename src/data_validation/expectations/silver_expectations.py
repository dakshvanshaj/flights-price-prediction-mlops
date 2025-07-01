import logging
from typing import List, Dict

from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)


def build_silver_expectations(
    expected_cols_ordered: List[str],
    expected_col_types: Dict[str, str],
    non_null_cols: List[str],
    unique_record_cols: List[str],
) -> List:
    """
    Builds a list of Great Expectations for the Silver data layer.
    These expectations verify that our data processing and cleaning worked correctly.
    """
    logger.info("Building expectations for the Silver data quality gate...")

    # --- 1. Schema and Column Order Check ---
    expect_columns_ordered = gxe.ExpectTableColumnsToMatchOrderedList(
        column_list=expected_cols_ordered,
        meta={"notes": "Verify the final schema, column names, and their exact order."},
    )

    # --- 2. Data Type Conformance ---
    type_expectations = []
    for col, type_str in expected_col_types.items():
        # If the expected type in config is 'category', we must check for multiple
        # valid string representations that different backends might report.
        if type_str == "category":
            expectation = gxe.ExpectColumnValuesToBeInTypeList(
                column=col,
                type_list=["category", "CategoricalDtypeType"],
                meta={"notes": f"Verify column {col} is a categorical type."},
            )
        # For all other simple types, we can use a strict, single-type check.
        else:
            expectation = gxe.ExpectColumnValuesToBeOfType(
                column=col,
                type_=type_str,
                meta={
                    "notes": f"Verify data type optimization for {col} is {type_str}."
                },
            )
        type_expectations.append(expectation)

    # --- 3. Missing Value Check ---
    not_null_expectations = []
    for col in non_null_cols:
        not_null_expectations.append(
            gxe.ExpectColumnValuesToNotBeNull(
                column=col,
                meta={"notes": f"Verify no missing values in critical column: {col}."},
            )
        )

    # --- 4. Duplicate Record Check ---
    expect_no_duplicates = gxe.ExpectCompoundColumnsToBeUnique(
        column_list=unique_record_cols,
        meta={
            "notes": "Verify that there are no duplicate records based on the business key."
        },
    )

    # --- 5. Table Row Count Sanity Check ---
    expect_table_has_rows = gxe.ExpectTableRowCountToBeBetween(
        min_value=1,
        meta={"notes": "Ensures the silver table is not empty after processing."},
    )

    # --- 6. Data Distribution / Business Logic Expectation ---
    expect_price_range = gxe.ExpectColumnMeanToBeBetween(
        column="price",
        min_value=100,
        max_value=5000,
        meta={"notes": "Sanity check for flight prices to detect anomalies."},
    )

    # --- Assemble the final list of all expectations ---
    all_expectations = [
        expect_columns_ordered,
        expect_no_duplicates,
        expect_table_has_rows,
        expect_price_range,
    ]
    all_expectations.extend(type_expectations)
    all_expectations.extend(not_null_expectations)

    logger.info(
        f"Successfully built {len(all_expectations)} expectations for the Silver gate."
    )
    return all_expectations
