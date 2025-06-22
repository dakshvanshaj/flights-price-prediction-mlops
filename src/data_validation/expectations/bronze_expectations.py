# src/data_validation/expectations/bronze_expectations.py
import logging
from typing import List

from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)


def build_bronze_expectations() -> List:
    """
    Builds and returns a list of Expectation objects for the Bronze gate.

    Returns:
        A list of Expectation objects.
    """
    logger.info("Building expectations for the Bronze data quality gate...")

    # --- Expectation 1: Column Existence ---
    expected_columns = [
        "travelCode",
        "userCode",
        "from",
        "to",
        "flightType",
        "price",
        "time",
        "distance",
        "agency",
        "date",
    ]
    expect_columns_to_match_set = gxe.ExpectTableColumnsToMatchSet(
        column_set=expected_columns,
        meta={
            "name": "bronze_columns_match_set",
            "notes": "Checks for presence of all required columns.",
        },
    )

    # --- Expectation 2: Table Row Count ---
    expect_table_has_rows = gxe.ExpectTableRowCountToBeBetween(
        min_value=1,
        meta={
            "name": "bronze_table_has_rows",
            "notes": "Ensures the raw file is not empty.",
        },
    )

    # --- Expectation 3: Primary Key Not Null ---
    expect_travelcode_not_null = gxe.ExpectColumnValuesToNotBeNull(
        column="travelCode",
        meta={
            "name": "bronze_travelCode_not_null",
            "notes": "Ensures the primary identifier is always present.",
        },
    )

    # --- Create a list to hold all expectations ---
    expectations = [
        expect_columns_to_match_set,
        expect_table_has_rows,
        expect_travelcode_not_null,
    ]

    # --- Expectation 4: Programmatic Column Type and Range Checks ---

    # Define column groups
    numeric_columns = ["price", "time", "distance", "userCode", "travelCode"]
    string_columns = ["from", "to", "flightType", "agency"]

    # Generate type checks for numeric columns
    for col in numeric_columns:
        expectations.append(
            gxe.ExpectColumnValuesToBeInTypeList(
                column=col,
                type_list=["int", "int64", "float", "float64"],
                meta={"name": f"bronze_{col}_is_numeric"},
            )
        )
        # Add a non-negative check for relevant numeric columns
        if col in ["price", "distance", "time"]:
            expectations.append(
                gxe.ExpectColumnValuesToBeBetween(
                    column=col,
                    min_value=0,
                    meta={"name": f"bronze_{col}_is_not_negative"},
                )
            )

    # Generate type checks for string columns
    for col in string_columns:
        expectations.append(
            gxe.ExpectColumnValuesToBeInTypeList(
                column=col,
                type_list=["object", "str"],
                meta={"name": f"bronze_{col}_is_string"},
            )
        )

    # --- Expectation 5: More Precise and Robust Individual Checks ---

    # Check if date strings match YYYY-MM-DD format
    expectations.append(
        gxe.ExpectColumnValuesToMatchStrftimeFormat(
            column="date",
            strftime_format="%Y-%m-%d",
            meta={
                "name": "bronze_date_format_is_correct",
                "notes": "Checks if date strings match YYYY-MM-DD format.",
            },
        )
    )

    # Flexibly check for key categorical column values
    expectations.append(
        gxe.ExpectColumnValuesToBeInSet(
            column="flightType",
            value_set=["economic", "firstClass", "premium"],
            mostly=0.90,  # Allow for some flexibility
            meta={
                "name": "bronze_flightType_in_known_set",
                "notes": "Checks that most flight types are from a known list.",
            },
        )
    )

    logger.info(
        f"Successfully built {len(expectations)} expectations for the Bronze gate."
    )
    return expectations
