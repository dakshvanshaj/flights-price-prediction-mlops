# src/data_validation/expectations/bronze_expectations.py
import logging
from typing import List

# Corrected: Use 'gxe' alias for conciseness as per standard convention.
from great_expectations import expectations as gxe

# Create a logger object for this module
logger = logging.getLogger(__name__)


def build_bronze_expectations() -> List:
    """
    Builds and returns a list of Expectation objects for the Bronze gate.
    This version creates direct instances of Expectation classes.

    Returns:
        A list of Expectation objects.
    """
    logger.info("Building expectations for the Bronze data quality gate...")

    # --- Expectation 1: Column Existence ---
    # This check ensures that all expected columns are present, regardless of order.
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
            "notes": "Checks for the presence of all required columns.",
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

    # --- Expectation 3: No Null Primary Keys ---
    expect_travelcode_not_null = gxe.ExpectColumnValuesToNotBeNull(
        column="travelCode",
        meta={
            "name": "bronze_travelCode_not_null",
            "notes": "Ensures the primary identifier is always present.",
        },
    )

    # --- Expectation 4: Basic Column Type Checks ---
    numeric_types = ["int", "int64", "float", "float64"]
    string_types = ["object", "str"]

    expect_price_is_numeric = gxe.ExpectColumnValuesToBeInTypeList(
        column="price",
        type_list=numeric_types,
        meta={"name": "bronze_price_is_numeric"},
    )
    expect_time_is_numeric = gxe.ExpectColumnValuesToBeInTypeList(
        column="time", type_list=numeric_types, meta={"name": "bronze_time_is_numeric"}
    )
    expect_distance_is_numeric = gxe.ExpectColumnValuesToBeInTypeList(
        column="distance",
        type_list=numeric_types,
        meta={"name": "bronze_distance_is_numeric"},
    )
    expect_agency_is_string = gxe.ExpectColumnValuesToBeInTypeList(
        column="agency",
        type_list=string_types,
        meta={"name": "bronze_agency_is_string"},
    )
    # ... (add other string/object checks as needed) ...

    # --- Assemble the list of expectations ---
    expectations = [
        expect_columns_to_match_set,
        expect_table_has_rows,
        expect_travelcode_not_null,
        expect_price_is_numeric,
        expect_time_is_numeric,
        expect_distance_is_numeric,
        expect_agency_is_string,
    ]

    logger.info(
        f"Successfully built {len(expectations)} expectations for the Bronze gate."
    )
    return expectations
