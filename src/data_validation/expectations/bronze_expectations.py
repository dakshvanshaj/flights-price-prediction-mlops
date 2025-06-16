# src/data_validation/expectations/bronze_expectations.py
import logging
from typing import List
from great_expectations.core import ExpectationConfiguration

# Create a logger object for this module
logger = logging.getLogger(__name__)


def build_bronze_expectations() -> List[ExpectationConfiguration]:
    """
    Defines the set of expectations for the Bronze data validation gate.
    These checks ensure the raw data has the correct structure and schema.

    Returns:
        A list of ExpectationConfiguration objects.
    """
    logger.info("Building expectations for the Bronze data quality gate...")

    expectations = []

    # --- Expectation 1: Column Existence ---
    # This check ensures that all expected columns are present, regardless of order.
    # It is more flexible than checking for a strict order.
    expected_columns = {
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
    }
    expectations.append(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_set",
            kwargs={"column_set": list(expected_columns)},
            meta={"name": "bronze_columns_match_set"},
        )
    )

    # --- Expectation 2: Table Row Count ---
    expectations.append(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 1},
            meta={"name": "bronze_table_has_rows"},
        )
    )

    # --- Expectation 3: No Null Primary Keys ---
    expectations.append(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "travelCode"},
            meta={"name": "bronze_travelCode_not_null"},
        )
    )

    # --- Expectation 4: Basic Column Type Checks ---
    numeric_types = ["int", "int64", "float", "float64"]
    string_types = ["object", "str"]  # Pandas uses 'object' for strings from CSVs

    # Check numeric columns
    for col in ["price", "time", "distance"]:
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_type_list",
                kwargs={"column": col, "type_list": numeric_types},
                meta={"name": f"bronze_{col}_is_numeric"},
            )
        )

    # Check string/categorical columns, including the raw date column
    for col in ["from", "to", "flightType", "agency", "date"]:
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_type_list",
                kwargs={"column": col, "type_list": string_types},
                meta={"name": f"bronze_{col}_is_string"},
            )
        )

    logger.info(
        f"Successfully built {len(expectations)} expectations for the Bronze gate."
    )
    return expectations
