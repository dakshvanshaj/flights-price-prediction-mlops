# In src/data_validation/run_expectations_suite.py
import logging
from src.data_validation.expectations_suite import (
    get_or_create_expectation_suite,
    upsert_expectation,
)
from src.data_validation.expectations import (
    price_range_expectation,
)  # Import your expectation functions
from src.data_validation.config import SUITE_NAME

logger = logging.getLogger(__name__)


def create_expectation_suite(context):
    """
    Takes a GE context, creates a suite, upserts expectations, and returns it.
    """
    logger.info(f"Getting or creating expectation suite: {SUITE_NAME}")
    expectation_suite = get_or_create_expectation_suite(context, SUITE_NAME)
    logger.info(f"Expectation suite '{SUITE_NAME}' ready.")

    # --- Add all your expectations here using the upsert logic ---
    logger.info("Upserting expectations...")
    upsert_expectation(expectation_suite, price_range_expectation())
    # more can be upserted here
    return expectation_suite
