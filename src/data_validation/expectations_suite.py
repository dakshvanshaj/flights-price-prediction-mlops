import logging
import great_expectations as gx

logger = logging.getLogger(__name__)


def get_or_create_expectation_suite(context, suite_name: str):
    """
    Retrieve an existing expectation suite by name or create a new one.

    Args:
        context: Great Expectations DataContext.
        suite_name: Name of the expectation suite.

    Returns:
        ExpectationSuite object.
    """
    try:
        suite = context.suites.get(name=suite_name)
        logger.info(f"Loaded existing expectation suite: {suite_name}")
    except Exception:
        suite = gx.ExpectationSuite(name=suite_name)
        context.suites.add(suite)
        logger.info(f"Created new expectation suite: {suite_name}")
    return suite


def expect_column_max_to_be_between(
    suite,
    column: str,
    min_value,
    max_value,
    strict_max: bool = False,
    strict_min: bool = False,
):
    """
    Add an ExpectColumnMaxToBeBetween expectation instance to the suite.

    Args:
        suite: ExpectationSuite object.
        column: Column name.
        min_value: Minimum allowed max value or parameter dict.
        max_value: Maximum allowed max value or parameter dict.
        strict_max: Whether max value must be strictly less than max_value.
        strict_min: Whether max value must be strictly greater than min_value.
    """
    expectation = gx.expectations.ExpectColumnMaxToBeBetween(
        column=column,
        min_value=min_value,
        max_value=max_value,
        strict_max=strict_max,
        strict_min=strict_min,
    )
    suite.add_expectation(expectation)
    logger.info(f"Added ExpectColumnMaxToBeBetween for column '{column}'")
    return expectation  # Return the instance for saving
