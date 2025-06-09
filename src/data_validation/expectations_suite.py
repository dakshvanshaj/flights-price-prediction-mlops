import logging
import great_expectations as gx

logger = logging.getLogger(__name__)


def get_or_create_expectation_suite(context, suite_name: str):
    try:
        suite = context.suites.get(name=suite_name)
        logger.info(f"Loaded existing expectation suite: {suite_name}")
    except Exception:
        suite = gx.ExpectationSuite(name=suite_name)
        context.suites.add(suite)
        logger.info(f"Created new expectation suite: {suite_name}")
    return suite


def remove_expectation(suite, expectation_id: str):
    try:
        removed = suite.remove_expectation(id=expectation_id)
        if removed:
            logger.info(f"Removed expectation with ID '{expectation_id}'.")
        else:
            logger.warning(
                f"No expectation found with ID '{expectation_id}' to remove."
            )
        return removed
    except Exception as e:
        logger.error(f"Error removing expectation with ID '{expectation_id}': {e}")
        return False


def upsert_expectation(suite, expectation):
    if not hasattr(expectation, "id") or expectation.id is None:
        raise ValueError("Expectation must have a unique 'id' for upsertion.")

    existing = next(
        (exp for exp in suite.expectations if exp.id == expectation.id), None
    )

    if existing:
        if existing.dict() == expectation.dict():
            logger.info(
                f"Expectation '{expectation.id}' already exists with same parameters. Skipping."
            )
            return existing
        else:
            suite.remove_expectation(id=expectation.id)
            logger.info(
                f"Expectation '{expectation.id}' existed but had different parameters. Replacing."
            )

    suite.add_expectation(expectation)
    logger.info(f"Added/Updated expectation with ID '{expectation.id}'")
    return expectation
