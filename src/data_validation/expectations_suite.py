import logging
import great_expectations as gx
import copy

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


def delete_expectation(suite, expectation):
    """
    Delete a specific expectation from the suite.

    Args:
        suite (ExpectationSuite): The Great Expectations suite.
        expectation (ExpectationConfiguration): The expectation to remove.

    Returns:
        bool: True if the expectation was removed successfully, False otherwise.
    """
    try:
        result = suite.delete_expectation(expectation)
        if result:
            logger.info(
                f"Deleted expectation: {expectation.meta.get('name') or expectation.id}"
            )
        else:
            logger.warning(
                f"Expectation not found for deletion: {expectation.meta.get('name') or expectation.id}"
            )
        return result
    except Exception as e:
        logger.error(f"Failed to delete expectation: {e}")
        return False


def upsert_expectation(suite, expectation):
    """
    Add or update an expectation in the suite based on its 'meta.name' value.

    If an expectation with the same 'meta.name' exists, it is replaced with the new one.

    Args:
        suite (ExpectationSuite): The Great Expectations suite.
        expectation (ExpectationConfiguration): The new or updated expectation.

    Returns:
        ExpectationConfiguration: The added or updated expectation.
    """
    name = expectation.meta.get("name")
    if not name:
        raise ValueError(
            "Expectation must have a 'meta' field with a 'name' key for upsert logic."
        )

    try:
        # Check for existing expectation with the same name
        for idx, existing in enumerate(suite.expectations):
            if existing.meta.get("name") == name:
                delete_expectation(suite, existing)
                updated_expectation = copy.deepcopy(expectation)
                updated_expectation.id = None
                suite.add_expectation(updated_expectation)
                logger.info(
                    f"Updated existing expectation '{name}' with new ID '{updated_expectation.id}'."
                )
                return updated_expectation

        # If not found, add as new
        new_expectation = copy.deepcopy(expectation)
        new_expectation.id = None
        suite.add_expectation(new_expectation)
        logger.info(
            f"Added new expectation '{name}' with generated ID '{new_expectation.id}'."
        )
        return new_expectation

    except Exception as e:
        logger.error(f"Error during upsert of expectation '{name}': {e}")
        raise


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
