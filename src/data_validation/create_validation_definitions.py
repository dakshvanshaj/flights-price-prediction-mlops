# In src/data_validation/create_validation_definitions.py
import logging
from src.data_validation.validation_definition import (
    get_or_create_and_add_validation_definition,
)
from src.data_validation.config import VALIDATION_DEFINITION_NAME

logger = logging.getLogger(__name__)


def validation_definition_list(context, batch_definition, expectation_suite):
    """
    Takes context, batch_definition, and suite to create a ValidationDefinition.
    """
    logger.info(
        f"Getting or creating validation definition: {VALIDATION_DEFINITION_NAME}"
    )
    # create and add validation definition to context
    validation_definition = get_or_create_and_add_validation_definition(
        context, batch_definition, expectation_suite, VALIDATION_DEFINITION_NAME
    )
    logger.info(f"Validation definition '{VALIDATION_DEFINITION_NAME}' ready.")

    # Return a list of validation definitions for the Checkpoint to run
    return [validation_definition]
