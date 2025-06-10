import logging
import great_expectations as gx
# from great_expectations.exceptions import DataContextError

logger = logging.getLogger(__name__)


def get_or_create_and_add_validation_definition(
    context, batch_definition, expectation_suite, definition_name
):
    """
    Retrieve an existing ValidationDefinition from the context by name,
    or create a new one and add it to the context if it doesn't exist.

    Args:
        context: Great Expectations DataContext.
        batch_definition: BatchDefinition object specifying the data batch.
        expectation_suite: ExpectationSuite object to validate against.
        definition_name: Name for the validation definition.

    Returns:
        ValidationDefinition object from the context.
    """
    try:
        validation_definition = context.validation_definitions.get(definition_name)
        logger.info(f"Loaded existing validation definition: {definition_name}")
        return validation_definition
    except gx.core.context.DataContextError:
        logger.info(
            f"Validation definition '{definition_name}' not found. Creating a new one."
        )
        try:
            validation_definition = gx.ValidationDefinition(
                data=batch_definition, suite=expectation_suite, name=definition_name
            )
            context.validation_definitions.add(validation_definition)
            logger.info(
                f"Validation definition '{definition_name}' created and added to context."
            )
            return validation_definition
        except Exception as e:
            logger.error(
                f"Failed to create or add validation definition '{definition_name}': {e}"
            )
            raise
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving validation definition '{definition_name}': {e}"
        )
        raise


# Run the validation definition with given batch parameters
def run_validation(validation_definition, batch_parameters):
    """
    Run the validation definition with given batch parameters.

    Args:
        validation_definition: ValidationDefinition object.
        batch_parameters: Dictionary of batch parameters for the run.

    Returns:
        ValidationResult object containing validation results.
    """
    try:
        validation_results = validation_definition.run(
            batch_parameters=batch_parameters
        )
        return validation_results
    except Exception as e:
        logger.error(f"Error running validation: {e}")
        raise
