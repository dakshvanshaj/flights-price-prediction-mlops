import great_expectations as gx


import great_expectations as gx
from great_expectations.exceptions import DataContextError


def get_or_create_validation_definition(
    context, batch_definition, expectation_suite, definition_name
):
    """
    Retrieve an existing ValidationDefinition by name from the context,
    or create a new one if it does not exist.

    Args:
        context: Great Expectations DataContext.
        batch_definition: BatchDefinition object specifying the data batch.
        expectation_suite: ExpectationSuite object to validate against.
        definition_name: Name for the validation definition.

    Returns:
        ValidationDefinition object.
    """
    try:
        validation_definition = context.validation_definitions.get(definition_name)
        print(f"Loaded existing validation definition: {definition_name}")
    except DataContextError:
        print(
            f"Validation definition '{definition_name}' not found. Creating a new one."
        )
        validation_definition = gx.ValidationDefinition(
            data=batch_definition, suite=expectation_suite, name=definition_name
        )
    except Exception as e:
        print(f"Error retrieving validation definition '{definition_name}': {e}")
        raise

    return validation_definition


# Add a ValidationDefinition object to the Great Expectations context
def add_validation_definition_to_context(context, validation_definition):
    """
    Add a ValidationDefinition to the context's validation definitions.

    Args:
        context: Great Expectations DataContext.
        validation_definition: ValidationDefinition object to add.

    Returns:
        The added ValidationDefinition object.
    """
    try:
        added_definition = context.validation_definitions.add(validation_definition)
        print(f"Validation definition '{validation_definition.name}' added to context.")
        return added_definition
    except Exception as e:
        print(f"Error adding validation definition '{validation_definition.name}': {e}")
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
        print(f"Error running validation: {e}")
        raise
