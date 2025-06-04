import great_expectations as gx


# Create a ValidationDefinition object but do not add it to the context yet
def create_validation_definition(
    context, batch_definition, expectation_suite, definition_name
):
    """
    Create a ValidationDefinition object.

    Args:
        context: Great Expectations DataContext.
        batch_definition: BatchDefinition object specifying the data batch.
        expectation_suite: ExpectationSuite object to validate against.
        definition_name: Name for the validation definition.

    Returns:
        ValidationDefinition object.
    """
    try:
        validation_definition = gx.ValidationDefinition(
            data=batch_definition, suite=expectation_suite, name=definition_name
        )
        return validation_definition
    except Exception as e:
        print(f"Error creating validation definition '{definition_name}': {e}")
        raise


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


# Retrieve an existing ValidationDefinition by name from the context
def get_validation_definition(context, validation_definition_name):
    """
    Retrieve an existing ValidationDefinition by name from the context.

    Args:
        context: Great Expectations DataContext.
        validation_definition_name: Name of the validation definition.

    Returns:
        ValidationDefinition object if found, else None.
    """
    try:
        return context.validation_definitions.get(validation_definition_name)
    except KeyError:
        print(f"Validation definition '{validation_definition_name}' not found.")
        return None
    except Exception as e:
        print(
            f"Error retrieving validation definition '{validation_definition_name}': {e}"
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
        print(f"Error running validation: {e}")
        raise
