from expectations_suite import get_or_create_expectation_suite

from validation_definition import (
    get_or_create_validation_definition,
    run_validation,
)

from utils import initialize_ge_components
from config import (
    ROOT_DIR,
    SOURCE_NAME,
    BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
    SUITE_NAME,
    VALIDATION_DEFINITION_NAME,
    BATCH_PARAMETERS,
)


def main():
    # Load data context, data, asset, batch def and batch from utils.py
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        ROOT_DIR,
        SOURCE_NAME,
        BASE_DIR,
        ASSET_NAME,
        BATCH_NAME,
        BATCH_PATH,
    )
    # Try to get existing expectation_suite
    expectation_suite = get_or_create_expectation_suite(context, SUITE_NAME)

    # Try to get existing validation definition, create if not found
    validation_definition = get_or_create_validation_definition(
        context, batch_definition, expectation_suite, VALIDATION_DEFINITION_NAME
    )

    # Run validation with batch parameters
    results = run_validation(validation_definition, BATCH_PARAMETERS)

    # Print results
    print(results)


if __name__ == "__main__":
    main()
