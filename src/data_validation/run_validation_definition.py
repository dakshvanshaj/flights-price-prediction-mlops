from expectations_suite import get_or_create_expectation_suite
from validation_definition import (
    get_or_create_validation_definition,
    run_validation,
)
from utils import setup_logger, initialize_ge_components
from config import (
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
    SUITE_NAME,
    VALIDATION_DEFINITION_NAME,
    BATCH_PARAMETERS,
    VAL_DEF_LOGS,
)
import logging


def main():
    # Setup logger basic configuration
    setup_logger(verbose=False, log_file=VAL_DEF_LOGS, mode="w")
    logger = logging.getLogger(__name__)

    # Load data context, data, asset, batch def and batch from utils.py
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        GE_ROOT_DIR,
        SOURCE_NAME,
        DATA_BASE_DIR,
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

    logger.info(f"Validation results: {results}")


if __name__ == "__main__":
    main()
