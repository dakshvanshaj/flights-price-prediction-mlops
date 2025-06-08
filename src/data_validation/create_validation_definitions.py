from expectations_suite import get_or_create_expectation_suite
from validation_definition import get_or_create_validation_definition
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
    VAL_DEF_LOGS,
)
import logging


def validation_definition_list():
    # Setup logger basic configuration
    setup_logger(verbose=False, log_file=VAL_DEF_LOGS, mode="w")
    logger = logging.getLogger(__name__)

    logger.info("Initializing Great Expectations components...")
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        GE_ROOT_DIR,
        SOURCE_NAME,
        DATA_BASE_DIR,
        ASSET_NAME,
        BATCH_NAME,
        BATCH_PATH,
    )
    logger.info("Great Expectations components initialized.")

    logger.info(f"Getting or creating expectation suite: {SUITE_NAME}")
    expectation_suite = get_or_create_expectation_suite(context, SUITE_NAME)
    logger.info(f"Expectation suite '{SUITE_NAME}' ready.")

    logger.info(
        f"Getting or creating validation definition: {VALIDATION_DEFINITION_NAME}"
    )
    validation_definition = get_or_create_validation_definition(
        context, batch_definition, expectation_suite, VALIDATION_DEFINITION_NAME
    )
    logger.info(f"Validation definition '{VALIDATION_DEFINITION_NAME}' ready.")

    # Return a list of validation definitions for the Checkpoint to run
    return [validation_definition]


if __name__ == "__main__":
    validation_definition_list()
