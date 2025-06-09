import logging
from expectations_suite import (
    get_or_create_expectation_suite,
    upsert_expectation,
    delete_expectation,
)
from expectations import price_range_expectation
from utils import setup_logger, initialize_ge_components
from config import (
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
    SUITE_NAME,
    SUITE_LOGS,
)


def main():
    setup_logger(verbose=False, log_file=SUITE_LOGS, mode="w")
    logger = logging.getLogger(__name__)

    context, *_ = initialize_ge_components(
        GE_ROOT_DIR, SOURCE_NAME, DATA_BASE_DIR, ASSET_NAME, BATCH_NAME, BATCH_PATH
    )

    suite = get_or_create_expectation_suite(context, SUITE_NAME)

    # Add/Update expectations
    upsert_expectation(suite, price_range_expectation())

    logger.info(f"Expectation suite '{SUITE_NAME}' updated successfully.")

    return suite


if __name__ == "__main__":
    suite = main()

    # expectation = price_range_expectation()
    # delete_expectation(suite, expectation)
    # suite.delete_expectation(suite.expectations[0])
