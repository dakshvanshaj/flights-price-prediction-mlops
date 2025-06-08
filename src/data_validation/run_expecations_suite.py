from expectations_suite import (
    get_or_create_expectation_suite,
    expect_column_max_to_be_between,
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
    SUITE_LOGS,
)
import logging


def main():
    # Setup logger basic configuration
    setup_logger(verbose=False, log_file=SUITE_LOGS, mode="w")
    logger = logging.getLogger(__name__)

    # Initialize GE components using config values
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        GE_ROOT_DIR,
        SOURCE_NAME,
        DATA_BASE_DIR,
        ASSET_NAME,
        BATCH_NAME,
        BATCH_PATH,
    )

    # Get or create expectation suite
    suite = get_or_create_expectation_suite(context, SUITE_NAME)

    # Add expectation on 'price' column
    # More will be added here
    # we will also handle case when we may already hav expectation added
    expect_column_max_to_be_between(suite, "price", 1, 1500)

    logger.info(f"Expectation suite '{SUITE_NAME}' updated.")


if __name__ == "__main__":
    main()
