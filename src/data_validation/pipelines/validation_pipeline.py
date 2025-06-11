import logging
from src.data_validation.utils import setup_logger, initialize_ge_components
from src.data_validation.run_expectations_suite import create_expectation_suite
from src.data_validation.run_checkpoints import run_data_validation_checkpoint
from src.data_validation.config import (
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
    VALIDATION_PIPELINE_LOGS,
)


def data_validation_pipeline():
    """
    A streamlined, efficient data validation pipeline that initializes components once.
    """
    logger = logging.getLogger(__name__)

    # --- Step 1: Initialize Components ONCE ---
    logger.info("Initializing Great Expectations components...")
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        GE_ROOT_DIR,
        SOURCE_NAME,
        DATA_BASE_DIR,
        ASSET_NAME,
        BATCH_NAME,
        BATCH_PATH,
    )
    logger.info("Core components initialized successfully.")

    # --- Step 2: Create/Update the Expectation Suite ---
    # This function now accepts the context and returns the populated suite
    expectation_suite = create_expectation_suite(context)
    logger.info(f"Expectation suite '{expectation_suite.name}' is ready.")

    # --- Step 3: Run the Checkpoint ---
    # This function now accepts all the objects it needs to run the validation
    result = run_data_validation_checkpoint(
        context=context,
        batch_definition=batch_definition,
        expectation_suite=expectation_suite,
    )

    if result.success:
        logger.info("--- DATA VALIDATION PIPELINE: PASSED ---")
    else:
        logger.warning("--- DATA VALIDATION PIPELINE: FAILED ---")

    # You can return the result if this pipeline is called by another process
    return result


if __name__ == "__main__":
    # Setup logger for the entire pipeline run
    setup_logger(verbose=True, log_file=VALIDATION_PIPELINE_LOGS, mode="w")
    data_validation_pipeline()
