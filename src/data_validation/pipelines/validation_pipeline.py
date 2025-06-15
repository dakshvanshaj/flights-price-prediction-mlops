# src/data_validation/pipelines/validation_pipeline.py
import logging

from data_validation.utils import initialize_ge_components
from shared.utils import setup_logger
from data_validation.run_expectations_suite import create_expectation_suite
from data_validation.run_checkpoints import run_data_validation_checkpoint
from config import (
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_DIR,
    ASSET_NAME,
    BATCH_NAME,
    VALIDATION_PIPELINE_LOGS_PATH,
    PREPARED_DATA_DIR,  # Import the prepared data directory
)


def data_validation_pipeline():
    """
    A  data validation pipeline that initializes components and runs checkpoints.
    """
    # --- SETUP LOGGING ---
    # This now runs every time the pipeline is called, ensuring logs are always configured.
    setup_logger(verbose=True, log_file=VALIDATION_PIPELINE_LOGS_PATH, mode="w")

    logger = logging.getLogger(__name__)
    logger.info("Data validation pipeline started.")

    # --- Step 1: Initialize Components ONCE ---
    logger.info("Initializing Great Expectations components...")

    # Define the specific batch of data we want to validate for this pipeline run
    batch_data_path = PREPARED_DATA_DIR / "development_data.csv"

    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        GE_ROOT_DIR,
        SOURCE_NAME,
        DATA_DIR,  # The base directory for the GE datasource
        ASSET_NAME,
        BATCH_NAME,
        batch_data_path,
    )
    logger.info("Core components initialized successfully.")

    # --- Step 2: Create/Update the Expectation Suite ---
    expectation_suite = create_expectation_suite(context)
    logger.info(f"Expectation suite '{expectation_suite.name}' is ready.")

    # --- Step 3: Run the Checkpoint ---
    result = run_data_validation_checkpoint(
        context=context,
        batch_definition=batch_definition,
        expectation_suite=expectation_suite,
    )

    if result.success:
        logger.info("--- DATA VALIDATION PIPELINE: PASSED ---")
    else:
        logger.warning("--- DATA VALIDATION PIPELINE: FAILED ---")

    return result


if __name__ == "__main__":
    data_validation_pipeline()
