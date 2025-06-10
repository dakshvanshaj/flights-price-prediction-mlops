import logging
from utils import setup_logger, initialize_ge_components
from create_validation_definitions import validation_definition_list
from actions import action_list
from checkpoints import get_or_create_checkpoint

from config import (
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
    CHECKPOINT_NAME,
    CHECKPOINTS_LOGS,
)

logger = logging.getLogger(__name__)


def run_data_validation_checkpoint():
    """
    Initialize GE context, get or create checkpoint, run it, and log results.

    Returns:
        checkpoint_result: Result object from checkpoint.run()
    """

    # Initialize GE components using config values
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        GE_ROOT_DIR,
        SOURCE_NAME,
        DATA_BASE_DIR,
        ASSET_NAME,
        BATCH_NAME,
        BATCH_PATH,
    )

    logger.info("Great Expectations context initialized.")

    # Prepare validation definitions and actions
    validation_definitions = validation_definition_list()
    actions = action_list()

    # Get or create checkpoint
    checkpoint = get_or_create_checkpoint(
        context=context,
        checkpoint_name=CHECKPOINT_NAME,
        validation_definitions=validation_definitions,
        actions=actions,
    )
    logger.info(f"Checkpoint '{checkpoint.name}' ready to run.")

    # Run checkpoint
    try:
        result = checkpoint.run()
        logger.info(f"Checkpoint '{checkpoint.name}' run completed successfully.")
        logger.info(result)
        logger.info(f"Validation success: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running checkpoint '{checkpoint.name}': {e}")
        raise


if __name__ == "__main__":
    # Setup basic logging config
    setup_logger(verbose=True, log_file=CHECKPOINTS_LOGS, mode="w")
    run_data_validation_checkpoint()
