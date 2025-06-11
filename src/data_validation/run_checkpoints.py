import logging
from src.data_validation.actions import action_list
from src.data_validation.checkpoints import get_or_create_checkpoint
from src.data_validation.config import CHECKPOINT_NAME
from src.data_validation.create_validation_definitions import validation_definition_list

logger = logging.getLogger(__name__)


def run_data_validation_checkpoint(context, batch_definition, expectation_suite):
    """
    Takes GE components, creates a checkpoint, runs it, and returns the result.
    """
    # Prepare validation definitions and actions
    validation_definitions = validation_definition_list(
        context, batch_definition, expectation_suite
    )
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
    logger.info("Running checkpoint...")
    result = checkpoint.run()
    logger.info(f"Checkpoint run completed. Success: {result.success}")

    return result
