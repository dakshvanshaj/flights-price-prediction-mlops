import logging
import great_expectations as gx

logger = logging.getLogger(__name__)


def get_or_create_checkpoint(
    context,
    checkpoint_name: str,
    validation_definitions,
    actions,
    result_format={"result_format": "COMPLETE"},
):
    """
    Retrieve an existing checkpoint by name or create and register a new one.

    Args:
        context: Great Expectations DataContext.
        checkpoint_name: Name of the checkpoint.
        validation_definitions: List of ValidationDefinition objects.
        actions: List of Action objects.
        result_format: Dict specifying result format for validation results.

    Returns:
        Checkpoint object.
    """
    try:
        checkpoint = context.checkpoints.get(checkpoint_name)
        logger.info(f"Loaded existing checkpoint: {checkpoint_name}")
    except Exception as e:
        logger.warning(
            f"Checkpoint '{checkpoint_name}' not found. Creating new one. Error: {e}"
        )
        checkpoint = gx.Checkpoint(
            name=checkpoint_name,
            validation_definitions=validation_definitions,
            actions=actions,
            result_format=result_format,
        )
        context.checkpoints.add(checkpoint)
        logger.info(
            f"Created and registered new checkpoint to context: {checkpoint_name}"
        )
    return checkpoint
