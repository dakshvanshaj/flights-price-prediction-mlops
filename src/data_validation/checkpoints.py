import great_expectations as gx
from create_validation_definitions import validation_definition_list
from actions import action_list
from utils import setup_logger, initialize_ge_components
from config import (
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
)
import logging

# Initialize context
# Initialize GE components using config values
context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
    GE_ROOT_DIR,
    SOURCE_NAME,
    DATA_BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
)

# Get validation definitions and actions
validation_definitions = validation_definition_list()
actions = action_list()

# Define checkpoint name
checkpoint_name = "datavalidationcheckpoint"

# Create checkpoint
checkpoint = gx.Checkpoint(
    name=checkpoint_name,
    validation_definitions=validation_definitions,
    actions=actions,
    result_format={"result_format": "COMPLETE"},
)

# Add checkpoint to context
context.checkpoints.add(checkpoint)

# Retrieve checkpoint
checkpoint = context.checkpoints.get(checkpoint_name)

# Run checkpoint
result = checkpoint.run()
print(result)
