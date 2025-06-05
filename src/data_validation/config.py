# configuration for variable parameters

# Great Expectations project root directory
ROOT_DIR = "./great_expectations"

# Data source and asset names
SOURCE_NAME = "flights"
ASSET_NAME = "flights_data"

# Data directory and batch info
BASE_DIR = "../../data"
BATCH_NAME = "flights_main"
BATCH_PATH = "flights.csv"

# Expectation suite and validation definition names
SUITE_NAME = "flights_expectations_suite"
VALIDATION_DEFINITION_NAME = "flights_validation_definition"

# Batch parameters for validation run
BATCH_PARAMETERS = {"path": BATCH_PATH}
