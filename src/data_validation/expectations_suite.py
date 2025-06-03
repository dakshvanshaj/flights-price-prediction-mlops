import great_expectations as gx
from great_expectations import expectations as gxe

# Import your functions from load_data.py (adjust the import path as needed)
from load_data import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    load_batch,
)

root_dir = "./great_Expectations"
source_name = "flights"
base_dir = "../../data"  # go two folders above and then data
asset_name = "flights_data"
batch_name = "flights_main"
batch_path = "flights.csv"

# Load GE context
context = get_ge_context(project_root_dir=root_dir)

# Get or create data source
data_source = get_or_create_datasource(
    context, data_source_name=source_name, base_directory=base_dir
)

# Get or create CSV asset
csv_asset = get_or_create_csv_asset(data_source, asset_name=asset_name)

# Load batch
batch = load_batch(
    csv_asset,
    batch_definition_name=batch_name,
    path_to_batch_file=batch_path,
)

# Create a expectations suite
suite_name = "flights_expectations_suite"
suite = gx.ExpectationSuite(name=suite_name)

try:
    # get expecations duite from data context
    suite = context.suites.get(name="flights_expectations_suite")


except Exception:
    # add expectations suite to data context object
    suite = context.suites.add(suite)

# define a expectation
column = "price"
min_value = 1
max_value = 2000
strict_max = False
strict_min = False

range_expectation = gxe.ExpectColumnMaxToBeBetween(
    column=column,
    min_value=min_value,
    max_value=max_value,
    strict_max=strict_max,
    strict_min=strict_min,
)

# takes in an instance of an Expectation and adds it to the Expectation Suite's configuraton:
suite.add_expectation(range_expectation)
