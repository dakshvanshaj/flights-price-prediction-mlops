import great_expectations as gx

context = gx.get_context(mode="file", project_root_dir="../Great_Expectations")

source_folder = "../data"
data_source_name = "flights"
asset_name = "flights_data"
batch_definition_name = "flights_main"
path_to_batch_file = "flights.csv"

# Retrieve existing data source to avoid duplicate creation
existing_datasources = context.list_datasources()
data_source = None

for ds_config in existing_datasources:
    if ds_config["name"] == data_source_name:
        print(f"Using existing data source: {data_source_name}")
        data_source = context.get_datasource(data_source_name)
        break

# Create data source if not found (first-time setup)
if data_source is None:
    print(f"Initializing new data source: {data_source_name}")
    try:
        data_source = context.data_sources.add_pandas_filesystem(
            name=data_source_name, base_directory=source_folder
        )
        print("Data source created successfully")
    except Exception as e:
        print(f"Failed to create data source: {e}")
        raise

# Configure CSV asset for the training data pipeline
try:
    file_csv_asset = data_source.add_csv_asset(name=asset_name)
    print(f"CSV asset '{asset_name}' configured")
except Exception as e:
    print(f"Asset creation failed, attempting to retrieve existing: {e}")
    try:
        file_csv_asset = data_source.get_asset(asset_name)
        print(f"Using existing asset: {asset_name}")
    except Exception as retrieval_error:
        print(f"Asset retrieval failed: {retrieval_error}")
        raise


# Create batch definition for specific training file
print(f"Loading training batch: {path_to_batch_file}")

try:
    batch_definition = file_csv_asset.add_batch_definition_path(
        name=batch_definition_name, path=path_to_batch_file
    )
    batch = batch_definition.get_batch()

    # Validate batch loaded successfully for pipeline health check
    print("Training batch loaded successfully")
    print("Sample data preview:")
    print(batch.head())

except FileNotFoundError:
    print(f"Training file not found: {path_to_batch_file}")
    print("Ensure data pipeline has generated the required training file")
    raise
except Exception as e:
    print(f"Batch loading failed: {e}")
    print("Check data source configuration and file permissions")
    raise
