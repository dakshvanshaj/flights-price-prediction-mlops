from data_validation.great_expectations_components import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    get_or_create_batch_definition,
    load_batch_from_definition,
)


def initialize_ge_components(
    root_dir: str,
    source_name: str,
    base_dir: str,
    asset_name: str,
    batch_name: str,
    batch_path: str,
):
    """
    Initialize and return core GE components including validation definition.

    Returns:
        tuple: (
            context,
            data_source,
            csv_asset,
            batch_definition,
            batch,
        )
    """
    context = get_ge_context(project_root_dir=root_dir)
    data_source = get_or_create_datasource(context, source_name, base_dir)
    csv_asset = get_or_create_csv_asset(data_source, asset_name)
    batch_definition = get_or_create_batch_definition(csv_asset, batch_name, batch_path)
    batch = load_batch_from_definition(batch_definition)

    return (context, data_source, csv_asset, batch_definition, batch)
