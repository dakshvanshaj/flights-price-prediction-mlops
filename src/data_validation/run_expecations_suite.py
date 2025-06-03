from load_data import (
    get_ge_context,
    get_or_create_datasource,
    get_or_create_csv_asset,
    load_batch,
)
from expectations_suite import (
    get_or_create_expectation_suite,
    expect_column_max_to_be_between,
)


def main():
    root_dir = "./great_expectations"
    source_name = "flights"
    base_dir = "../../data"
    asset_name = "flights_data"
    batch_name = "flights_main"
    batch_path = "flights.csv"
    suite_name = "flights_expectations_suite"

    context = get_ge_context(project_root_dir=root_dir)
    data_source = get_or_create_datasource(context, source_name, base_dir)
    csv_asset = get_or_create_csv_asset(data_source, asset_name)
    batch = load_batch(csv_asset, batch_name, batch_path)

    suite = get_or_create_expectation_suite(context, suite_name)

    expect_column_max_to_be_between(suite, "price", 1, 2000)


if __name__ == "__main__":
    main()
