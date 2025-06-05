from expectations_suite import (
    get_or_create_expectation_suite,
    expect_column_max_to_be_between,
)
from utils import initialize_ge_components
from config import (
    ROOT_DIR,
    SOURCE_NAME,
    BASE_DIR,
    ASSET_NAME,
    BATCH_NAME,
    BATCH_PATH,
    SUITE_NAME,
)


def main():
    # Initialize GE components using config values
    context, data_source, csv_asset, batch_definition, batch = initialize_ge_components(
        ROOT_DIR,
        SOURCE_NAME,
        BASE_DIR,
        ASSET_NAME,
        BATCH_NAME,
        BATCH_PATH,
    )

    # Get or create expectation suite
    suite = get_or_create_expectation_suite(context, SUITE_NAME)

    # Add expectation on 'price' column
    expect_column_max_to_be_between(suite, "price", 1, 2000)

    print(f"Expectation suite '{SUITE_NAME}' updated.")


if __name__ == "__main__":
    main()
