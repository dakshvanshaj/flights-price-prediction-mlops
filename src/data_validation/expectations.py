import great_expectations as gx


def price_range_expectation():
    return gx.expectations.ExpectColumnMaxToBeBetween(
        id="column_range",
        column="price",
        min_value=1,
        max_value=1500,
        strict_min=False,
        strict_max=False,
    )


# Add more expectation builder functions here
