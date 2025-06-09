import great_expectations as gx


def price_range_expectation():
    return gx.expectations.ExpectColumnMaxToBeBetween(
        column="price",
        min_value=1,
        max_value=500,
        strict_min=False,
        strict_max=False,
        meta={"name": "price_range_expectation"},
    )
