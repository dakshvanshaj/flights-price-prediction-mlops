import great_expectations as gx


def price_range_expectation():
    return gx.expectations.ExpectColumnMaxToBeBetween(
        column="price",
        min_value=5,
        max_value=2000,
        strict_min=True,
        strict_max=True,
        meta={"name": "price_range_expectation"},
        result_format={"result_format": "COMPLETE"},
    )
