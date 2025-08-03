import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from gold_data_preprocessing.missing_value_imputer import SimpleImputer, MICEImputer

# --- Fixtures ---


@pytest.fixture
def imputer_sample_df() -> pd.DataFrame:
    """
    Provides a DataFrame with various missing values for imputer testing.
    - 'numeric_col_1': Missing value to be imputed by mean/median.
    - 'numeric_col_2': Missing value for MICE imputer.
    - 'category_col': Missing value to be imputed by mode.
    - 'constant_col': Missing value to be imputed by a constant.
    """
    data = {
        "numeric_col_1": [10, 20, np.nan, 40, 50],
        "numeric_col_2": [5, 15, 25, np.nan, 45],
        "category_col": ["A", "B", "A", np.nan, "B"],
        "constant_col": [100, 200, 300, 400, np.nan],
    }
    return pd.DataFrame(data)


# --- Tests for SimpleImputer ---


def test_simple_imputer_fit(imputer_sample_df: pd.DataFrame):
    """
    Tests that the SimpleImputer correctly learns the imputation values.
    """
    # ARRANGE
    strategy_dict = {
        "median": ["numeric_col_1"],
        "mode": ["category_col"],
        "constant": {"constant_col": 999},
    }
    imputer = SimpleImputer(strategy_dict=strategy_dict)

    # ACT
    imputer.fit(imputer_sample_df)

    # ASSERT
    assert imputer.imputers_["numeric_col_1"] == 30.0  # Median of [10, 20, 40, 50]
    assert imputer.imputers_["category_col"] == "A"  # Mode of ['A', 'B', 'A', 'B']
    assert imputer.imputers_["constant_col"] == 999


def test_simple_imputer_transform(imputer_sample_df: pd.DataFrame):
    """
    Tests that the SimpleImputer correctly transforms the data using fitted values.
    """
    # ARRANGE
    df = imputer_sample_df.copy()
    strategy_dict = {"median": ["numeric_col_1"], "mode": ["category_col"]}
    imputer = SimpleImputer(strategy_dict=strategy_dict)
    imputer.fit(df)

    # ACT
    result_df = imputer.transform(df)

    # ASSERT
    assert result_df["numeric_col_1"].isnull().sum() == 0
    assert result_df["category_col"].isnull().sum() == 0
    assert result_df.loc[2, "numeric_col_1"] == 30.0  # [index_label, column_label]
    assert result_df.loc[3, "category_col"] == "A"


def test_simple_imputer_save_and_load(imputer_sample_df: pd.DataFrame, tmp_path: Path):
    """
    Tests that a fitted SimpleImputer can be saved and loaded correctly.
    """
    # ARRANGE
    df = imputer_sample_df.copy()
    strategy_dict = {"mean": ["numeric_col_1"]}
    imputer = SimpleImputer(strategy_dict=strategy_dict)
    imputer.fit(df)
    save_path = tmp_path / "simple_imputer.json"

    # ACT
    imputer.save(save_path)
    loaded_imputer = SimpleImputer.load(save_path)
    result_df = loaded_imputer.transform(df)

    # ASSERT
    assert loaded_imputer.imputers_["numeric_col_1"] == 30.0  # Mean of [10, 20, 40, 50]
    assert result_df["numeric_col_1"].isnull().sum() == 0


def test_simple_imputer_transform_before_fit_raises_error():
    """
    Tests that calling transform before fit raises a RuntimeError.
    """
    # ARRANGE
    imputer = SimpleImputer(strategy_dict={"mean": ["some_col"]})
    df = pd.DataFrame({"some_col": [1, np.nan]})

    # ACT & ASSERT
    with pytest.raises(RuntimeError, match="Imputer has not been fitted yet"):
        imputer.transform(df)


# --- Tests for MICEImputer ---


def test_mice_imputer_fit_transform(imputer_sample_df: pd.DataFrame):
    """
    Tests that the MICEImputer successfully fits and imputes missing values.
    """
    # ARRANGE
    df = imputer_sample_df.copy()
    numerical_cols = ["numeric_col_1", "numeric_col_2"]
    # Use a low max_iter for faster testing
    mice_imputer = MICEImputer(
        numerical_cols=numerical_cols, max_iter=2, random_state=42
    )

    # ACT
    result_df = mice_imputer.fit_transform(df)

    # ASSERT
    # The primary check is that no NaNs remain in the specified columns.
    assert result_df[numerical_cols].isnull().sum().sum() == 0
    # Check that other columns are untouched
    assert result_df["category_col"].equals(df["category_col"])


def test_mice_imputer_save_and_load(imputer_sample_df: pd.DataFrame, tmp_path: Path):
    """
    Tests that a fitted MICEImputer can be saved and loaded correctly.
    """
    # ARRANGE
    df = imputer_sample_df.copy()
    numerical_cols = ["numeric_col_1", "numeric_col_2"]
    mice_imputer = MICEImputer(
        numerical_cols=numerical_cols, max_iter=2, random_state=42
    )
    mice_imputer.fit(df)
    save_path = tmp_path / "mice_imputer.joblib"

    # ACT
    mice_imputer.save(save_path)
    loaded_imputer = MICEImputer.load(save_path)
    result_df = loaded_imputer.transform(df)

    # ASSERT
    assert isinstance(loaded_imputer, MICEImputer)
    assert loaded_imputer._is_fitted
    assert result_df[numerical_cols].isnull().sum().sum() == 0


def test_mice_imputer_fit_with_missing_column_raises_error(
    imputer_sample_df: pd.DataFrame,
):
    """
    Tests that MICEImputer raises a ValueError if a specified column is not in the DataFrame.
    """
    # ARRANGE
    df = imputer_sample_df.copy()
    numerical_cols = ["numeric_col_1", "nonexistent_col"]
    mice_imputer = MICEImputer(numerical_cols=numerical_cols)

    # ACT & ASSERT
    with pytest.raises(
        ValueError, match="The following columns are not in the DataFrame"
    ):
        mice_imputer.fit(df)
