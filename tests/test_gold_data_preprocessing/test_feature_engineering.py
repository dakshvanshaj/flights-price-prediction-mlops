import pandas as pd
import pytest
from gold_data_preprocessing.feature_engineering import (
    create_cyclical_features,
    create_categorical_interaction_features,
    create_numerical_interaction_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Provides a sample DataFrame for testing.
    """
    data = {
        "month": [1, 2, 3, 4, 5],
        "day": [10, 15, 20, 25, 30],
        "from_location": ["A", "B", "C", "A", "B"],
        "to_location": ["X", "Y", "Z", "X", "Y"],
        "time": [100, 120, 150, 110, 130],
        "distance": [50, 60, 75, 55, 65],
    }
    return pd.DataFrame(data)


class TestFeatureEngineering:
    """
    Tests for the feature engineering functions.
    """

    def test_create_cyclical_features_success(self, sample_df: pd.DataFrame):
        """
        Tests the successful creation of cyclical features.
        """
        cyclical_map = {"month": 12, "day": 31}
        df_transformed = create_cyclical_features(sample_df, cyclical_map)

        assert "month_sin" in df_transformed.columns
        assert "month_cos" in df_transformed.columns
        assert "day_sin" in df_transformed.columns
        assert "day_cos" in df_transformed.columns
        assert "month" not in df_transformed.columns
        assert "day" not in df_transformed.columns
        assert len(df_transformed) == len(sample_df)

    def test_create_cyclical_features_missing_column(self, sample_df: pd.DataFrame):
        """
        Tests that a missing column is handled gracefully.
        """
        cyclical_map = {"hour": 24}
        df_transformed = create_cyclical_features(sample_df, cyclical_map)

        assert "hour_sin" not in df_transformed.columns
        assert "hour_cos" not in df_transformed.columns
        assert len(df_transformed) == len(sample_df)

    def test_create_categorical_interaction_features_success(
        self, sample_df: pd.DataFrame
    ):
        """
        Tests the successful creation of categorical interaction features.
        """
        interaction_map = {"route": ["from_location", "to_location"]}
        df_transformed = create_categorical_interaction_features(
            sample_df, interaction_map
        )

        assert "route" in df_transformed.columns
        assert df_transformed["route"][0] == "A_X"
        assert df_transformed["route"][1] == "B_Y"
        assert len(df_transformed) == len(sample_df)

    def test_create_categorical_interaction_features_missing_column(
        self, sample_df: pd.DataFrame
    ):
        """
        Tests that a missing column in the interaction map is handled gracefully.
        """
        interaction_map = {"route": ["from_location", "destination"]}
        df_transformed = create_categorical_interaction_features(
            sample_df, interaction_map
        )

        assert "route" not in df_transformed.columns
        assert len(df_transformed) == len(sample_df)

    def test_create_numerical_interaction_features_success(
        self, sample_df: pd.DataFrame
    ):
        """
        Tests the successful creation of numerical interaction features.
        """
        interaction_map = {
            "time_per_distance": ("time", "distance", "divide"),
            "time_plus_distance": ("time", "distance", "add"),
        }
        df_transformed = create_numerical_interaction_features(
            sample_df, interaction_map
        )

        assert "time_per_distance" in df_transformed.columns
        assert "time_plus_distance" in df_transformed.columns
        assert df_transformed["time_per_distance"][0] == 2.0
        assert df_transformed["time_plus_distance"][0] == 150
        assert len(df_transformed) == len(sample_df)

    def test_create_numerical_interaction_features_missing_column(
        self, sample_df: pd.DataFrame
    ):
        """
        Tests that a missing column for numerical interaction is handled gracefully.
        """
        interaction_map = {"speed": ("duration", "distance", "divide")}
        df_transformed = create_numerical_interaction_features(
            sample_df, interaction_map
        )

        assert "speed" not in df_transformed.columns
        assert len(df_transformed) == len(sample_df)

    def test_create_numerical_interaction_features_invalid_operation(
        self, sample_df: pd.DataFrame
    ):
        """
        Tests that an invalid operation is handled gracefully.
        """
        interaction_map = {"invalid_op": ("time", "distance", "power")}
        df_transformed = create_numerical_interaction_features(
            sample_df, interaction_map
        )

        assert "invalid_op" not in df_transformed.columns
        assert len(df_transformed) == len(sample_df)
