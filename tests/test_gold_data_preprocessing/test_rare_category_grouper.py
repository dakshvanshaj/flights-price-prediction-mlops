import pandas as pd
import pytest
from pathlib import Path

# Assuming the file is in src/gold_data_preprocessing/
from gold_data_preprocessing.rare_category_grouper import RareCategoryGrouper


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Provides a sample DataFrame for testing the RareCategoryGrouper.
    'city' has frequent and rare categories.
    'country' has only frequent categories.
    """
    data = {
        "city": ["A"] * 10 + ["B"] * 5 + ["C"] * 2 + ["D"],
        "country": ["X"] * 15 + ["Y"] * 3,
        "value": range(18),
    }
    return pd.DataFrame(data)


class TestRareCategoryGrouper:
    """
    Tests for the RareCategoryGrouper class.
    """

    def test_initialization_success(self):
        """
        Tests that the grouper initializes correctly with valid arguments.
        """
        grouper = RareCategoryGrouper(columns=["city"], threshold=0.1)
        assert grouper.columns == ["city"]
        assert grouper.threshold == 0.1
        assert not grouper._is_fitted

    def test_initialization_invalid_threshold(self):
        """
        Tests that initialization raises a ValueError for an invalid threshold.
        """
        with pytest.raises(
            ValueError, match="Threshold must be a float between 0 and 1."
        ):
            RareCategoryGrouper(columns=["city"], threshold=1.5)
        with pytest.raises(
            ValueError, match="Threshold must be a float between 0 and 1."
        ):
            RareCategoryGrouper(columns=["city"], threshold=0)

    def test_fit_identifies_frequent_categories(self, sample_df: pd.DataFrame):
        """
        Tests that the fit method correctly identifies and stores frequent categories.
        """
        grouper = RareCategoryGrouper(columns=["city", "country"], threshold=0.2)
        grouper.fit(sample_df)

        assert grouper._is_fitted
        # For 'city', A (10/18 > 0.2) and B (5/18 > 0.2) are frequent.
        assert sorted(grouper.frequent_categories_map_["city"]) == sorted(["A", "B"])
        # For 'country', only X (15/18 > 0.2) is frequent. Y (3/18 < 0.2) is not.
        assert sorted(grouper.frequent_categories_map_["country"]) == sorted(["X"])

    def test_transform_groups_rare_categories(self, sample_df: pd.DataFrame):
        """
        Tests that the transform method correctly groups rare categories into 'Other'.
        """
        grouper = RareCategoryGrouper(columns=["city"], threshold=0.2)
        grouper.fit(sample_df)
        transformed_df = grouper.transform(sample_df)

        # Expected: C (2/18) and D (1/18) are rare and should be grouped.
        expected_values = (["A"] * 10) + (["B"] * 5) + (["Other"] * 3)
        assert transformed_df["city"].tolist() == expected_values
        # Ensure other columns are untouched.
        assert "country" in transformed_df.columns
        assert sample_df["value"].equals(transformed_df["value"])

    def test_transform_handles_unseen_categories(self, sample_df: pd.DataFrame):
        """
        Tests that transform correctly handles categories not seen during fitting.
        """
        grouper = RareCategoryGrouper(columns=["city"], threshold=0.2)
        grouper.fit(sample_df)

        # Create a new DataFrame with a new category 'E'
        test_data = pd.DataFrame({"city": ["A", "C", "E"]})
        transformed_df = grouper.transform(test_data)

        # 'A' is frequent, 'C' is rare (so becomes 'Other'), 'E' is unseen (so becomes 'Other')
        assert transformed_df["city"].tolist() == ["A", "Other", "Other"]

    def test_fit_transform_produces_correct_output(self, sample_df: pd.DataFrame):
        """
        Tests the fit_transform method for correct combined functionality.
        """
        grouper = RareCategoryGrouper(columns=["city"], threshold=0.2)
        transformed_df = grouper.fit_transform(sample_df)

        expected_values = (["A"] * 10) + (["B"] * 5) + (["Other"] * 3)
        assert transformed_df["city"].tolist() == expected_values
        assert grouper._is_fitted

    def test_transform_before_fit_raises_error(self, sample_df: pd.DataFrame):
        """
        Tests that calling transform before fit raises a RuntimeError.
        """
        grouper = RareCategoryGrouper(columns=["city"])
        with pytest.raises(RuntimeError, match="Grouper has not been fitted yet."):
            grouper.transform(sample_df)

    def test_save_before_fit_raises_error(self, tmp_path: Path):
        """
        Tests that calling save before fit raises a RuntimeError.
        """
        grouper = RareCategoryGrouper(columns=["city"])
        filepath = tmp_path / "grouper.joblib"
        with pytest.raises(
            RuntimeError, match="Grouper has not been fitted. Cannot save."
        ):
            grouper.save(filepath)

    def test_save_and_load_preserves_state(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ):
        """
        Tests that saving and loading the grouper preserves its learned state.
        """
        filepath = tmp_path / "grouper.joblib"
        original_grouper = RareCategoryGrouper(columns=["city"], threshold=0.2)
        original_grouper.fit(sample_df)
        original_grouper.save(filepath)

        # Load the grouper and check its state
        loaded_grouper = RareCategoryGrouper.load(filepath)
        assert loaded_grouper._is_fitted
        assert (
            loaded_grouper.frequent_categories_map_
            == original_grouper.frequent_categories_map_
        )

        # Check that the loaded grouper transforms data correctly
        transformed_df = loaded_grouper.transform(sample_df)
        expected_values = (["A"] * 10) + (["B"] * 5) + (["Other"] * 3)
        assert transformed_df["city"].tolist() == expected_values

    def test_load_raises_file_not_found(self, tmp_path: Path):
        """
        Tests that loading from a non-existent file raises FileNotFoundError.
        """
        filepath = tmp_path / "non_existent_grouper.joblib"
        with pytest.raises(FileNotFoundError):
            RareCategoryGrouper.load(filepath)

    def test_fit_handles_missing_column_gracefully(self, sample_df: pd.DataFrame):
        """
        Tests that fit ignores columns that are not in the DataFrame without error.
        """
        # 'non_existent_col' is not in sample_df
        grouper = RareCategoryGrouper(
            columns=["city", "non_existent_col"], threshold=0.2
        )
        grouper.fit(sample_df)

        # Should have fitted for 'city' but not for the missing column
        assert "city" in grouper.frequent_categories_map_
        assert "non_existent_col" not in grouper.frequent_categories_map_
