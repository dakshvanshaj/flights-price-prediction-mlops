import pandas as pd


# work in progress
def load_data(file_path):
    """
    Load a CSV file containing flights data.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the date data.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["date"])
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()


def preprocess_data(df):
    """
    Preprocess the flights data.

    Args:
        df (pd.DataFrame): The DataFrame containing the flights data.

    Returns:
        pd.DataFrame: A DataFrame with the 'date' column set as the index.
    """
    if df.empty:
        print("DataFrame is empty. Returning an empty DataFrame.")
        return df
