# Checking duplicate values
def check_duplicates(df):
    """
    Check for duplicate rows in the DataFrame and return the number of duplicates.
    """
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = duplicate_rows / len(df) * 100
    print(f"Percentage of rows involved in duplication: {duplicate_percentage:.2f}%")
    return
