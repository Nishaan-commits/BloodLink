import pandas as pd
import os

def load_blood_donation_data(file_path=None):
    """Lo
    Loads the Blood donation dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file. If None,
    uses default path.
    
    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """

    if file_path is None:
        file_path = os.path.join("data","raw","transfusion.csv")

    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print("First few rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return None