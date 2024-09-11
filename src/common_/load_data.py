"""Utility (as helper) functions for models
"""
import logging
logging.basicConfig(level=logging.DEBUG)
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df