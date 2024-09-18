
from typing import List

import pandas as pd


def uniqueVal(df: pd.DataFrame, attrs : List):
    """
    Get unique values for each attribute in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        attrs (list): List of attribute names.

    Returns:
        dict: A dictionary with attribute names as keys and lists of unique values as values.
        e.g.
        {
            A: ['a1', 'a3', 'a9', ...],
            B: ['b1', 'b4', 'b8', ...]
        }
    """
    cols = df.columns
    return dict((col, df[col].unique()) for col in cols)