
from typing import List

import pandas as pd
import numpy as np
import functools

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
    return dict((col, df[col].unique()) for col in attrs)


# def conjct(predicates):
#     return functools.reduce(np.logical_and, predicates)

# def predicate(df, table):
#     return df. 
# c_1 = data.col1 == True
# c_2 = data.col2 < 64
# c_3 = data.col3 != 4