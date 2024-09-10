import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import OneHotEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

def addColName(row, col):
    """
    Concatenate column name and value for a given row and column.

    Args:
        row (pd.Series): A row from the dataframe.
        col (str): The column name.

    Returns:
        str: A string in the format "column_name = value".
    """
    val = row[col]
    return col +" = "+str(val)

def removeHeader(df_org, name):
    """
    Remove header from the dataframe and process it for transaction analysis.

    This function creates a new column for each original column, combining the column name and its value.
    It then saves the processed dataframe to a CSV file and reloads it without headers.

    Args:
        df_org (pd.DataFrame): The original dataframe.
        name (str): The name of the output CSV file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed dataframe.
            - int: Number of rows in the processed dataframe.
            - int: Number of columns in the processed dataframe.
    """
    logging.info(f"Removing header and processing dataframe for {name}")
    df = df_org.copy(deep = True)
    columns = df.columns
    for col in columns:
        df[col] = df.apply(lambda row: addColName(row, col), axis=1)

    # Save to CSV without header and index
    df.to_csv(name, header=None, index=False)

    # Read the CSV back
    df = pd.read_csv(name, header=None)
    rows = len(df)
    columns = len(df.columns)
    logging.info(f"Size of df: {rows} rows, {columns} columns")
    return df, rows, columns

def all_equal_ivo(lst):
    """
    Check if all elements in a list are equal.

    Args:
        lst (list): The input list.

    Returns:
        bool: True if all elements are equal or the list is empty, False otherwise.
    """
    return not lst or lst.count(lst[0]) == len(lst)

def aggregateColumn(att, g, k):
    """
    Aggregate a column based on equality of values.

    Args:
        att (str): The attribute (column) name.
        g (pd.GroupBy): A pandas GroupBy object.
        k (int): Not used in the current implementation.

    Returns:
        str: The aggregated value if all values are equal, 'NotAllEqual' otherwise.
    """
    vals = g[att].tolist()
    if all_equal_ivo(vals):
        return vals[0]
    else:
        return 'NotAllEqual'

def getRules(df, rows, columns, min_support):
    """
    Generate association rules from the dataframe using the Apriori algorithm.

    This function converts the dataframe into a format suitable for the Apriori algorithm,
    applies the algorithm, and then processes the results into a list of rules.

    Args:
        df (pd.DataFrame): The input dataframe.
        rows (int): Number of rows in the dataframe.
        columns (int): Number of columns in the dataframe.
        min_support (float): The minimum support threshold for the Apriori algorithm.

    Returns:
        list: A list of dictionaries, where each dictionary represents a rule.
    """
    logging.info(f"Getting rules with min_support={min_support}")
   
    def entry_with_col_name(col_name, entry):
        """Prefix an entry with it's, combined with ' = '
        e.g:
            -------------------------
            | Age                   |
            | '18 - 24 years old'   | 
            -------------------------
            becomes 
            -----------------------------
            | Age                       |
            | 'Age_18 - 24 years old'   | 
            -----------------------------
        
        """
        return f"{col_name}_{entry}"
    enc = OneHotEncoder(handle_unknown='ignore', feature_name_combiner=entry_with_col_name, sparse_output=False)
    enc.set_output(transform = 'pandas')
    df = enc.fit_transform(df)
    df = df.reindex(sorted(df.columns), axis=1)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    def set_to_dict(s):
        """Transport a string to to a tuple in the dictionary

        Args:
            s (list[str]): list of attributes in the format:
                x_y

        Returns:
            _type_: _description_
        """
        _rule = {}
        for i in s:
            k, v = i.split('_')
            _rule[k] = v
        return _rule
    rules = list(map(set_to_dict, frequent_itemsets['itemsets'])) 

    
    logging.info(f"Generated {len(rules)} rules")
    return rules
