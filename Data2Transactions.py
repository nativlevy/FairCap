import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

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
    records = []
    for i in range(0, rows):
        records.append([str(df.values[i, j]) for j in range(0, columns)])

    logging.info(f"Number of records: {len(records)}")

    te = TransactionEncoder()

    te_ary = te.fit(records).transform(records)

    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    rules = []
    for index, row in frequent_itemsets.iterrows():
        parts = set(row["itemsets"])
        temp = {}
        for part in parts:
            part = part.split("=")
            temp[part[0].strip()] = part[1].strip()
        rules.append(temp)
    
    logging.info(f"Generated {len(rules)} rules")
    return rules
