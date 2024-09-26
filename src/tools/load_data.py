import os
from pathlib import Path
import sys


import logging
from typing import Any, Tuple
import pandas as pd
import pygraphviz as pgv

def load_data(datatable_path: str, dag_path:str) -> Tuple[pd.DataFrame, Any]:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        datatable_path (str): Path to the CSV file.
        dag_path(str): Path to the dot file.

    Returns:
        pd.DataFrame: Loaded data
        DAG
    """
  
    logging.info(f"Loading data from {datatable_path}")
    df = pd.read_csv(datatable_path)
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    DAG = pgv.AGraph(dag_path, directed=True)
    DAG_str = DAG.to_string().replace(DAG.get_name(), " ")
    return df, DAG_str


