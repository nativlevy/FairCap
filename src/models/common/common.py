"""Utility (as helper) functions for models
"""
import logging
from typing import Dict, Set
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

class PrescriptionRule:
    """
    Represents a prescription rule with associated metrics.

    Attributes:
        condition (Dict): The condition part of the rule.
        treatment (Dict): The treatment part of the rule.
        covered_indices (Set[int]): Indices of individuals covered by this rule.
        covered_protected_indices (Set[int]): Indices of protected individuals covered by this rule.
        utility (float): The utility of this rule.
        protected_utility (float): The utility of this rule for the protected group.
    """

    def __init__(self, condition: Dict, treatment: Dict, covered_indices: Set[int],
                 covered_protected_indices: Set[int], utility: float, protected_utility: float):
        self.condition = condition
        self.treatment = treatment
        self.covered_indices = covered_indices
        self.covered_protected_indices = covered_protected_indices
        self.utility = utility
        self.protected_utility = protected_utility

    def utility(self) -> float:
        return self.utility
