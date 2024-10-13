from functools import partial
import logging
from typing import Dict, List, Set, Tuple
import os, sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import apriori

sys.path.append(os.path.join(Path(__file__).parent, 'metrics'))
from coverage import rule_coverage


def getGroups(df: pd.DataFrame, attrI: List[str], min_sup: float) -> List[dict]:
    """
    Generate all possible grouping patterns using Apriori algorithm.
    No constraints
    Args:
        df (pd.DataFrame): The dataframe.
        attrI (list): List of immutable attributes to consider for grouping.
        min_sup (float): The minimum support threshold for Apriori algorithm.

    Returns:
        list: All generated grouping patterns.
        e.g.
        [ {'Dependents': 'No', 'Gender': 'Male'},
          {'Dependents': 'No'},
          {'Gender': 'Male'}
        ]
    """
    df = df.copy(deep=True)
    logging.debug(f"Getting rules with min_support={min_sup}")
    enc = OneHotEncoder(handle_unknown='ignore', feature_name_combiner=entry_with_col_name, sparse_output=False)
    enc.set_output(transform = 'pandas')
    df = df[attrI] # SELECT df.attI from df
    df = enc.fit_transform(df)
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
    grouping_patterns = list(map(set_to_dict, frequent_itemsets['itemsets'])) 
    logging.debug(f"Generated {len(grouping_patterns)} grouping patterns") 
    if (len(grouping_patterns) == 0):
        raise AssertionError("No Grouping pattern found")
    return grouping_patterns

def getConstrGroups(df: pd.DataFrame, idx_p: Set[int], attrI: List[str], min_sup: float, cvrg_constr: Dict) -> List[dict]:
    """
    Same as `getConstrGroups` except that constraints are applied
    Generate all possible grouping patterns using Apriori algorithm.
    Then rule constraints or group constraints are applied.
    Args:
        df (pd.DataFrame): The dataframe.
        attrI (list): List of immutable attributes to consider for grouping.
        min_sup (float): The minimum support threshold for Apriori algorithm.

    Returns:
        list: All generated and constrained grouping patterns.
        e.g.
        [ {'Dependents': 'No', 'Gender': 'Male'},
          {'Dependents': 'No'},
          {'Gender': 'Male'}
        ]
    """
    if cvrg_constr != None and 'rule' in cvrg_constr['variant']: 
        min_sup = cvrg_constr['threshold']
 
    group_patterns = getGroups(df, attrI, min_sup)
    # Default: no constraint
    if cvrg_constr == None:
        logging.debug(f"No constraint applied on grouping patterns")
        return group_patterns
    # Group Constraint: same as default at this stage
    if 'group' in cvrg_constr['variant']: 
        logging.debug(f"Group constraint applied on grouping patterns (no group patterns are discarded at this stage)")
        return group_patterns
    
    # Rule Constraint: group should cover at least x% of rows in the dataset
    threshold = cvrg_constr['threshold'] 
    threshold_p = cvrg_constr['threshold'] 
    logging.debug(f"Initial grouping patterns: {len(group_patterns)}")
    partialFn = partial(rule_coverage, df, idx_p)
    rule_cvrg = list(map(partialFn, group_patterns))

    group_patterns_filtered = []
    for i in range(len(rule_cvrg)):
        if rule_cvrg[i][0] / len(df) > threshold and rule_cvrg[i][1] / len(idx_p) > threshold_p:
            group_patterns_filtered.append(group_patterns[i])

    # Sort filtered patterns by length (shorter first) and then by coverage size (larger first)
    logging.debug(f"Filtered grouping patterns: {len(group_patterns_filtered)}")

    # if cvrg_constr == None or 'rule' not in cvrg_constr['variant']:
    #     group_patterns_filtered.append({}) # Add a pattern that covers all
    if (len(group_patterns_filtered) == 0):
        raise AssertionError("No Grouping pattern found")
    return group_patterns_filtered

def set_to_dict(s: str):
    """Translate a string to to a tuple in the dictionary
    Args:
        s (list[str]): list of attributes in the format:
            x___y
    Returns:
        list[Dict]: dictionary of attributes in the format:
            {
                "x0": y0,
                "x1": y1,
                "x2": y2
            }
    """
    _rule = {}
    for i in s:
        k, v = i.split('___')
        _rule[k] = v
    return _rule

def entry_with_col_name(col_name, entry):
    """Prefix an entry with it's column name, connected with '___'
    e.g:
        -------------------------
        | Age                   |
        | '18 - 24 years old'   | 
        -------------------------
        becomes 
        -----------------------------
        | Age                       |
        | 'Age___18 - 24 years old' | 
        -----------------------------
    
    """
    return f"{col_name}___{entry}"