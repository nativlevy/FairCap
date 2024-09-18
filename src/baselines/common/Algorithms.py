import os
from pathlib import Path
import sys
import warnings
import ast
import statistics

import time
import logging
import multiprocessing
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import apriori
import concurrent
import pandas as pd

import Utils
import Data2Transactions
warnings.filterwarnings('ignore')
PATH = "../../../data/"


logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')


def filterPatterns(df, groupingAtt, groups):
    """
    Filter and consolidate grouping patterns based on their coverage.

    Args:
        df (pd.DataFrame): The input dataframe.
        groupingAtt (str): The attribute used for grouping.
        groups (list): List of group patterns to filter.

    Returns:
        list: Filtered and consolidated list of group patterns.
    """
    groups_dic = {}
    for group in groups:
        df_g = df.loc[(df[group.keys()] == group.values()).all(axis=1)]
        covered = set(df_g[groupingAtt].tolist())
        groups_dic[str(group)] = frozenset(covered)
    from collections import defaultdict

    grouped = defaultdict(list)
    for key in groups_dic:
        grouped[groups_dic[key]].append(key)

    ans = []
    for k, v in grouped.items():
        if len(v) > 1:
            v = [ast.literal_eval(i) for i in v]
            ans.append(min(v, key=lambda x: len(x)))
        else:
            ans.append(ast.literal_eval(v[0]))
    return ans

def isGroupMember(row, group):
    """
    Check if a row belongs to a specific group.

    Args:
        row (pd.Series): A row from the dataframe.
        group (dict): The group pattern to check against.

    Returns:
        int: 1 if the row is a member of the group, 0 otherwise.
    """
    for att in group:
        column_c_type = type(row[att])
        if type(row[att]) == int:
            if not row[att] == int(group[att]):
                return 0
        elif type(row[att]) == str:
            if row[att] == group[att]:
                return 1
            else:
                return 0
        elif int(row[att]) == int(group[att]):
            return 1
        elif not row[att] == group[att]:
            return 0
    return 1

def filter_above_median(treatments_cate):
    """
    Filter treatments to keep only those with above-median positive CATE values.

    Args:
        treatments_cate (dict): Dictionary of treatments and their CATE values.

    Returns:
        dict: Filtered dictionary of treatments with above-median positive CATE values.
    """
    positive_values = [
        value for value in treatments_cate.values() if value > 0]

    if not positive_values:
        return {}

    positive_median = statistics.median(positive_values)

    filtered = {treatment: value for treatment, value in treatments_cate.items()
                if value > positive_median}

    logging.debug(f"Filtered treatments_cate: {filtered}")

    return filtered
