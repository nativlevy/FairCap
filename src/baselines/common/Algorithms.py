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


def getAllGroups(df, atts, min_support):
    """
    Generate all possible grouping patterns using Apriori algorithm.

    Args:
        df_org (pd.DataFrame): The original dataframe.
        atts (list): List of attributes to consider for grouping.
        t (float): The minimum support threshold for Apriori algorithm.

    Returns:
        list: All generated grouping patterns.
        e.g.
        [ {'Dependents': 'No', 'Gender': 'Male'},
          {'Dependents': 'No'},
          {'Gender': 'Male'}
        ]
    """
    logging.info(f"Getting rules with min_support={min_support}")
   
    def entry_with_col_name(col_name, entry):
        """Prefix an entry with it's, connected with '___'
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
    enc = OneHotEncoder(handle_unknown='ignore', feature_name_combiner=entry_with_col_name, sparse_output=False)
    enc.set_output(transform = 'pandas')
    df = df[atts] # SELECT df.atts from df
    df = enc.fit_transform(df)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    def set_to_dict(s):
        """Transport a string to to a tuple in the dictionary

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
    rules = list(map(set_to_dict, frequent_itemsets['itemsets'])) 
    
    logging.info(f"Generated {len(rules)} rules")

 
    return rules


def getGroupstreatmentsforGreedy(DAG, df, groups, ordinal_atts, targetClass, actionable_atts, print_times, protected_group):
    """
    Get treatments for each group using a greedy approach.

    Args:
        DAG (list): The causal graph represented as a list of edges.
        df (pd.DataFrame): The input dataframe.
        groups (list): List of group patterns.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        targetClass (str): The target variable name.
        actionable_atts (list): List of actionable attributes.
        print_times (bool): Whether to print execution times.
        protected_group (set): Set of indices representing the protected group.

    Returns:
        tuple: A dictionary of group treatments and the elapsed time.
        e.g.
        {"group predicate 1":   
            {'group_size': 123,
            'covered_indices': {4,5,6},
            'treatment': None,
            'utility': 0
            }
        }
    """
    start_time = time.time()

    # Create a partial function with fixed arguments
    process_group_partial = partial(process_group_greedy, df=df,
                                    targetClass=targetClass, DAG=DAG, ordinal_atts=ordinal_atts,
                                    actionable_atts=actionable_atts, protected_group=protected_group)

    # Use multiprocessing to process groups in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(process_group_partial, groups)

    # Combine results into groups_dic
    groups_dic = {str(group): result for group, result in zip(groups, results)}

    elapsed_time = time.time() - start_time

    logging.warning(f"Elapsed time step 2: {elapsed_time} seconds")

    # Log summary statistics for utilities
    utilities = [result['utility'] for result in groups_dic.values()]
    logging.info(f"Utility statistics: min={min(utilities):.4f}, max={max(utilities):.4f}, "
                 f"mean={statistics.mean(utilities):.4f}, median={statistics.median(utilities):.4f}")

    return groups_dic, elapsed_time


def process_group_greedy(group, df, targetClass, DAG, ordinal_atts, actionable_atts, protected_group):
    """
    Process a single group to find the best treatment.

    Args:
        group (dict): The group pattern.
        df (pd.DataFrame): The input dataframe.
        targetClass (str): The target variable name.
        DAG (list): The causal graph represented as a list of edges.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        actionable_atts (list): List of actionable attributes.
        protected_group (set): Set of indices representing the protected group.

    Returns:
        dict: Information about the best treatment for the group.
    """
    # Filtering tuples with grouping predicates
    df_g = df.loc[(df[group.keys()] == group.values()).all(axis=1)]
    drop_atts = list(group.keys())
    # drop_atts.append('GROUP_MEMBER')

    # covered = set(df_g['GROUP_MEMBER'].tolist())

    (t_h, cate_h) = getHighTreatments(df_g, group, targetClass,
                                      DAG, drop_atts,
                                      ordinal_atts, actionable_atts, protected_group)
    # TODO investigate the effect of removing 'covered': covered
    covered_indices = set(df_g.index)
    return {
        'group_size': len(df_g),
        'covered_indices': covered_indices,
        'treatment': t_h,
        'utility': cate_h
    }


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


def calculate_fairness_score(treatment, df_g, DAG, ordinal_atts, target, protected_group):
    """
    Calculate the fairness score for a given treatment.

    Args:
        treatment (dict): The treatment to evaluate.
        df_g (pd.DataFrame): The group-specific dataframe.
        DAG (list): The causal graph represented as a list of edges.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        target (str): The target variable name.
        protected_group (set): Set of indices representing the protected group.

    Returns:
        float: The calculated fairness score.
    """
    cate_all = Utils.getTreatmentCATE(
        df_g, DAG, treatment, ordinal_atts, target)
    protected_df = df_g[df_g.index.isin(protected_group)]
    cate_protected = Utils.getTreatmentCATE(
        protected_df, DAG, treatment, ordinal_atts, target)

    logging.debug(
        f"CATE all: {cate_all:.4f}, CATE protected: {cate_protected:.4f}")

    if cate_all == cate_protected:
        return cate_all
    return cate_all / abs(cate_all - cate_protected)


def getHighTreatments(df_g, group, target, DAG, dropAtt, ordinal_atts, actionable_atts_org, protected_group):
    """
    Find the best treatment for a given group that maximizes fairness and effectiveness.

    This function iteratively explores treatments of increasing complexity (up to 5 levels)
    to find the one that yields the highest combined score of fairness and CATE.

    Args:
        df_g (pd.DataFrame): The dataframe for the specific group.
        group (dict): The group definition.
        target (str): The target variable name.
        DAG (list): The causal graph represented as a list of edges.
        dropAtt (list): Attributes to be dropped from consideration.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        actionable_atts_org (list): Original list of actionable attributes.
        protected_group (set): Set of indices representing the protected group.

    Returns:
        tuple: A tuple containing:
            - best_treatment (dict): The treatment with the highest combined score.
            - best_cate (float): The CATE for the best treatment.

    The function logs detailed information about its progress and decisions.
    """
    logging.info(f'Starting getHighTreatments for group: {group}')
    logging.debug(f'Initial df_g shape: {df_g.shape}')

    df_g.drop(dropAtt, axis=1, inplace=True)
    actionable_atts = [a for a in actionable_atts_org if not a in dropAtt]
    df_g = df_g.loc[:, ~df_g.columns.str.contains('^Unnamed')]

    logging.debug(f'df_g shape after dropping attributes: {df_g.shape}')
    logging.debug(f'Actionable attributes: {actionable_atts}')

    max_score = float('-inf')
    best_treatment = None
    best_cate = 0
    best_protected_cate = 0

    for level in range(1, 6):  # Up to 5 treatment levels
        logging.info(f'Processing treatment level {level}')

        if level == 1:
            treatments = Utils.getLevel1treatments(
                actionable_atts, df_g, ordinal_atts)
        else:
            positive_treatments = [t for t in treatments if Utils.getTreatmentCATE(
                df_g, DAG, t, ordinal_atts, target) > 0]
            treatments = Utils.getNextLeveltreatments(
                positive_treatments, df_g, ordinal_atts, True, False, DAG, target)

        logging.info(
            f'Number of treatments at level {level}: {len(treatments)}')
        logging.debug(
            f'Sample of treatments: {treatments[:5] if len(treatments) > 5 else treatments}')

        for treatment in treatments:
            fairness_score = calculate_fairness_score(
                treatment, df_g, DAG, ordinal_atts, target, protected_group)
            cate = Utils.getTreatmentCATE(
                df_g, DAG, treatment, ordinal_atts, target)
            protected_df = df_g[df_g.index.isin(protected_group)]
            protected_cate = Utils.getTreatmentCATE(
                protected_df, DAG, treatment, ordinal_atts, target)

            # Combine fairness score, CATE, and protected CATE with more emphasis on protected CATE
            score = fairness_score * cate * (protected_cate ** 2)

            logging.debug(
                f'Treatment: {treatment}, Fairness Score: {fairness_score:.4f}, CATE: {cate:.4f}, Protected CATE: {protected_cate:.4f}, Combined Score: {score:.4f}')

            if score > max_score and cate > 0 and protected_cate > 0:
                max_score = score
                best_treatment = treatment
                best_cate = cate
                best_protected_cate = protected_cate
                logging.info(
                    f'New best treatment found at level {level}: {best_treatment}')
                logging.info(
                    f'New best score: {max_score:.4f}, CATE: {best_cate:.4f}, Protected CATE: {best_protected_cate:.4f}')

        if level > 1 and max_score <= prev_max_score:
            logging.info(
                f'Stopping at level {level} as no better treatment found')
            break

        prev_max_score = max_score

    logging.info(f'Finished processing group: {group}')
    logging.info(
        f'Final best treatment: {best_treatment}, CATE: {best_cate:.4f}, Protected CATE: {best_protected_cate:.4f}, Combined Score: {max_score:.4f}')
    logging.info('#######################################')
    # Return protected CATE instead of overall CATE
    return (best_treatment, best_protected_cate)


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
