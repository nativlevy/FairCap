import logging
from pathlib import Path
from typing import Dict, List
from xmlrpc.client import boolean
import attr
import pandas as pd
from z3 import *
import copy
import ast
from itertools import product
from itertools import chain, combinations
import random
from dowhy import CausalModel
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(Path(__file__).parent.parent.parent, 'helper'))
import MutePrint 

"""
This module contains utility functions for causal inference and treatment effect estimation.
It provides tools for generating and evaluating treatments, calculating conditional average
treatment effects (CATE), and solving optimization problems related to set coverage.
"""

THRESHOLD = 0.1


def getRandomTreatment(atts, df):
    """
    Generate a random treatment from the given attributes and dataframe.

    Args:
        atts (list): List of attribute names to consider for treatment.
        df (pd.DataFrame): The input dataframe.

    Returns:
        tuple: A tuple containing the treatment dictionary and the updated dataframe,
               or None if no valid treatment is found.
    """
    ans = {}
    k = random.randrange(1, len(atts))
    selectedAtts = random.sample(atts, k)

    for a in selectedAtts:
        val = random.choice(list(set(df[a].tolist())))
        ans[a] = val
    df['TempTreatment'] = df.apply(
        lambda row: addTempTreatment(row, ans), axis=1)
    
    logging.info(
        f"TempTreatment value counts: {df['TempTreatment'].value_counts()}")
    valid = list(set(df['TempTreatment'].tolist()))
    # no tuples in treatment group
    if len(valid) < 2:
        return None
    return ans, df


def getAllTreatments(atts, df):
    """
    Generate all possible treatments from the given attributes and dataframe.

    Args:
        atts (list): List of attribute names to consider for treatment.
        df (pd.DataFrame): The input dataframe.

    Returns:
        list: A list of all valid treatment dictionaries.
    """
    ans = []
    uniqueVal = uniqueVal(df, attrM)
    for selectedAtts in chain.from_iterable(combinations(atts, r) for r in range(len(atts)+1)):
        if len(selectedAtts) == 0:
            continue
        dict_you_want = {your_key: atts_vals[your_key]
                         for your_key in selectedAtts}
        keys, values = zip(*dict_you_want.items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        for p in permutations_dicts:
            df['TempTreatment'] = df.apply(
                lambda row: addTempTreatment(row, p), axis=1)
            valid = list(set(df['TempTreatment'].tolist()))
            # no tuples in treatment group
            if len(valid) < 2:
                continue
            ans.append(p)
    logging.info(f"Number of patterns to consider: {len(ans)}")
    return ans


def countHighLow(df, bound, att):
    """
    Count the number of high and low values in a dataframe column based on a bound.

    Args:
        df (pd.DataFrame): The input dataframe.
        bound (float): The threshold value.
        att (str): The name of the attribute (column) to count.

    Returns:
        tuple: A tuple containing the count of high values and low values.
    """
    vals = df[att].tolist()

    high = 0
    low = 0
    for v in vals:
        if v >= bound:
            high = high + 1
        else:
            low = low + 1
    return high, low



## NOT USED 
def getCates(DAG, t_h, t_l, cate_h, cate_l, df_g, ordinal_atts, target, treatments):
    """
    Calculate Conditional Average Treatment Effects (CATE) for a list of treatments.

    Args:
        DAG (list): The causal graph represented as a list of edges.
        t_h (dict): The current treatment with the highest CATE.
        t_l (dict): The current treatment with the lowest CATE.
        cate_h (float): The current highest CATE value.
        cate_l (float): The current lowest CATE value.
        df_g (pd.DataFrame): The input dataframe.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        target (str): The target variable name.
        treatments (list): List of treatments to evaluate.

    Returns:
        tuple: A tuple containing the updated treatments_cate dictionary, t_h, cate_h, t_l, and cate_l.
    """
    treatments_cate = {}
    for treatment in treatments:
        CATE = getTreatmentCATE(df_g, DAG, treatment, ordinal_atts, target)
        if CATE == 0:
            continue
        treatments_cate[str(treatment)] = CATE
        if CATE > cate_h:
            cate_h = CATE
            t_h = treatment
        if CATE < cate_l:
            cate_l = CATE
            t_l = treatment

    logging.debug(f"treatments_cate in getCates: {treatments_cate}")

    return treatments_cate, t_h, cate_h, t_l, cate_l


## NOT USED
def getCatesGreedy(DAG, t_h, cate_h, df_g, ordinal_atts, target, treatments):
    """
    Calculate Conditional Average Treatment Effects (CATE) for a list of treatments using a greedy approach.

    Args:
        DAG (list): The causal graph represented as a list of edges.
        t_h (dict): The current treatment with the highest CATE.
        cate_h (float): The current highest CATE value.
        df_g (pd.DataFrame): The input dataframe.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        target (str): The target variable name.
        treatments (list): List of treatments to evaluate.

    Returns:
        tuple: A tuple containing the updated treatments_cate dictionary, t_h, and cate_h.
    """
    treatments_cate = {}
    for treatment in treatments:
        CATE = getTreatmentCATE(df_g, DAG, treatment, ordinal_atts, target)
        if CATE == 0:
            continue
        treatments_cate[str(treatment)] = CATE
        if CATE > cate_h:
            cate_h = CATE
            t_h = treatment

    logging.debug(f"treatments_cate in getCatesGreedy: {treatments_cate}")

    return treatments_cate, t_h, cate_h



def addTempTreatment(row, ans, ordinal_atts):
    """
    Add a temporary treatment column to the dataframe based on the given treatment.

    Args:
        row (pd.Series): A row from the dataframe.
        ans (dict): The treatment dictionary.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.

    Returns:
        int: 1 if the row satisfies the treatment conditions, i.e. the 
        treatment is effective, 0 otherwise.
    """

    # Each treatment {A:a1} = to setting A to a1
    for a in ans:
        if a in ordinal_atts:
            # In case ordinal_attr is defined
            # current value <p treatment value => treatment is not effective
            index = ordinal_atts[a].index(ans[a])
            index_i = ordinal_atts[a].index(row[a])
            if index_i < index:
                return 0
        else:
            # In case ordinal_attr not defined
            # treatment value == current value => no effect on this tuple 
            if not row[a] == ans[a]:
                return 0
    return 1

# TODO add constraint flags
def LP_solver(sets, weights, tau, k, m):
    """
    Solve the Set Cover Problem using Linear Programming.

    Args:
        sets (dict): Dictionary of sets with their names as keys and elements as values.
        weights (dict): Dictionary of weights for each set.
        tau (float): The minimum fraction of elements that must be covered.
        k (int): The maximum number of sets that can be selected.
        m (int): The total number of elements.

    Returns:
        list: A list of selected set names that satisfy the constraints and maximize the objective.
    """
    solver = Optimize()

    # Create a boolean variable for each set
    set_vars = {name: Bool(name) for name in sets}

    # # Add the constraint that at most k sets can be selected
    solver.add(Sum([set_vars[name] for name in sets]) <= k)

    # Add the constraint that at least tau fraction of all elements must be covered
    elements = set.union(*[set(sets[name]) for name in sets])
    element_covered = [Bool(f"Element_{element}") for element in elements]
    for i, element in enumerate(elements):
        solver.add(Implies(element_covered[i], Or(
            [set_vars[name] for name in sets if element in sets[name]])))

    solver.add(Sum(element_covered) >= (tau * m))

    # Maximize the sum of weights
    solver.maximize(Sum([set_vars[name] * weights[name] for name in sets]))

    # Check for satisfiability and retrieve the optimal solution
    if solver.check() == sat:
        model = solver.model()
        selected_sets = [
            name for name in sets if is_true(model[set_vars[name]])]
        return selected_sets
    else:
        logging.warning("No solution was found!")
        return []
