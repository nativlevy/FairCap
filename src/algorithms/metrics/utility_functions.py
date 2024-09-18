"""Anything related to utility functions/CATE:
    - CATE
    - Expected CATE
"""
import logging
import time
import pandas as pd
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

from prescription import Prescription
warnings.filterwarnings('ignore')
SRC_PATH = Path(__file__).parent.parent.parent
sys.path.append(os.path.join(SRC_PATH, 'tools'))
import MutePrint 

"""
This module contains utility functions for causal inference and treatment effect estimation.
It provides tools for generating and evaluating treatments, calculating conditional average
treatment effects (CATE), and solving optimization problems related to set coverage.
"""

THRESHOLD = 0.1

def CATE(df_g: pd.DataFrame, DAG, treatment, attrOrdinal, tgtO):
    """
    Calculate the Conditional Average Treatment Effect (CATE) for a given treatment.

    Args:
        df_g (pd.DataFrame): The input dataframe.
        DAG (list): The causal graph represented as a list of edges.
        treatment (dict): The treatment to evaluate.
        attrOrdinal (dict): Dictionary of ordinal attributes and their ordered values.
        tgtO (str): The target variable name.

    Returns:
        float: The calculated CATE value, or 0 if the calculation fails or is insignificant.
    """
    
    ## --------------------- DAG Modification begins --------------------------
    # Add a new column named TempTreatment 
    # TODO make this looks more readable
    df_g['TempTreatment'] = df_g.apply(
        lambda row: isTreatable(row, treatment, attrOrdinal), axis=1)
    DAG_ = changeDAG(DAG, treatment)
    causal_graph = """
                        digraph {
                        """
    for line in DAG_:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"
    ## --------------------- DAG Modification ends ---------------------------

    try:
        ATE, p_value = estimateATE(causal_graph, df_g, 'TempTreatment', tgtO)
        if p_value > THRESHOLD:
            ATE = 0
    except:
        ATE = 0
        p_value = 0

    logging.debug(f"Treatment: {treatment}, ATE: {ATE}, p_value: {p_value}")

    return ATE


def estimateATE(causal_graph, df, T, O):
    """
    Estimate the Average Treatment Effect (ATE) using the CausalModel from DoWhy.

    Args:
        causal_graph (str): The causal graph in DOT format.
        df (pd.DataFrame): The input dataframe.
        T (str): The name of the treatment variable.
        O (str): The name of the outcome variable.

    Returns:
        tuple: A tuple containing the estimated ATE value and its p-value.
    """
    # Filter for required records
    df_filtered = df[(df[T] == 0) | (df[T] == 1)]

    model = CausalModel(
        data=df_filtered,
        graph=causal_graph.replace("\n", " "),
        treatment=T,
        outcome=O)

    estimands = model.identify_effect()
    with MutePrint:
        causal_estimate_reg = model.estimate_effect(estimands,
                                                    method_name="backdoor.linear_regression",
                                                    target_units="ate",
                                                    effect_modifiers=[],
                                                    test_significance=True)
    return causal_estimate_reg.value, causal_estimate_reg.test_stat_significance()['p_value']


def expected_utility(rules: List[Prescription]) -> float:
    """
    Calculate the expected utility of a set of rules.

    Args:
        rules (List[Rule]): List of rules to calculate the expected utility for.

    Returns:
        float: The calculated expected utility.
    """
    # TODO double check old implementation
    cvrg = set()
    for rule in rules:
        cvrg.update(rule.covered_indices)

    if not cvrg:
        return 0.0

    total_utility = 0.0
    for t in cvrg:
        rules_covering_t = [r for r in rules if t in r.covered_indices]
        min_utility = min(r.utility for r in rules_covering_t)
        total_utility += min_utility
    return total_utility / len(cvrg)


def isTreatable(record, treatments, attrOrdinal):
    """
    Checks record is treatable using the given treatments
    Args:
        record (pd.Series): A row from the dataframe.
        treatment (dict): The treatment dictionary.
        attrOrdinal (dict): Dictionary of ordinal attributes and their ordered values.

    Returns:
        int: 1 if the row satisfies the treatment conditions, i.e. the 
        treatment is effective, 0 otherwise.
    """    

    # Each treatment {A:a1} = to setting A to a1
    for treatment_attr in treatments:
        if treatment_attr in attrOrdinal:
            # In case ordinal_attr is defined
            # current value <p treatment value => treatment is not effective
            treatment_rank = attrOrdinal[treatment_attr].index(treatments[treatment_attr])
            record_rank = attrOrdinal[treatment_attr].index(record[treatment_attr])
            if record_rank < treatment_rank:
                return 0
        else:
            # In case ordinal_attr not defined
            # treatment value == current value => no effect on this tuple 
            if not record[treatment_attr] == treatments[treatment_attr]:
                return 0
    return 1


# TODO: make this more readable
def changeDAG(dag, randomTreatment):
    """
    Modify the causal graph (DAG) to incorporate the treatment variable.

    Args:
        dag (list): The original causal graph represented as a list of edges.
        randomTreatment (dict): The treatment to incorporate into the DAG.

    Returns:
        list: The modified causal graph with the treatment variable incorporated.
    """
    DAG = copy.deepcopy(dag)
    toRomove = []
    toAdd = ['TempTreatment;']
    for a in randomTreatment:
        for c in DAG:
            if '->' in c:
                if a in c:
                    toRomove.append(c)
                    # left hand side
                    if c.find(a) == 0:
                        string = c.replace(a, "TempTreatment")
                        if not string in toAdd:
                            toAdd.append(string)
    for r in toRomove:
        if r in DAG:
            DAG.remove(r)
    for a in toAdd:
        if not a in DAG:
            DAG.append(a)

    # Ensure TempTreatment is connected to the outcome
    DAG.append('TempTreatment -> ConvertedSalary;')

    return list(set(DAG))

