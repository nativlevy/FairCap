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
from MutePrint import MutePrint 

"""
This module contains utility functions for causal inference and treatment effect estimation.
It provides tools for generating and evaluating treatments, calculating conditional average
treatment effects (CATE), and solving optimization problems related to set coverage.
"""

THRESHOLD = 0.1

def CATE(df_g: pd.DataFrame, DAG, treatments, attrOrdinal, tgtO):
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
    # TODO experimental: 
    # TODO question: 1 if (all attributes != treatval) or
    # TODO question: 1 if (exists attributes != treatval) 

    # df_g['TempTreatment'] = df_g.apply(
    #     lambda row: isTreatable(row, treatment, attrOrdinal), axis=1
    # Read more: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy 
    df_g = df_g.copy()
    df_g.loc[:,'TempTreatment'] = df_g[treatments.keys()] != treatments.values() 
    DAG_ = DAG_after_treatments(DAG, treatments, tgtO)
    causal_graph = DAG_.to_string()
    # remove graph name as dowhy doesn't support named graph string
    causal_graph = causal_graph.replace(DAG_.get_name(), "") 
    ## --------------------- DAG Modification ends ---------------------------
    try:   
        ATE, p_value = estimateATE(graph=causal_graph, df=df_g, T ='TempTreatment', O=tgtO)
        if p_value > THRESHOLD:
            ATE = 0
    except Exception as e:
        logging.debug(e)
        ATE = 0
        p_value = 0

    logging.debug(f"Treatment: {treatments}, ATE: {ATE}, p_value: {p_value}")

    return ATE


def estimateATE(graph, df, T, O):
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
    graph_ = graph.replace("\n", " ")
    model = CausalModel(
        data=df_filtered,
        graph=graph_,
        treatment=T,
        outcome=O)

    estimands = model.identify_effect(proceed_when_unidentifiable = False)
    with MutePrint():
        causal_estimate_reg = model.estimate_effect(estimands,
                                                    method_name="backdoor.linear_regression",
                                                    target_units="ate",
                                                    effect_modifiers=[],
                                                    test_significance=True)
    if causal_estimate_reg.value==0:
        return 0, 0
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

    # Each treatment {A:a1} = to check if A can be set to a1
    for treat_attr in treatments:
        if treat_attr in attrOrdinal:
            # In case ordinal_attr is defined
            # current value <p treatment value => treatment is not effective
            treat_rank = attrOrdinal[treat_attr].index(treatments[treat_attr])
            record_rank = attrOrdinal[treat_attr].index(record[treat_attr])
            if record_rank < treat_rank:
                return 0
        else:
            # In case ordinal_attr not defined
            # treatment value == current value => no effect on this tuple 
            if not record[treat_attr] == treatments[treat_attr]:
                return 0
    return 1


# TODO: make this more readable
def DAG_after_treatments(DAG, treats: Dict, tgtO:str):
    """
    Modify the causal graph (DAG) to incorporate the treatment variable.

    Args:
        dag (list): The original causal graph represented as a list of edges.
        randomTreatment (dict): The treatment to incorporate into the DAG.
        tgtO (str): Target outcome
    Returns:
        list: The modified causal graph with the treatment variable incorporated.
    """
    newDAG = DAG.copy()
 
    # For all attributes treat, 
    # replace edge `treat -> dep`  to `temp -> dep`
    # remove all edges `par->treat`
    nodes = set(DAG.nodes())
    treated_nodes = nodes.intersection(treats.keys())
    outEdges = DAG.out_edges(treats.keys())
    newOutEdges = set(map(lambda tup: ('TempTreatment', tup[1]), outEdges))
    # remove edges associated with treat nodes
    newDAG.remove_nodes_from(treated_nodes) 
    newDAG.add_nodes_from(treated_nodes)

    newDAG.add_edges_from(newOutEdges) # add new nodes
    if not newDAG.has_edge('TempTreatment', tgtO):
        newDAG.add_edge('TempTreatment', tgtO)
    return newDAG


