import ast
import cProfile
from functools import partial
from itertools import combinations
import logging
import math
import multiprocessing
import timeit
import pygraphviz as pgv
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, shared_memory
import pickle
import statistics
import time
from typing import Dict, List, Set, Tuple
import os, sys
from pathlib import Path
import pandas as pd
from pygraphviz import AGraph
from sympy import Q
from fairness import benefit
from copy import deepcopy, copy
from helpers import uniqueVal
from prescription import Prescription
from utility_functions import CATE
import platform

sys.path.append(os.path.join(Path(__file__).parent, 'metrics'))
sys.path.append(os.path.join(Path(__file__).parent))
from coverage import rule_coverage
    
def getTreatmentForAllGroups(DAG_str, df, idx_protec, groupPatterns, attrOrdinal, tgtO, attrM, fair_constr: Dict):
    """
    Get treatments for all group using a greedy approach.

    Args:
        DAG: The causal graph represented as a list of edges.
        df (pd.DataFrame): The input dataframe.
        idx_protec (Set): The indices of protected individual
        groupPatterns (list): List of group patterns.
        attrOrdinal (dict): Dictionary of ordinal attributes and their ordered values.
        tgtO (str): The target variable name.
        attrM (list): List of mutable/actionable attributes.

    Returns:
        tuple: A dictionary of group treatments and the elapsed time.
        e.g.
        {"group predicate 1":   
            {'group_size': 123,
            'covered_indices': {4,5,6},
            'treatment': {...},
            'utility': 0
            },
         "group predicate 2":   
            {'group_size': 100,
            'covered_indices': {7,8,9},
            'treatment':  {...},
            'utility': 0
            }
        },
        6.123 (seconds)
    """
    start_time = time.time()
    # Shared data for all forked process
    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = df
    ns.DAG_str = DAG_str
    ns.idx_protec = idx_protec
    ns.attrOrdinal = attrOrdinal
    ns.tgtO = tgtO
    ns.attrM = attrM
    ns.fair_constr = fair_constr
    
    # Use multiprocessing to process groups in parallel
    numProc=os.cpu_count()-1
    if 'i386' in platform.platform():
       # If running on old mac, use single core.
       numProc = 1
    multiprocessing.set_start_method('fork', force="True")
    with multiprocessing.Pool(processes=numProc) as pool:
        candidateRx = pool.map(partial(getTreatmentForEachGroup, ns), \
                                  groupPatterns)
    # # Combine results into groups_dic
    # group_treat_dic = {str(group): result for group, result in zip(groupPatterns, groupTreatList)}

    # Log summary statistics for utilities
    utilities = [rx.utility for rx in candidateRx]
    logging.info(f"Utility statistics: min={min(utilities):.4f}, max={max(utilities):.4f}, "
                 f"mean={statistics.mean(utilities):.4f}, median={statistics.median(utilities):.4f}")
    elapsed_time = time.time() - start_time
    return candidateRx, elapsed_time


def getTreatmentForEachGroup(ns, group):
    
    """
    Process a single group pattern (Pg1 ^ Pg2 ^...) to find the best treatment.
    Step 1. get all single treament
    Step 2. For level 1-5
        Get all possible combinations of treatment at the level,
        Traverse through every combinations,
        Discard the ones that:
            1. Has less treatments than the level number
            2. Doesn't yield positive CATE
            3. Doesn't meet individual fairness (if any)
        while traversing, mark the treatment with highest benefit
        # TODO try retain top 1 in each metric
        # Best cate
        # Best protec cate
        # Best unprotec cate 
        return a List of nodes
    Args: 

    Returns:
        dict: Information about the best treatment for the group.
    """
    
    df = ns.df.copy()
    DAG_str = ns.DAG_str
    idx_protec = ns.idx_protec
    attrM = ns.attrM
    tgtO = ns.tgtO
    attrOrdinal = ns.attrOrdinal
    fair_constr = ns.fair_constr
 
    # Use grouping predicates to get subpopulation
    mask = (df[group.keys()] == group.values()).all(axis=1)
    df_g = df.loc[mask]
    covered = set(mask.tolist())
    # drop grouping attributes
    df_g = df_g.drop(group.keys(), axis=1)
    logging.info(f'Starting getHighTreatments for group: {group}')
    logging.debug(f'Actionable attributes: {attrM}') 
    
    best_benefit = float('-inf')
    best_treatment = None
    best_cate = 0
    best_cate_protec = 0
    
    candidateTreatments = getSingleTreatments(attrM, df_g, attrOrdinal)
    prev_best_benefit = 0
    for level in range(1, 6):  # Up to 5 treatment levels
        start = time.time()
        # get all combinations
        allCombination = list(combinations(candidateTreatments, 2))
        # get map each combination into a merged treatment
        candidateTreatments = list(map(lambda c: {**c[0], **c[1]}, allCombination))
        # Filter 1: discard combined treatments that treat too few or too many
        candidateTreatments = filter(partial(isValidTreatment, df_g, level), candidateTreatments)            
        logging.debug(f"Combine treatments={candidateTreatments} at level={level}")

        selectedTreatments = []
        for treatment in candidateTreatments:
            # Filter 2: discard treatments w/negative CATE
            cate_all = CATE(
            df_g, DAG_str, treatment, attrOrdinal, tgtO) 
            if cate_all <= 0:
                continue
            
            # Filter 3: impose fairness constraints
            cate_protec = 0
            cate_unprotec = 0
            if fair_constr != None:
                threshold = fair_constr['threshold'] 
                # For SP constraints, discard treatments if the absolute
                #   difference between protected and unprotected CATE 
                #   exceeds some threshold epsilon
                if fair_constr['variant'] == 'individual_sp':
                    if abs(cate_protec-cate_unprotec) > threshold:
                        continue  

                # For BGL constraints, we discard treatments if 
                #   protected CATE does not meet some threshold epsilon 
                if fair_constr['variant'] == 'individual_bgl':
                    if cate_protec < threshold:
                        continue
            # Passing all requirements, save the node in the lattice
            selectedTreatments.append(treatment)
            df_protec = df_g.loc[df_g.index.intersection(idx_protec)]
            cate_protec = CATE(df_protec, DAG_str, treatment, attrOrdinal, tgtO)
            candidate_benefit = benefit(cate_all, cate_protec, cate_unprotec, fair_constr)
                
            if candidate_benefit > best_benefit and cate_all > 0 and cate_protec >= 0:
                best_benefit = candidate_benefit
                best_treatment = treatment
                best_cate = cate_all
                best_cate_protec = cate_protec
                logging.info(
                    f'New best treatment found at level {level}: {best_treatment}')
                logging.info(
                    f'New best score: {best_benefit:.4f}, CATE: {best_cate:.4f}')

        if level > 1 and best_benefit <= prev_best_benefit:
            logging.info(
                f'Stopping at level {level} as no better treatment found')
            break
        candidateTreatments = selectedTreatments
        prev_best_benefit = best_benefit

    logging.info(f'Finished processing group: {group}')
    logging.info(
        f'Final best treatment: {best_treatment}, CATE: {best_cate:.4f}, Protected CATE: {best_cate_protec:.4f}, Combined Score: {best_benefit:.4f}')
    logging.info('#######################################')
    covered_idx = set(df_g.index)
    covered_idx_protected = set(idx_protec) & covered_idx 
    return Prescription(group, treatment=best_treatment, covered_idx=covered_idx, covered_idx_protected=covered_idx_protected, utility=best_cate, protected_utility=best_cate_protec)

def isValidTreatment(df_g, level, newTreatment):
    """ 
        A helper function for filtering new combine treatment
        Ensure that:
        1. Number of treatment predicates > level
        2. |Treatment group| < 90% of the subpopulation
        3. |Treatment group| > 10% of the subpopulation
    """
    if len(newTreatment.keys()) >= level:
        treatable = (df_g[newTreatment.keys()] != newTreatment.values()).all(axis=1)
        valid = list(set(treatable.tolist()))
        # no tuples in treatment group
        if len(valid) < 2:
            return False
        size = len(df_g[treatable == 1])
        # treatment group is too big or too small
        if size > 0.9 * len(df_g) or size < 0.1 * len(df_g):
            return False
        return True

  


def getSingleTreatments(attrM: List[Dict], df: pd.DataFrame, attrOrdinal=None) -> List[Dict]:
    """
    Generate level 1 treatments (single attribute-value pairs).

    Args:
        attrM (list): List of mutable attribute names.
        df (pd.DataFrame): The input dataframe.
        attrOrdinal (dict): Dictionary of ordinal attributes and their ordered values.

    Returns:
        list: A list of level 1 treatment dictionaries.
    """
    ans = []
    uniqueVals = uniqueVal(df, attrM)
    # All possible values of all mutable attribute
    count = 0
    # for all possible value of each attribute 
    for attr in uniqueVals:
        for val in uniqueVals[attr]:
            treatment = {attr: val} 
            treatable = df[attr] != val
            valid = list(set(treatable.tolist()))
            # Skip treatment where either all or none will be treated
            if len(valid) < 2:
                continue
            size = len(df[treatable == 1])
            count = count+1
            # treatment group is too big or too small
            if size > 0.9*len(df) or size < 0.1*len(df):
                logging.debug(
                    f"Treatment group {treatment} is too big or too small: {size} out of total {len(df)}")
                continue
            ans.append(treatment)
    return ans


