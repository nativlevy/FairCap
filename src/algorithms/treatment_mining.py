import ast
import copy
from functools import partial
from itertools import combinations
import logging
import multiprocessing
import pygraphviz as pgv
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
import pickle
import statistics
import time
from typing import Dict, List, Set, Tuple
import os, sys
from pathlib import Path
import pandas as pd
from pygraphviz import AGraph
from fairness import benefit
from copy import deepcopy, copy
from helpers import uniqueVal
from utility_functions import CATE, isTreatable

sys.path.append(os.path.join(Path(__file__).parent, 'metrics'))
sys.path.append(os.path.join(Path(__file__).parent))
from coverage import rule_coverage


# TODO better function name
def getTreatmentForAllGroups(DAG, df, idx_protec: Set, groupPatterns, attrOrdinal, tgtO, attrM, fair_constr: Tuple[str, float] = None, cvrg_constr: Tuple[str, float] = None):
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
        fair_constr(Tuple[str, float]): fairness constraint
        cvrg_constr(Tuple[str, float]): coverage constraint

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

    # pickle df for efficiency
    pickled_df = pickle.dumps(df)
    df_nbytes = len(pickled_df)
    shm = shared_memory.SharedMemory(create=True, size=df_nbytes)
    shm.buf[:df_nbytes] = pickled_df

    # Create a partial function with fixed arguments
    partialFn = partial(getTreatmentForEachGroup, shm_name=shm.name, DAG_str=DAG.to_string(), idx_protec=idx_protec, tgtO=tgtO, attrOrdinal=attrOrdinal, attrM=attrM, fair_constr=None, cvrg_constr=None)

    # Use multiprocessing to process groups in parallel
    multiprocessing.set_start_method('fork')
    with multiprocessing.Pool(processes=os.cpu_count()-1) as pool:
        groupTreatList = pool.map(partialFn, groupPatterns)

    shm.close()


    # Combine results into groups_dic
    groups_dic = {str(group): result for group, result in zip(groupPatterns, groupTreatList)}

    elapsed_time = time.time() - start_time
    # Log summary statistics for utilities
    utilities = [result['utility'] for result in groups_dic.values()]
    logging.info(f"Utility statistics: min={min(utilities):.4f}, max={max(utilities):.4f}, "
                 f"mean={statistics.mean(utilities):.4f}, median={statistics.median(utilities):.4f}")

    return groups_dic, elapsed_time


def getTreatmentForEachGroup(group, shm_name: str, DAG_str: str, idx_protec: pd.Index, tgtO: str, attrOrdinal: Dict, attrM: List[str], fair_constr: Tuple[str, float] = None, cvrg_constr: Tuple[str, float] = None):
    
    """
    Process a single group pattern (Pg1 ^ Pg2 ^...) to find the best treatment.

    Args:
        group (dict): The group pattern.
        df (pd.DataFrame): The input dataframe.
        tgtO (str): The target variable name.
        DAG: The causal graph represented as a list of edges.
        attrOrdinal (dict): Dictionary of ordinal attributes and their ordered values.
        attrM (list): List of actionable attributes.
        idx_protec (): Set of indices representing the protected group.
        fair_constr(Tuple[str, float]): fairness constraint
        cvrg_constr(Tuple[str, float]): coverage constraint

    Returns:
        dict: Information about the best treatment for the group.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    df = pickle.loads(shm.buf)
    shm.close()
    DAG = pgv.AGraph(DAG_str, strict=True) 
    # Filtering tuples with grouping predicates
    mask = (df[group.keys()] == group.values()).all(axis=1)
    df_g = df.loc[mask]
    covered = set(mask.tolist())
    # drop grouping attributes
    df_g = df_g.drop(group.keys(), axis=1)
    logging.info(f'Starting getHighTreatments for group: {group}')
    logging.debug(f'Actionable attributes: {attrM}') 
    
    max_score = float('-inf')

    best_treatment = None
    best_cate = 0
    best_cate_protec = 0

    for level in range(1, 6):  # Up to 5 treatment levels
        logging.info(f'Processing treatment level {level}')
        treatments = []
        if level == 1:
            treatments = getRootTreatments(
                attrM, df_g, attrOrdinal)
            
        else:
            treatments = getLeafTreatments(
                treatments, df_g, attrOrdinal, True, False, DAG, tgtO)
            
        # Filter 1: weed out treats w/negative CATE
        treatmentCATEs = [CATE(
                df_g, DAG, t, attrOrdinal, tgtO) for t in treatments]
        treatments = [treatments[i] for i in range(len(treatments)) if treatmentCATEs[i] > 0]

        logging.info(
            f'Number of treatments at level {level}: {len(treatments)}')
        logging.debug(
            f'Sample of treatments: {treatments[:5] if len(treatments) > 5 else treatments}')
        # Filter 2: impose fairness constraints
        # Note: individual fairness constraint implies which treatment is to 
        # exclude whereas group fairness constraint implies which treatment is
        # preferred 
        
        for treatment in treatments:
            cate_all = CATE(
            df_g, DAG, treatment, attrOrdinal, tgtO)
            cate_protec = 0
            cate_unprotec = 0
            df_protec = df_g.loc[df_g.index.intersection(idx_protec)]
            df_unprotec = df_g.loc[df_g.index.difference(idx_protec)]
            
            if len(df_protec) != 0:
                cate_protec = CATE(df_protec, DAG, treatment, attrOrdinal, tgtO)
            if len(df_unprotec) != 0:
                cate_unprotec = CATE(df_unprotec, DAG, treatment, attrOrdinal, tgtO)

            logging.debug(f"CATE unprotected: {cate_unprotec:.4f}, CATE protected: {cate_protec:.4f}")
            treat_benefit = benefit(cate_all, cate_protec, cate_unprotec, fair_constr)
      
            # Combine fairness score, CATE, and protected CATE with more emphasis on protected CATE
            #TODO figure out what this means
            # score = fairness_score * cate * (protec_cate ** 2)
            score = treat_benefit 
            logging.debug(
                f'Treatment: {treatment}, Fairness Score: {treat_benefit:.4f}, CATE: {cate_all:.4f}, Combined Score: {score:.4f}')

            if score > max_score and cate_all > 0 and cate_protec > 0:
                max_score = score
                best_treatment = treatment
                best_cate = cate_all
                best_cate_protec = cate_protec
                logging.info(
                    f'New best treatment found at level {level}: {best_treatment}')
                logging.info(
                    f'New best score: {max_score:.4f}, CATE: {best_cate:.4f}')

        if level > 1 and max_score <= prev_max_score:
            logging.info(
                f'Stopping at level {level} as no better treatment found')
            break

        prev_max_score = max_score

    logging.info(f'Finished processing group: {group}')
    logging.info(
        f'Final best treatment: {best_treatment}, CATE: {best_cate:.4f}, Protected CATE: {best_cate_protec:.4f}, Combined Score: {max_score:.4f}')
    logging.info('#######################################')
    # Return protected CATE instead of overall CATE
    # TODO investigate the effect of removing 'covered': covered
    covered_indices = set(df_g.index)
    return {
        'group_size': len(df_g),
        'covered_indices': covered_indices,
        'covered': covered,
        'treatment': best_treatment,
        'utility': best_cate
    }



def getRootTreatments(attrM: List[Dict], df: pd.DataFrame, attrOrdinal) -> List[Dict]:
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



def getLeafTreatments(treatments: list[Dict], df_g: pd.DataFrame, ordinal_atts: Dict, high:bool, low: bool, dag, tgtO: str):
    """
    Generate next level treatments based on the current treatments and their effects.

    Args:
        treatments_cate (dict): Dictionary of treatments and their effects.
        df_g (pd.DataFrame): The input dataframe.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        high (bool): Flag to include high treatments.
        low (bool): Flag to include low treatments.
        dag (list): The causal graph represented as a list of edges.
        tgtO (str): The tgtO variable name.

    Returns:
        list: A list of next level treatment dictionaries.
    """
    logging.debug(
        f"getLeafTreatments input: treatments_cate={treatments}, high={high}, low={low}")
    treatments = []

    positives = getTreatmentsInBounds(
        treatments, 'positive', df_g, dag, ordinal_atts, tgtO)
    treatments = getCombTreatments(df_g, positives, treatments, ordinal_atts)
    logging.debug(f"getLeafTreatments output: treatments={treatments}")
    return treatments


def getCombTreatments(df_g, positives, treatments, attrOrdinal):
    """
    Generate combined treatments from positive treatments.

    Args:
        df_g (pd.DataFrame): The input dataframe.
        positives (list): List of positive treatments.
        treatments (list): List to store the combined treatments.
        attrOrdinal (dict): Dictionary of ordinal attributes and their ordered values.

    Returns:
        list: Updated list of treatments including the new combined treatments.
    """
    for comb in combinations(positives, 2):
        multi_treat = copy.deepcopy(comb[1])
        multi_treat.update(comb[0])
        if len(multi_treat.keys()) == 2:
            # TODO: experimental
            df_g_copy = df_g.copy()
            df_g_copy['TempTreatment'] = df_g_copy[treatments.keys()] != treatments.values() 

            valid = list(set(df_g_copy['TempTreatment'].tolist()))
            # no tuples in treatment group
            if len(valid) < 2:
                continue
            size = len(df_g_copy[df_g_copy['TempTreatment'] == 1])
            # treatment group is too big or too small
            if size > 0.9 * len(df_g_copy) or size < 0.1 * len(df_g_copy):
                continue
            treatments.append(multi_treat)

    return treatments



def getTreatmentsInBounds(treatments_cate, bound, df_g, DAG, ordinal_atts, tgtO):
    """
    Get treatments based on their effects and a specified bound.

    Args:
        treatments_cate (dict or list): Treatments and their effects.
        bound (str): The bound type ('positive' or 'negative').
        df_g (pd.DataFrame): The input dataframe.
        DAG (list): The causal graph represented as a list of edges.
        ordinal_atts (dict): Dictionary of ordinal attributes and their ordered values.
        target (str): The target variable name.

    Returns:
        list: A list of treatments meeting the specified criteria.
    """
    logging.debug(
        f"getTreatments input: treatments_cate={treatments_cate}, bound={bound}")
    ans = []
    if isinstance(treatments_cate, list):
        for treatment in treatments_cate:
            if bound == 'positive':
                if CATE(df_g, DAG, treatment, ordinal_atts, tgtO) > 0:
                    ans.append(treatment)
    else:
        # TODO HUH
        for k, v in treatments_cate.items():
            if bound == 'positive':
                if v > 0:
                    ans.append(ast.literal_eval(k))
    logging.debug(f"getTreatments output: ans={ans}")
    return ans