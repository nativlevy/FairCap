from heapq import nlargest
import logging
import os
import sys
from pathlib import Path
import pandas as pd
from typing import List, Set, Dict, Tuple

import time
import csv
import json
import statistics
import concurrent


SRC_PATH = Path(__file__).parent.parent 
sys.path.append(os.path.join(SRC_PATH, 'tools'))
sys.path.append(os.path.join(SRC_PATH, 'algorithms'))
sys.path.append(os.path.join(SRC_PATH, 'algorithms', 'metrics'))
from group_mining import getConstrGroups, getGroups
from treatment_mining import getTreatmentForAllGroups
from fairness import score_rule
from LP_solver import LP_solver

from load_data import load_data
from prescription import Prescription, PrescriptionSet
import prescription

# from utility.logging_util import init_logger

sys.path.append(os.path.join(Path(__file__).parent, 'common'))
from consts import APRIORI, MIX_K, MAX_K, DATA_PATH, PROJECT_PATH, unprotected_coverage_threshold, protected_coverage_threshold, fairness_threshold  # NOQA



def selectKRules(k: int, candidateRx, df: pd.DataFrame, idx_protec: Set[int], config: Dict, cvrg_constr, fair_constr, exec_time12) -> Tuple[PrescriptionSet, float]:
    """
    Run an experiment for a specific number of rules (k).

    Args:
        k (int): Number of rules to select.
        grouping_patterns (List[dict]): List of grouping pattern
        df (pd.DataFrame): The input dataframe.
        idx_protec (Set): The index of protected individuals.
        attrI (List[str]): List of immutable attributes for grouping patterns.
        attrM (List[str]): List of mutable attributes for grouping patterns.
        tgtO (str): Target outcome
        unprotected_coverage_threshold (float): Threshold for unprotected group coverage.
        protected_coverage_threshold (float): Threshold for protected group coverage.
        fairness_threshold (float): Threshold for fairness constraint.

    Returns:
        Dict: A dictionary containing the results of the experiment.
    """
    start_time = time.time()
    rules: List[Prescription] = []
    # If NEITHER group coverage and group fairness coverage exists,
    # we run greedy based on utility because all constraints have been met
    # in previous steps
    if cvrg_constr == None or 'rule' in cvrg_constr['variant']:
        rules = nlargest(k, candidateRx, key=lambda rx: rx.utility) 
    elif cvrg_constr != None and 'group' in cvrg_constr['variant']: 
        rules = nlargest(k, candidateRx, key=lambda rx: len(rx.covered_idx)+len(rx.covered_idx))

    elif fair_constr == None or 'individual' in fair_constr['variant']:
        rules = nlargest(k, candidateRx, key=lambda rx: rx.utility)
   

    #     # Calculate protected_utility based on the proportion of protected individuals covered
    #     # TODO confirm
    #     # protected_proportion = len(covered_idx_protec) / \
    #     #     len(covered_idx) if len(covered_idx) > 0 else 0
    #     # protec_utility = utility * protected_proportion
    #     protected_utility = treat_data['protected_utility']
 

    #     rules.append(Prescription(condition, treatment, covered_idx,
    #                  covered_idx_protec, utility, protected_utility))

    logging.info(f"Created {len(rules)} Rule objects")

    # Log utility statistics for all rules
    all_utilities = [rule.utility for rule in rules]
    logging.info(f"All rules utility statistics: min={min(all_utilities):.4f}, "
                 f"max={max(all_utilities):.4f}, mean={statistics.mean(all_utilities):.4f}, "
                 f"median={statistics.median(all_utilities):.4f}")



    # save all rules to an output file
    with open(os.path.join(config['_output_path'], 'rules_greedy.json'), 'w+') as f:
        json.dump([{
            'condition': rule.condition,
            'treatment': rule.treatment,
            'utility': round(rule.utility),
            'protected_utility': round(rule.protected_utility),
            'coverage': len(rule.covered_idx),
            'protected_coverage': len(rule.covered_idx_protected)
        } for rule in rules], f, indent=4)
    # TODO Implement LP

    finalizedKRules = rules
    # Calculate metrics
    rxSet = PrescriptionSet(finalizedKRules, idx_protec)

    exec_time3 = time.time() - start_time 
    logging.info(f"Experiment results for k={k}:")
    logging.info(f"Expected utility: {rxSet.expected_utility:.4f}")
    logging.info(
        f"Protected expected utility: {rxSet.protected_expected_utility:.4f}")
    logging.info(f"Coverage: {len(rxSet.covered_idx):.2%}")
    logging.info(
        f"Protected coverage: {len(rxSet.covered_idx_protected):.2%}")

    return rxSet, exec_time12 + exec_time3
 

def main_cmd(config_str):
    config = json.loads(config_str)
    os.makedirs(config['_output_path'], exist_ok=True)
    main(config)

# TODO unwind me


def main(config):
    """
    Main function to run the greedy fair prescription rules algorithm for different values of k.
    """
    # ------------------------ PARSING CONFIG BEGINS  -------------------------

    """
        attrI := Immutable/unactionable attributes
        attrM := Mutable/actionable attributes
        attrP := Protected attributes
        valP  := Values of protected attributes
        tgt   := Target outcome
    """
    dataset_path = config.get('_dataset_path')
    datatable_path = config.get('_datatable_path')
    dag_path = config.get('_dag_path')
    attrI = config.get('_immutable_attributes')
    attrM = config.get('_mutable_attributes')
    attrP = config.get('_protected_attributes')
    # TODO extend to support multiple protected value
    valP, asProtected = config.get('_protected_values')
    tgtO =  config.get('_target_outcome')
    cvrg_constr = config.get('_coverage_constraint', None)
    fair_constr = config.get('_fairness_constraint', None)

    MIN_K, MAX_K = config.get('_k', [4,4])
    # Remove protected attributes from immutable attributes
    attrI.remove(attrP) 

    # ------------------------- PARSING CONFIG ENDS  -------------------------
    # ------------------------ DATASET SETUP BEGINS -------------------------- 
    df, DAG_str =load_data(os.path.join(DATA_PATH, dataset_path, datatable_path), os.path.join(DATA_PATH, dataset_path, dag_path))
    df['TempTreatment'] = 0
    # Define protected group
    
    df_protec = None
    if asProtected:
        df_protec = df[(df[attrP] == valP)]
    else:
        df_protec = df[(df[attrP] != valP)]
    idx_protected: Set = set(df_protec.index)
    logging.info(
        f"Protected group size: {len(idx_protected)} out of {len(df)} total")
    # ------------------------ DATASET SETUP ENDS ----------------------------


    start_time = time.time()
    # Step 1. Grouping pattern mining
    
    groupPatterns = getConstrGroups(df, attrI, min_sup=APRIORI, constr=cvrg_constr)
    # TODO Testing

    exec_time1 = time.time() - start_time 
    logging.warning(f"Elapsed time for group mining: {exec_time1} seconds")


    start_time = time.time()
    # Step 2. Treatment mining using greedy
    # Get treatments for each grouping pattern
    logging.info("Step2: Getting candidate treatments for each grouping pattern")
    rxCandidates:list[Prescription] = getTreatmentForAllGroups(DAG_str, df, idx_protected, groupPatterns, {}, tgtO, attrM, fair_constr=fair_constr)
    exec_time2 = time.time() - start_time 
    logging.warning(f"Elapsed time for treatment mining: {exec_time2} seconds")
    # Save all rules found so far
    with open(os.path.join(config['_output_path'], 'rules_greedy_all.json'), 'w+') as f:
        json.dump([{
            'condition': rule.condition,
            'treatment': rule.treatment,
            'utility': rule.utility,
            'protected_utility': rule.protected_utility,
            'coverage': list(rule.covered_idx),
            'protected_coverage': list(rule.covered_idx_protected)
        } for rule in rxCandidates], f, indent=4)
    start_time = time.time() 
    rxCandidates = LP_solver(rxCandidates, set(df.index), idx_protected, cvrg_constr, fair_constr)
    exec_time3 = time.time() - start_time
    rxSet = PrescriptionSet(rxCandidates, idx_protected)
    with open(os.path.join(config['_output_path'], 'experiment_results_greedy.csv'), 'w+', newline='') as csvfile:
        fieldnames = ['k', 'execution_time', 'expected_utility', 'protected_expected_utility', 'coverage_rate',
                      'protected_coverage_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        k = len(rxCandidates)
        writer.writeheader()
        writer.writerow({
            'k': k,
            'execution_time': exec_time3,
            'expected_utility': rxSet.getExpectedUtility(),    
            'protected_expected_utility': rxSet.getProtectedExpectedUtility(),
            'coverage_rate': round(rxSet.getCoverage() / len(df) * 100, 2),
            'protected_coverage_rate': round(rxSet.getProtectedCoverage() / len(idx_protected) * 100, 2),
        })

    # Convert selected_rules to a JSON string
    with open(os.path.join(config['_output_path'], 'rules_greedy_selected.json'), 'w+') as f:
        json.dump([{
        'condition': rx.getGroup(),
        'treatment': rx.getTreatment(),
        'utility': rx.getUtility(),
        'protected_utility': rx.getProtectedUtility(),
        'coverage_rate': round(rx.getCoverage()/len(df) * 100, 2),
        'protected_coverage_rate': round(rx.getProtectedCoverage()/len(idx_protected) *100, 2)
    } for rx in rxCandidates], f, indent=4)
    logging.info("Results written to experiment_results_greedy.csv")
   


if __name__ == "__main__":
    main_cmd(sys.argv[1])
