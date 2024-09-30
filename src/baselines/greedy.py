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

from load_data import load_data
from prescription import Prescription, PrescriptionSet
import prescription

# from utility.logging_util import init_logger

sys.path.append(os.path.join(Path(__file__).parent, 'common'))
from consts import APRIORI, MIX_K, MAX_K, DATA_PATH, PROJECT_PATH, unprotected_coverage_threshold, protected_coverage_threshold, fairness_threshold  # NOQA



# TODO maybe reuse for LPsolution 
def greedy_fair_prescription_rules(rules: List[Prescription], protected_group: Set[int],
                                   unprotected_coverage_threshold: float, protected_coverage_threshold: float,
                                   max_rules: int, total_individuals: int, fairness_threshold: float) -> List[Prescription]:
    """
    Greedy algorithm to select fair prescription rules.

    Args:
        rules (List[Rule]): List of all possible rules.
        protected_group (Set[int]): Set of indices in the protected group.
        unprotected_coverage_threshold (float): Threshold for unprotected group coverage.
        protected_coverage_threshold (float): Threshold for protected group coverage.
        max_rules (int): Maximum number of rules to select.
        total_individuals (int): Total number of individuals in the dataset.
        fairness_threshold (float): Threshold for fairness constraint.

    Returns:
        List[Rule]: Selected rules that maximize utility while satisfying fairness constraints.
    """
    solution = []
    covered = set()
    covered_protected = set()
    total_utility = 0
    protected_utility = 0

    unprotected_count = total_individuals - len(protected_group)
    logging.info(f"Starting greedy algorithm with {len(rules)} rules, "
                 f"{len(protected_group)} protected individuals, "
                 f"{unprotected_count} unprotected individuals, "
                 f"protected coverage threshold {protected_coverage_threshold}, "
                 f"unprotected coverage threshold {unprotected_coverage_threshold}, "
                 f"and max {max_rules} rules")

    # Log initial utility statistics
    initial_utilities = [rule.utility for rule in rules]
    logging.info(f"Initial utility statistics: min={min(initial_utilities):.4f}, "
                 f"max={max(initial_utilities):.4f}, mean={statistics.mean(initial_utilities):.4f}, "
                 f"median={statistics.median(initial_utilities):.4f}")

    while len(solution) < max_rules:
        best_rule = None
        best_score = float('-inf')

        for rule in rules:
            if rule not in solution:
                score = score_rule(rule, solution, covered, covered_protected,
                                   protected_group,
                                   unprotected_coverage_threshold, protected_coverage_threshold)
                if score > best_score:
                    best_score = score
                    best_rule = rule

        if best_rule is None:
            logging.info("No more rules can improve the solution, stopping")
            break

        solution.append(best_rule)
        covered.update(best_rule.covered_idx)
        covered_protected.update(best_rule.covered_protected_indices)
        total_utility += best_rule.utility
        protected_utility += best_rule.protected_utility

        logging.info(f"Added rule {len(solution)}: score={best_score:.4f}, "
                     f"total_covered={len(covered)}, protected_covered={len(covered_protected)}, "
                     f"total_utility={total_utility:.4f}, protected_utility={protected_utility:.4f}")

        # Check if coverage thresholds are met
        if (len(covered) >= unprotected_coverage_threshold * total_individuals and
                len(covered_protected) >= protected_coverage_threshold * len(protected_group)):
            logging.info(
                "Coverage thresholds met, focusing on protected group")
            break

    # After meeting coverage thresholds, focus on improving utility for the protected group
    while len(solution) < max_rules:
        best_rule = None
        best_protected_utility = float('-inf')

        for rule in rules:
            if rule not in solution:
                if rule.protected_utility > best_protected_utility:
                    best_protected_utility = rule.protected_utility
                    best_rule = rule

        if best_rule is None:
            logging.info(
                "No more rules can improve protected utility, stopping")
            break

        solution.append(best_rule)
        covered.update(best_rule.covered_idx)
        covered_protected.update(best_rule.covered_protected_indices)
        total_utility += best_rule.utility
        protected_utility += best_rule.protected_utility

        logging.info(f"Added protected-utility-improving rule {len(solution)}: protected_utility={best_protected_utility:.4f}, "
                     f"total_covered={len(covered)}, protected_covered={len(covered_protected)}, "
                     f"total_utility={total_utility:.4f}, protected_utility={protected_utility:.4f}")

    # Log final utility statistics for selected rules
    final_utilities = [rule.utility for rule in solution]
    logging.info(f"Final utility statistics: min={min(final_utilities):.4f}, "
                 f"max={max(final_utilities):.4f}, mean={statistics.mean(final_utilities):.4f}, "
                 f"median={statistics.median(final_utilities):.4f}")

    return solution


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
    df_protec = df[(df[attrP] != valP)]

    idx_protec: Set = set(df_protec.index)
    logging.info(
        f"Protected group size: {len(idx_protec)} out of {len(df)} total")
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
    candidateRx, _ = getTreatmentForAllGroups(DAG_str, df, idx_protec, groupPatterns, {}, tgtO, attrM, fair_constr=fair_constr)
    exec_time2 = time.time() - start_time 
    logging.warning(f"Elapsed time for treatment mining: {exec_time2} seconds")
    # Create Rule objects
    # Run experiments for different values of k = the number of rules

    # Save all rules found so far
    with open(os.path.join(config['_output_path'], 'rules_greedy_all.json'), 'w+') as f:
        json.dump([{
            'condition': rule.condition,
            'treatment': rule.treatment,
            'utility': round(rule.utility),
            'protected_utility': round(rule.protected_utility),
            'coverage': len(rule.covered_idx),
            'protected_coverage': len(rule.covered_idx_protected)
        } for rule in candidateRx], f, indent=4)
    results = []
    for k in range(MIN_K, MAX_K + 1):
        # TODO make signature shorter 
        kRules_and_time= selectKRules(k, candidateRx, df, idx_protec, config, cvrg_constr, fair_constr, exec_time1 + exec_time2)
        results.append(kRules_and_time)
        logging.info(f"Completed experiment for k={k}")

    # Write results to CSV
    with open(os.path.join(config['_output_path'], 'experiment_results_greedy.csv'), 'w+', newline='') as csvfile:
        fieldnames = ['k', 'execution_time', 'expected_utility', 'protected_expected_utility', 'coverage_rate',
                      'protected_coverage_rate', 'selected_rules']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        k = MIN_K - 1
        writer.writeheader()
        for kRules_and_time in results:
            k += 1
            rxSet: PrescriptionSet = kRules_and_time[0]
            if len(rxSet.getRules()) == 0:
                logging.warn(f"0 rules found when k={k}")
                continue
            ttl_exec_time: float = kRules_and_time[1] 
            # Convert selected_rules to a JSON string
            prescriptions: List[Prescription] = rxSet.getRules()
            selected_rules_json = json.dumps([{
                'group': rx.getGroup(),
                'treatment': rx.getTreatment(),
                'utility': rx.getUtility(),
                'protected_utility': rx.getProtectedUtility(),
                'coverage': round(rx.getCoverage()/len(df) * 100, 2),
                'protected_coverage': round(rx.getProtectedCoverage()/len(idx_protec) *100, 2)
            } for rx in prescriptions])

            writer.writerow({
                'k': k,
                'execution_time': ttl_exec_time,
                'expected_utility': rxSet.getExpectedUtility(),    
                'protected_expected_utility': rxSet.getProtectedExpectedUtility(),
                'coverage_rate': round(rxSet.getCoverage() / len(df) * 100, 2),
                'protected_coverage_rate': round(rxSet.getProtectedCoverage() / len(idx_protec) * 100, 2),
                'selected_rules': selected_rules_json
            })

    logging.info("Results written to experiment_results_greedy.csv")
    # Log detailed results for each k
    k = MIN_K - 1
    for kRules_and_time in results:
        k += 1
        rxSet: PrescriptionSet = kRules_and_time[0]
        if len(rxSet.getRules()) == 0:
            continue
        ttl_exec_time: float = kRules_and_time[1] 
        logging.info(f"\nDetailed results for k={k}:")
        logging.info(f"Execution time: {ttl_exec_time=:.2f} seconds")
        logging.info(f"Expected utility: {rxSet.getExpectedUtility():.4f}")
        logging.info(
            f"Protected expected utility: {rxSet.getProtectedExpectedUtility():.4f}")
        logging.info(f"Coverage: {rxSet.getCoverage():.2%}")
        logging.info(f"Protected coverage: {rxSet.getProtectedCoverage():.2%}")


if __name__ == "__main__":
    main_cmd(sys.argv[1])
