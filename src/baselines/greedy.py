import logging
import os
import sys
from pathlib import Path
import pandas as pd
from typing import List, Set, Dict

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
from utility_functions import expected_utility
from fairness import score_rule

from load_data import load_data
from prescription import Prescription


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
        covered.update(best_rule.covered_indices)
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
        covered.update(best_rule.covered_indices)
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


def getKRules(k_rules: int, grouping_patterns: List[dict], df: pd.DataFrame, idx_protec: Set[int], attrI: List[str], attrM: List[str], tgtO: str,
                   unprotected_coverage_threshold: float, protected_coverage_threshold: float,
                   fairness_threshold: float,
                   DAG, config: Dict) -> Dict:
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

    # Step 2. Treatment mining using greedy
    # Get treatments for each grouping pattern
    logging.info("Getting treatments for each grouping pattern")
    group_treatment_dict, _ = getTreatmentForAllGroups(DAG, df, idx_protec, grouping_patterns, {}, tgtO, attrM)
    # Create Rule objects
    rules = []  
    for group, data in group_treatment_dict.items():
        condition = eval(group)
        treatment = data['treatment']
        covered_indices = data['covered_indices']
        covered_protected_indices = covered_indices.intersection(
            idx_protec)
        utility = data['utility']

        # Calculate protected_utility based on the proportion of protected individuals covered
        protected_proportion = len(covered_protected_indices) / \
            len(covered_indices) if len(covered_indices) > 0 else 0
        protec_utility = utility * protected_proportion

        rules.append(Prescription(condition, treatment, covered_indices,
                     covered_protected_indices, utility, protec_utility))

    logging.info(f"Created {len(rules)} Rule objects")

    # Log utility statistics for all rules
    all_utilities = [rule.utility for rule in rules]
    logging.info(f"All rules utility statistics: min={min(all_utilities):.4f}, "
                 f"max={max(all_utilities):.4f}, mean={statistics.mean(all_utilities):.4f}, "
                 f"median={statistics.median(all_utilities):.4f}")

    # Run greedy algorithm
    total_individuals = len(df)
    logging.info(f"Running greedy algorithm with unprotected coverage threshold {unprotected_coverage_threshold}, "
                 f"protected coverage threshold {protected_coverage_threshold}, "
                 f"{k_rules} rules, and fairness threshold {fairness_threshold}")

    # save all rules to an output file
    with open(os.path.join(config['_output_path'], 'rules_greedy.json'), 'w+') as f:
        json.dump([{
            'condition': rule.condition,
            'treatment': rule.treatment,
            'utility': rule.utility,
            'protected_utility': rule.protected_utility,
            'coverage': len(rule.covered_indices),
            'protected_coverage': len(rule.covered_protected_indices)
        } for rule in rules], f, indent=4)

    selected_rules = greedy_fair_prescription_rules(rules, idx_protec, unprotected_coverage_threshold,protected_coverage_threshold, k_rules, total_individuals, fairness_threshold)

    # Calculate metrics
    exp_util = expected_utility(selected_rules)
    total_coverage = set().union(
        *[rule.covered_indices for rule in selected_rules])
    total_protected_coverage = set().union(
        *[rule.covered_protected_indices for rule in selected_rules])
    protec_exp_util = expected_utility(
        [Prescription(r.condition, r.treatment, r.covered_protected_indices, r.covered_protected_indices, r.protected_utility, r.protected_utility) for r in selected_rules])

    end_time = time.time()
    execution_time = end_time - start_time

    logging.info(f"Experiment results for k={k_rules}:")
    logging.info(f"Expected utility: {exp_util:.4f}")
    logging.info(
        f"Protected expected utility: {protec_exp_util:.4f}")
    logging.info(f"Coverage: {len(total_coverage) / len(df):.2%}")
    logging.info(
        f"Protected coverage: {len(total_protected_coverage) / len(idx_protec):.2%}")

    return {
        'k': k_rules,
        'execution_time': execution_time,
        'selected_rules': selected_rules,
        'expected_utility': exp_util,
        'protected_expected_utility': protec_exp_util,
        'coverage': len(total_coverage) / len(df),
        'protected_coverage': len(total_protected_coverage) / len(idx_protec)
    }


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
    valP = config.get('_protected_values')
    tgt =  config.get('_target_outcome')
    cvrg_constr = config.get('_coverage_constraint', None)

    MIX_K, MAX_K = config.get('_k', [4,4])

    # ------------------------- PARSING CONFIG ENDS  -------------------------
    # ------------------------ DATASET SETUP BEGINS -------------------------- 
    df, DAG =load_data(os.path.join(DATA_PATH, dataset_path, datatable_path), os.path.join(DATA_PATH, dataset_path, dag_path))

    # TODO Formalize DAG
    # V, E = DAG_.nodes(), list(map(lambda a: "%s -> %s" % a, DAG_.edges()))
    # DAG = []
    # for v in V:
    #     DAG.append(v)
    # for e in E:
    #     DAG.append(e) 

    # Define protected group
    protected_group = set(
        df[df[attrP] != valP].index)
    df_protec = df[(df[attrP] == valP).all(axis=1)]
    idx_protec: Set = set(df_protec.index)
    logging.info(
        f"Protected group size: {len(protected_group)} out of {len(df)} total")
    # ------------------------ DATASET SETUP ENDS ----------------------------



    start_time = time.time()
    # Step 1. Grouping pattern mining
    
    grouping_patterns = getConstrGroups(df, attrI, min_sup=APRIORI, constr=cvrg_constr)

    elapsed_time = time.time() - start_time 
    logging.warning(f"Elapsed time for group mining: {elapsed_time} seconds")
    # Run experiments for different values of k = the number of rules
    results = []
    for k in range(MIX_K, MAX_K + 1):
        # TODO make signature shorter 
        result = getKRules(k, grouping_patterns, df, idx_protec, attrI, attrM, tgt,
                                unprotected_coverage_threshold, protected_coverage_threshold,
                                fairness_threshold, DAG, config)
        results.append(result)
        logging.info(f"Completed experiment for k={k}")

    # Write results to CSV
    with open(os.path.join(config['_output_path'], 'experiment_results_greedy.csv'), 'w+', newline='') as csvfile:
        fieldnames = ['k', 'execution_time', 'expected_utility', 'protected_expected_utility', 'coverage',
                      'protected_coverage', 'selected_rules']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            # Convert selected_rules to a JSON string
            selected_rules_json = json.dumps([{
                'condition': rule.condition,
                'treatment': rule.treatment,
                'utility': rule.utility,
                'protected_utility': rule.protected_utility,
                'coverage': len(rule.covered_indices),
                'protected_coverage': len(rule.covered_protected_indices)
            } for rule in result['selected_rules']])

            writer.writerow({
                'k': result['k'],
                'execution_time': result['execution_time'],
                'expected_utility': result['expected_utility'],
                'protected_expected_utility': result['protected_expected_utility'],
                'coverage': result['coverage'],
                'protected_coverage': result['protected_coverage'],
                'selected_rules': selected_rules_json
            })

    logging.info("Results written to experiment_results_greedy.csv")

    # Log detailed results for each k
    for result in results:
        logging.info(f"\nDetailed results for k={result['k']}:")
        logging.info(f"Execution time: {result['execution_time']:.2f} seconds")
        logging.info(f"Expected utility: {result['expected_utility']:.4f}")
        logging.info(
            f"Protected expected utility: {result['protected_expected_utility']:.4f}")
        logging.info(f"Coverage: {result['coverage']:.2%}")
        logging.info(f"Protected coverage: {result['protected_coverage']:.2%}")


if __name__ == "__main__":
    main_cmd(sys.argv[1])
