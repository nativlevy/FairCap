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
from common import PrescriptionRule, load_data
# from utility.logging_util import init_logger

sys.path.append(os.path.join(Path(__file__).parent, 'common'))
from consts import APRIORI, MIX_K, MAX_K, DATA_PATH, PROJECT_PATH, unprotected_coverage_threshold, protected_coverage_threshold, fairness_threshold  # NOQA
from Algorithms import getAllGroups, getGroupstreatmentsforGreedy  # NOQA

# logger = init_logger('greedy')




def get_grouping_patterns(df: pd.DataFrame, grp_attrs: List[str], apriori_th: float) -> List[dict]:
    """
    Generate and filter grouping patterns from the data.

    Args:
        df (pd.DataFrame): Input data.
        grp_attrs (List[str]): Attributes to consider for grouping.
        apriori_th (float): Apriori threshold for pattern generation.

    Returns:
        List[dict]: Filtered list of grouping patterns.
    """
    logging.info(f"Getting grouping patterns with apriori={apriori_th}")
    grouping_patterns = getAllGroups(df, grp_attrs, apriori_th)
    logging.info(f"Initial grouping patterns: {len(grouping_patterns)}")

    def apply_pattern(pattern):
        mask = pd.Series(True, index=df.index)
        for col, val in pattern.items():
            mask &= df[col] == val
        return frozenset(df.index[mask])

    # Create a dictionary to store patterns by their coverage
    coverage_dict = {}
    for pattern in grouping_patterns:
        coverage = apply_pattern(pattern)
        if coverage in coverage_dict:
            if len(pattern) < len(coverage_dict[coverage]):
                coverage_dict[coverage] = pattern
        else:
            coverage_dict[coverage] = pattern

    filtered_patterns = list(coverage_dict.values())

    # Sort filtered patterns by length (shorter first) and then by coverage size (larger first)
    filtered_patterns.sort(key=lambda x: (len(x), -len(apply_pattern(x))))

    logging.info(f"Filtered grouping patterns: {len(filtered_patterns)}")
    logging.info("Final filtered patterns:")
    for i, pattern in enumerate(filtered_patterns):
        covered_indices = apply_pattern(pattern)
        logging.info(f"Pattern {i}: {pattern}")
        logging.info(f"  Length (key-value pairs): {len(pattern)}")
        logging.info(f"  Coverage: {len(covered_indices)}")

    return filtered_patterns

def calculate_expected_utility(rules: List[PrescriptionRule]) -> float:
    """
    Calculate the expected utility of a set of rules.

    Args:
        rules (List[Rule]): List of rules to calculate the expected utility for.

    Returns:
        float: The calculated expected utility.
    """
    coverage = set()
    for rule in rules:
        coverage.update(rule.covered_indices)

    if not coverage:
        return 0.0

    total_utility = 0.0
    for t in coverage:
        rules_covering_t = [r for r in rules if t in r.covered_indices]
        min_utility = min(r.utility for r in rules_covering_t)
        total_utility += min_utility

    expected_utility = total_utility / len(coverage)
    return expected_utility


def score_rule(rule: PrescriptionRule, solution: List[PrescriptionRule], covered: Set[int], covered_protected: Set[int],
               protected_group: Set[int],
               unprotected_coverage_threshold: float, protected_coverage_threshold: float) -> float:
    """
    Calculate the score for a given rule based on various factors.

    Args:
        rule (Rule): The rule to score.
        solution (List[Rule]): Current set of selected rules.
        covered (Set[int]): Set of indices covered by current solution.
        covered_protected (Set[int]): Set of protected indices covered by current solution.
        protected_group (Set[int]): Set of indices in the protected group.
        unprotected_coverage_threshold (float): Threshold for unprotected group coverage.
        protected_coverage_threshold (float): Threshold for protected group coverage.

    Returns:
        float: The calculated score for the rule.
    """
    new_covered = rule.covered_indices - covered
    new_covered_protected = rule.covered_protected_indices - covered_protected

    logging.debug(
        f"Scoring rule: new_covered={len(new_covered)}, new_covered_protected={len(new_covered_protected)}")

    if len(rule.covered_indices) == 0:
        logging.warning("Rule covers no individuals, returning -inf score")
        return float('-inf')

    # Calculate expected utility with the new rule added to the solution
    new_solution = solution + [rule]
    expected_utility = calculate_expected_utility(new_solution)

    # Calculate coverage factors for both protected and unprotected groups
    protected_coverage_factor = (len(new_covered_protected) / len(protected_group)) / \
        protected_coverage_threshold if protected_coverage_threshold > 0 else 1
    unprotected_coverage_factor = (len(new_covered - new_covered_protected) / (len(rule.covered_indices) - len(
        protected_group))) / unprotected_coverage_threshold if unprotected_coverage_threshold > 0 else 1

    # Use the minimum of the two coverage factors
    coverage_factor = min(protected_coverage_factor,
                          unprotected_coverage_factor)

    score = rule.utility * coverage_factor

    logging.debug(f"Rule score: {score:.4f} (expected_utility: {expected_utility:.4f}, "
                  f"utility: {rule.utility:.4f}, coverage_factor: {coverage_factor:.4f}")

    return score


def greedy_fair_prescription_rules(rules: List[PrescriptionRule], protected_group: Set[int],
                                   unprotected_coverage_threshold: float, protected_coverage_threshold: float,
                                   max_rules: int, total_individuals: int, fairness_threshold: float) -> List[PrescriptionRule]:
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


def run_single_experiment_with_k(k_rules: int, df: pd.DataFrame, protected_group: Set[int], attrI: List[str], attrM: List[str], tgtO: str,
                   unprotected_coverage_threshold: float, protected_coverage_threshold: float,
                   fairness_threshold: float,
                   DAG: List[str], config: Dict) -> Dict:
    """
    Run an experiment for a specific number of rules (k).

    Args:
        k (int): Number of rules to select.
        df (pd.DataFrame): The input dataframe.
        protected_group (Set[int]): Set of indices in the protected group.
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

    grouping_patterns = get_grouping_patterns(df, attrI, APRIORI)

    # Get treatments for each grouping pattern
    logging.info("Getting treatments for each grouping pattern")
    group_treatments, _ = getGroupstreatmentsforGreedy(
        DAG, df, grouping_patterns, {}, tgtO, attrM, True, protected_group)

    # Create Rule objects
    rules = []
    for group, data in group_treatments.items():
        condition = eval(group)
        treatment = data['treatment']
        covered_indices = data['covered_indices']
        covered_protected_indices = covered_indices.intersection(
            protected_group)
        utility = data['utility']

        # Calculate protected_utility based on the proportion of protected individuals covered
        protected_proportion = len(covered_protected_indices) / \
            len(covered_indices) if len(covered_indices) > 0 else 0
        protected_utility = utility * protected_proportion

        rules.append(PrescriptionRule(condition, treatment, covered_indices,
                     covered_protected_indices, utility, protected_utility))

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

    selected_rules = greedy_fair_prescription_rules(rules, protected_group, unprotected_coverage_threshold,
                                                    protected_coverage_threshold, k_rules, total_individuals, fairness_threshold)

    # Calculate metrics
    expected_utility = calculate_expected_utility(selected_rules)
    total_coverage = set().union(
        *[rule.covered_indices for rule in selected_rules])
    total_protected_coverage = set().union(
        *[rule.covered_protected_indices for rule in selected_rules])
    protected_expected_utility = calculate_expected_utility(
        [PrescriptionRule(r.condition, r.treatment, r.covered_protected_indices, r.covered_protected_indices, r.protected_utility, r.protected_utility) for r in selected_rules])

    end_time = time.time()
    execution_time = end_time - start_time

    logging.info(f"Experiment results for k={k_rules}:")
    logging.info(f"Expected utility: {expected_utility:.4f}")
    logging.info(
        f"Protected expected utility: {protected_expected_utility:.4f}")
    logging.info(f"Coverage: {len(total_coverage) / len(df):.2%}")
    logging.info(
        f"Protected coverage: {len(total_protected_coverage) / len(protected_group):.2%}")

    return {
        'k': k_rules,
        'execution_time': execution_time,
        'selected_rules': selected_rules,
        'expected_utility': expected_utility,
        'protected_expected_utility': protected_expected_utility,
        'coverage': len(total_coverage) / len(df),
        'protected_coverage': len(total_protected_coverage) / len(protected_group)
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

    dataset_path, datatable_path, dag_path, immutable_attributes, mutable_attributes, protected_attributes, protected_values, target_outcome = config['_dataset_path'], config[
        '_datatable_path'], config['_dag_path'], config['_immutable_attributes'], config['_mutable_attributes'], config['_protected_attributes'], config['_protected_values'],  config['_target_outcome']
    MIX_K, MAX_K = config['_k']
    sys.path.append(os.path.join(DATA_PATH, dataset_path))
    from dags import SO_DAG

    # ------------------------- PARSING CONFIG ENDS  -------------------------
    # ------------------------ DATASET SETUP BEGINS -------------------------- 
    df = load_data(os.path.join(DATA_PATH, dataset_path, datatable_path))
    # Define protected group
    protected_group = set(
        df[df[protected_attributes] != protected_values].index)

    logging.info(
        f"Protected group size: {len(protected_group)} out of {len(df)} total")
    # ------------------------ DATASET SETUP ENDS ----------------------------

    # Run experiments for different values of k = the number of rules
    results = []
    for k in range(MIX_K, MAX_K + 1):
        result = run_single_experiment_with_k(k, df, protected_group, immutable_attributes, mutable_attributes, target_outcome,
                                unprotected_coverage_threshold, protected_coverage_threshold,
                                fairness_threshold, SO_DAG, config)
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
