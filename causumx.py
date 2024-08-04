import pandas as pd
from typing import List, Set, Dict
from Algorithms_causumx import getAllGroups, getGroupstreatmentsforGreeedy
from consts import APRIORI, MIX_K, MAX_K, unprotected_coverage_threshold, protected_coverage_threshold, \
    fairness_threshold
from dags import SO_DAG
import logging
import time
import csv
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

class Rule:
    """
    Represents a prescription rule with associated metrics.

    Attributes:
        condition (Dict): The condition part of the rule.
        treatment (Dict): The treatment part of the rule.
        covered_indices (Set[int]): Indices of individuals covered by this rule.
        covered_protected_indices (Set[int]): Indices of protected individuals covered by this rule.
        utility (float): The utility of this rule.
        protected_utility (float): The utility of this rule for the protected group.
    """
    def __init__(self, condition: Dict, treatment: Dict, covered_indices: Set[int],
                 covered_protected_indices: Set[int], utility: float, protected_utility: float):
        self.condition = condition
        self.treatment = treatment
        self.covered_indices = covered_indices
        self.covered_protected_indices = covered_protected_indices
        self.utility = utility
        self.protected_utility = protected_utility

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def get_grouping_patterns(df: pd.DataFrame, attributes: List[str], apriori: float) -> List[dict]:
    """
    Generate and filter grouping patterns from the data.

    Args:
        df (pd.DataFrame): Input data.
        attributes (List[str]): Attributes to consider for grouping.
        apriori (float): Apriori threshold for pattern generation.

    Returns:
        List[dict]: Filtered list of grouping patterns.
    """
    logging.info(f"Getting grouping patterns with apriori={apriori}")
    grouping_patterns = getAllGroups(df, attributes, apriori)
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

def return_rule_utility(rule: Rule) -> float:
    return rule.utility

def calculate_expected_utility(rules: List[Rule]) -> float:
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

def score_rule(rule: Rule, solution: List[Rule], covered: Set[int], covered_protected: Set[int],
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

    logging.debug(f"Scoring rule: new_covered={len(new_covered)}, new_covered_protected={len(new_covered_protected)}")

    if len(rule.covered_indices) == 0:
        logging.warning("Rule covers no individuals, returning -inf score")
        return float('-inf')

    # Calculate expected utility with the new rule added to the solution
    new_solution = solution + [rule]
    expected_utility = calculate_expected_utility(new_solution)

    # Calculate coverage factors for both protected and unprotected groups
    protected_coverage_factor = (len(new_covered_protected) / len(protected_group)) / protected_coverage_threshold if protected_coverage_threshold > 0 else 1
    unprotected_coverage_factor = (len(new_covered - new_covered_protected) / (len(rule.covered_indices) - len(protected_group))) / unprotected_coverage_threshold if unprotected_coverage_threshold > 0 else 1

    # Use the minimum of the two coverage factors
    coverage_factor = min(protected_coverage_factor, unprotected_coverage_factor)

    score = rule.utility * coverage_factor

    logging.debug(f"Rule score: {score:.4f} (expected_utility: {expected_utility:.4f}, "
                  f"fairness_score: {rule.utility:.4f}, coverage_factor: {coverage_factor:.4f}")

    return score

def greedy_fair_prescription_rules(rules: List[Rule], protected_group: Set[int], 
                                   unprotected_coverage_threshold: float, protected_coverage_threshold: float,
                                   max_rules: int, total_individuals: int, fairness_threshold: float) -> List[Rule]:
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
            logging.info("Coverage thresholds met, focusing on protected group")
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
            logging.info("No more rules can improve protected utility, stopping")
            break

        solution.append(best_rule)
        covered.update(best_rule.covered_indices)
        covered_protected.update(best_rule.covered_protected_indices)
        total_utility += best_rule.utility
        protected_utility += best_rule.protected_utility

        logging.info(f"Added protected-utility-improving rule {len(solution)}: protected_utility={best_protected_utility:.4f}, "
                     f"total_covered={len(covered)}, protected_covered={len(covered_protected)}, "
                     f"total_utility={total_utility:.4f}, protected_utility={protected_utility:.4f}")

    return solution

def run_experiment(k: int, df: pd.DataFrame, protected_group: Set[int], attributes: List[str], 
                   unprotected_coverage_threshold: float, protected_coverage_threshold: float, 
                   fairness_threshold: float) -> Dict:
    """
    Run an experiment for a specific number of rules (k).

    Args:
        k (int): Number of rules to select.
        df (pd.DataFrame): The input dataframe.
        protected_group (Set[int]): Set of indices in the protected group.
        attributes (List[str]): List of attributes for grouping patterns.
        unprotected_coverage_threshold (float): Threshold for unprotected group coverage.
        protected_coverage_threshold (float): Threshold for protected group coverage.
        fairness_threshold (float): Threshold for fairness constraint.

    Returns:
        Dict: A dictionary containing the results of the experiment.
    """
    start_time = time.time()

    grouping_patterns = get_grouping_patterns(df, attributes, APRIORI)

    # Get treatments for each grouping pattern
    DAG = SO_DAG
    targetClass = 'ConvertedSalary'
    actionable_atts = [
        'Exercise', 'HoursComputer', 'DevType', 'FormalEducation', 'UndergradMajor', 'Country', 'Continent', 'Hobby', 'Student',
    ]

    logging.info("Getting treatments for each grouping pattern")
    group_treatments, _ = getGroupstreatmentsforGreeedy(DAG, df, grouping_patterns, {}, targetClass, actionable_atts, True, protected_group)

    # Create Rule objects
    rules = []
    for group, data in group_treatments.items():
        condition = eval(group)
        treatment = data['treatment']
        covered_indices = data['covered_indices']
        covered_protected_indices = covered_indices.intersection(protected_group)
        utility = data['utility']
        
        # Calculate protected_utility based on the proportion of protected individuals covered
        protected_proportion = len(covered_protected_indices) / len(covered_indices) if len(covered_indices) > 0 else 0
        protected_utility = utility * protected_proportion
        
        rules.append(Rule(condition, treatment, covered_indices, covered_protected_indices, utility, protected_utility))

    logging.info(f"Created {len(rules)} Rule objects")

    # Run greedy algorithm
    total_individuals = len(df)
    logging.info(f"Running greedy algorithm with unprotected coverage threshold {unprotected_coverage_threshold}, "
                 f"protected coverage threshold {protected_coverage_threshold}, "
                 f"{k} rules, and fairness threshold {fairness_threshold}")

    # save all rules to an output file
    with open('rules_causumx.json', 'w') as f:
        json.dump([{
            'condition': rule.condition,
            'treatment': rule.treatment,
            'utility': rule.utility,
            'protected_utility': rule.protected_utility,
            'coverage': len(rule.covered_indices),
            'protected_coverage': len(rule.covered_protected_indices)
        } for rule in rules], f, indent=4)

    selected_rules = greedy_fair_prescription_rules(rules, protected_group, unprotected_coverage_threshold,
                                                    protected_coverage_threshold, k, total_individuals, fairness_threshold)

    # Calculate metrics
    expected_utility = calculate_expected_utility(selected_rules)
    total_coverage = set().union(*[rule.covered_indices for rule in selected_rules])
    total_protected_coverage = set().union(*[rule.covered_protected_indices for rule in selected_rules])
    protected_expected_utility = calculate_expected_utility([Rule(r.condition, r.treatment, r.covered_protected_indices, r.covered_protected_indices, r.protected_utility, r.protected_utility) for r in selected_rules])

    end_time = time.time()
    execution_time = end_time - start_time

    return {
        'k': k,
        'execution_time': execution_time,
        'selected_rules': selected_rules,
        'expected_utility': expected_utility,
        'protected_expected_utility': protected_expected_utility,
        'coverage': len(total_coverage) / len(df),
        'protected_coverage': len(total_protected_coverage) / len(protected_group)
    }

def main():
    """
    Main function to run the greedy fair prescription rules algorithm for different values of k.
    """
    # Load data
    df = load_data('data/so_countries_col_new.csv')

    # Define protected group (non-male in this case)
    protected_group = set(df[df['RaceEthnicity'] != 'White or of European descent'].index)
    logging.info(f"Protected group size: {len(protected_group)} out of {len(df)} total")

    # Define attributes for grouping patterns
    attributes = [
        'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity',
        'Age', 'YearsCoding', 'Dependents',
    ]

    # Run experiments for different values of k
    results = []
    for k in range(MIX_K, MAX_K + 1):
        result = run_experiment(k, df, protected_group, attributes, 
                                unprotected_coverage_threshold, protected_coverage_threshold,
                                fairness_threshold)
        results.append(result)
        logging.info(f"Completed experiment for k={k}")

    # Write results to CSV
    with open('experiment_results_causumx.csv', 'w', newline='') as csvfile:
        fieldnames = ['k', 'execution_time', 'expected_utility', 'protected_expected_utility', 'coverage', 'protected_coverage', 'selected_rules']
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

    logging.info("Results written to experiment_results_causumx.csv")

    # Log detailed results for each k
    for result in results:
        logging.info(f"\nDetailed results for k={result['k']}:")
        logging.info(f"Execution time: {result['execution_time']:.2f} seconds")
        logging.info(f"Expected utility: {result['expected_utility']:.4f}")
        logging.info(f"Protected expected utility: {result['protected_expected_utility']:.4f}")
        logging.info(f"Coverage: {result['coverage']:.2%}")
        logging.info(f"Protected coverage: {result['protected_coverage']:.2%}")
        # logging.info("Selected rules:")
        # for i, rule in enumerate(result['selected_rules'], 1):
        #     logging.info(f"Rule {i}:")
        #     logging.info(f"  Condition: {rule.condition}")
        #     logging.info(f"  Treatment: {rule.treatment}")
        #     logging.info(f"  Utility: {rule.utility:.4f}")
        #     logging.info(f"  Protected Utility: {rule.protected_utility:.4f}")
        #     logging.info(f"  Coverage: {len(rule.covered_indices)}")
        #     logging.info(f"  Protected Coverage: {len(rule.covered_protected_indices)}")

if __name__ == "__main__":
    main()
