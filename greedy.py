import pandas as pd
from typing import List, Set, Dict
from Algorithms import getAllGroups, getGroupstreatmentsforGreeedy
from functional_deps import calculate_functional_dependencies
import Utils
from dags import SO_DAG
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

class Rule:
    def __init__(self, condition: Dict, treatment: Dict, covered_indices: Set[int],
                 covered_protected_indices: Set[int], utility: float, protected_utility: float):
        self.condition = condition
        self.treatment = treatment
        self.covered_indices = covered_indices
        self.covered_protected_indices = covered_protected_indices
        self.utility = utility
        self.protected_utility = protected_utility

def load_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def get_grouping_patterns(df: pd.DataFrame, fds: List[str], apriori: float) -> List[dict]:
    logging.info(f"Getting grouping patterns with apriori={apriori}")
    grouping_patterns = getAllGroups(df, fds, apriori)
    logging.info(f"Found {len(grouping_patterns)} grouping patterns")
    return grouping_patterns

def score_rule(rule: Rule, solution: List[Rule], covered: Set[int], covered_protected: Set[int],
               total_utility: float, protected_utility: float, protected_group: Set[int],
               coverage_threshold: float) -> float:
    new_covered = rule.covered_indices - covered
    new_covered_protected = rule.covered_protected_indices - covered_protected

    if len(rule.covered_indices) == 0:
        logging.warning("Rule covers no individuals, returning -inf score")
        return float('-inf')

    utility_increase = rule.utility
    protected_utility_increase = rule.protected_utility

    if len(covered_protected) + len(new_covered_protected) == 0:
        fairness_factor = 1
    else:
        fairness_factor = 1 - abs(
            (protected_utility + protected_utility_increase) / (len(covered_protected) + len(new_covered_protected)) -
            (total_utility + utility_increase) / (len(covered) + len(new_covered)))

    coverage_factor = (len(new_covered_protected) / len(protected_group)) / coverage_threshold if coverage_threshold > 0 else 1

    unfairness_score = rule.utility / abs(rule.utility - rule.protected_utility) if rule.utility != rule.protected_utility else rule.utility

    score = (utility_increase + protected_utility_increase) * fairness_factor * coverage_factor * unfairness_score

    logging.debug(f"Rule score: {score:.4f} (utility_increase: {utility_increase:.4f}, "
                  f"protected_utility_increase: {protected_utility_increase:.4f}, "
                  f"fairness_factor: {fairness_factor:.4f}, coverage_factor: {coverage_factor:.4f}, "
                  f"unfairness_score: {unfairness_score:.4f})")

    return score

def greedy_fair_prescription_rules(rules: List[Rule], protected_group: Set[int], coverage_threshold: float,
                                   max_rules: int, total_individuals: int) -> List[Rule]:
    solution = []
    covered = set()
    covered_protected = set()
    total_utility = 0
    protected_utility = 0

    unprotected_count = total_individuals - len(protected_group)
    logging.info(f"Starting greedy algorithm with {len(rules)} rules, "
                 f"{len(protected_group)} protected individuals, "
                 f"{unprotected_count} unprotected individuals, "
                 f"coverage threshold {coverage_threshold}, and max {max_rules} rules")

    while len(solution) < max_rules:
        best_rule = None
        best_score = float('-inf')

        for rule in rules:
            if rule not in solution:
                score = score_rule(rule, solution, covered, covered_protected,
                                   total_utility, protected_utility, protected_group, coverage_threshold)
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
                     f"total_covered={len(covered)}, protected_covered={len(covered_protected)}")

    return solution

def main():
    # Load data
    df = load_data('data/so_countries_col_new.csv')

    grouping_attribute = 'Country'

    fds = calculate_functional_dependencies(df, grouping_attribute)

    # add the grouping attribute to the list of functional dependencies as the first element
    fds = [grouping_attribute] + fds
    logging.info(f"Functional Dependencies: {fds}")

    # Define protected group (non-male in this case)
    protected_group = set(df[df['Gender'] != 'Male'].index)
    logging.info(f"Protected group size: {len(protected_group)} out of {len(df)} total")

    APRIORI = 0.1

    # Get the Grouping Patterns
    grouping_patterns = get_grouping_patterns(df, fds, APRIORI)

    # Print each grouping pattern
    for i, pattern in enumerate(grouping_patterns, 1):
        logging.debug(f"Grouping Pattern {i}:")
        for attribute, value in pattern.items():
            logging.debug(f"  {attribute}: {value}")

    # Get treatments for each grouping pattern
    DAG = SO_DAG
    ordinal_atts = {}  # Define your ordinal attributes here
    targetClass = 'ConvertedSalary'
    groupingAtt = 'Country'
    actionable_atts = [
        'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity',
        'Age'
    ]

    logging.info("Getting treatments for each grouping pattern")
    groups_dic, _ = getGroupstreatmentsforGreeedy(DAG, df, grouping_attribute, grouping_patterns, ordinal_atts, targetClass, True, False, actionable_atts, True)

    # Create Rule objects
    rules = []
    for group, data in groups_dic.items():
        condition = eval(group)
        treatment = data[2]
        covered_indices = data[1]
        covered_protected_indices = covered_indices.intersection(protected_group)
        utility = data[3]
        protected_utility = utility * len(covered_protected_indices) / len(covered_indices)
        rules.append(Rule(condition, treatment, covered_indices, covered_protected_indices, utility, protected_utility))

    logging.info(f"Created {len(rules)} Rule objects")

    # Run greedy algorithm
    coverage_threshold = 0.8
    max_rules = 10
    total_individuals = len(df)  # Use the total number of rows in the dataframe
    logging.info(f"Running greedy algorithm with coverage threshold {coverage_threshold} and max {max_rules} rules")
    selected_rules = greedy_fair_prescription_rules(rules, protected_group, coverage_threshold, max_rules, total_individuals)

    # Print selected rules
    for i, rule in enumerate(selected_rules, 1):
        logging.info(f"Rule {i}:")
        logging.info(f"  Condition: {rule.condition}")
        logging.info(f"  Treatment: {rule.treatment}")
        logging.info(f"  Utility: {rule.utility}")
        logging.info(f"  Protected Utility: {rule.protected_utility}")
        logging.info(f"  Coverage: {len(rule.covered_indices)}")
        logging.info(f"  Protected Coverage: {len(rule.covered_protected_indices)}")

    # Calculate final fairness measure
    total_coverage = set().union(*[rule.covered_indices for rule in selected_rules])
    total_protected_coverage = set().union(*[rule.covered_protected_indices for rule in selected_rules])
    total_utility = sum(rule.utility for rule in selected_rules)
    total_protected_utility = sum(rule.protected_utility for rule in selected_rules)

    fairness_measure = abs(
        (total_protected_utility / len(total_protected_coverage)) -
        (total_utility / len(total_coverage))
    )

    logging.info(f"Final Fairness Measure: {fairness_measure}")
    logging.info(f"Total Coverage: {len(total_coverage)} out of {len(df)} ({len(total_coverage)/len(df)*100:.2f}%)")
    logging.info(f"Protected Coverage: {len(total_protected_coverage)} out of {len(protected_group)} ({len(total_protected_coverage)/len(protected_group)*100:.2f}%)")

if __name__ == "__main__":
    main()
