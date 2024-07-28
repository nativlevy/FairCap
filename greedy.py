import pandas as pd
from typing import List, Set, Dict
from Algorithms import getAllGroups, getGroupstreatmentsforGreeedy
from dags import SO_DAG
import logging
import time

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
        return set(df.index[mask])

    # Log details for each initial pattern
    for i, pattern in enumerate(grouping_patterns):
        covered_indices = apply_pattern(pattern)
        logging.debug(f"Pattern {i}: {pattern}")
        logging.debug(f"  Number of covered indices: {len(covered_indices)}")

    # Sort patterns by length (shorter first) and then by coverage (larger coverage first)
    sorted_patterns = sorted(grouping_patterns, key=lambda x: (len(x), -len(apply_pattern(x))))

    logging.info("Sorted patterns (shortest first, then largest coverage):")
    for i, pattern in enumerate(sorted_patterns):
        covered_indices = apply_pattern(pattern)
        logging.info(f"Pattern {i}: {pattern}")
        logging.info(f"  Length: {len(pattern)}")

    filtered_patterns = []
    for pattern in sorted_patterns:
        pattern_coverage = apply_pattern(pattern)
        is_subset = False
        for existing_pattern in filtered_patterns:
            if len(existing_pattern) <= len(pattern):
                existing_coverage = apply_pattern(existing_pattern)
                if pattern_coverage.issubset(existing_coverage):
                    is_subset = True
                    logging.debug(f"Pattern {pattern} is a subset of existing pattern {existing_pattern}")
                    break
        
        if not is_subset:
            filtered_patterns.append(pattern)
            logging.info(f"Added pattern to filtered list: {pattern}")

    logging.info(f"Filtered grouping patterns: {len(filtered_patterns)}")
    logging.info("Final filtered patterns:")
    for i, pattern in enumerate(filtered_patterns):
        covered_indices = apply_pattern(pattern)
        logging.info(f"Pattern {i}: {pattern}")
        logging.info(f"  Length: {len(pattern)}")

    return filtered_patterns

def calculate_fairness_score(rule: Rule) -> float:
    """
    Calculate the fairness score for a given rule.

    Args:
        rule (Rule): The rule to calculate the fairness score for.

    Returns:
        float: The calculated fairness score.
    """
    if rule.utility == rule.protected_utility:
        return rule.utility
    return rule.utility / abs(rule.utility - rule.protected_utility)

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

    fairness_score = calculate_fairness_score(rule)
    
    # Calculate coverage factors for both protected and unprotected groups
    protected_coverage_factor = (len(new_covered_protected) / len(protected_group)) / protected_coverage_threshold if protected_coverage_threshold > 0 else 1
    unprotected_coverage_factor = (len(new_covered - new_covered_protected) / (len(rule.covered_indices) - len(protected_group))) / unprotected_coverage_threshold if unprotected_coverage_threshold > 0 else 1

    # Use the minimum of the two coverage factors
    coverage_factor = min(protected_coverage_factor, unprotected_coverage_factor)

    score = expected_utility * fairness_score * coverage_factor

    logging.debug(f"Rule score: {score:.4f} (expected_utility: {expected_utility:.4f}, "
                  f"fairness_score: {fairness_score:.4f}, coverage_factor: {coverage_factor:.4f}")

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

    return solution

def main():
    """
    Main function to run the greedy fair prescription rules algorithm.
    """
    start_time = time.time()

    # Load data
    df = load_data('data/so_countries_col_new.csv')

    # Define protected group (non-male in this case)
    protected_group = set(df[df['Gender'] != 'Male'].index)
    logging.info(f"Protected group size: {len(protected_group)} out of {len(df)} total")

    # Define attributes for grouping patterns
    attributes = [
        'Country', 'Gender', 'SexualOrientation', 'EducationParents', 'RaceEthnicity',
        'Age'
    ]

    APRIORI = 0.1
    grouping_patterns = get_grouping_patterns(df, attributes, APRIORI)

    # Get treatments for each grouping pattern
    DAG = SO_DAG
    targetClass = 'ConvertedSalary'
    actionable_atts = [
        'HoursComputer', 'DevType', 'FormalEducation', 'UndergradMajor', 'Continent'
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
        
        logging.debug(f"Rule creation: condition={condition}, treatment={treatment}, "
                      f"covered={len(covered_indices)}, protected_covered={len(covered_protected_indices)}, "
                      f"utility={utility:.4f}, protected_utility={protected_utility:.4f}, "
                      f"protected_proportion={protected_proportion:.4f}")
        
        rules.append(Rule(condition, treatment, covered_indices, covered_protected_indices, utility, protected_utility))

    logging.info(f"Created {len(rules)} Rule objects")

    # Run greedy algorithm
    unprotected_coverage_threshold = 0.7
    protected_coverage_threshold = 0.5
    max_rules = 5
    fairness_threshold = 10000  # USD difference in utility between protected and unprotected groups
    total_individuals = len(df)
    logging.info(f"Running greedy algorithm with unprotected coverage threshold {unprotected_coverage_threshold}, "
                 f"protected coverage threshold {protected_coverage_threshold}, "
                 f"max {max_rules} rules, and fairness threshold {fairness_threshold}")

    selected_rules = greedy_fair_prescription_rules(rules, protected_group, unprotected_coverage_threshold,
                                                    protected_coverage_threshold, max_rules, total_individuals, fairness_threshold)

    # Log selected rules
    for i, rule in enumerate(selected_rules, 1):
        logging.info(f"Rule {i}:")
        logging.info(f"  Condition: {rule.condition}")
        logging.info(f"  Treatment: {rule.treatment}")
        logging.info(f"  Utility: {rule.utility:.4f}")
        logging.info(f"  Protected Utility: {rule.protected_utility:.4f}")
        logging.info(f"  Coverage: {len(rule.covered_indices)}")
        logging.info(f"  Protected Coverage: {len(rule.covered_protected_indices)}")

    # Calculate final fairness measure using expected utility
    expected_utility = calculate_expected_utility(selected_rules)
    total_coverage = set().union(*[rule.covered_indices for rule in selected_rules])
    total_protected_coverage = set().union(*[rule.covered_protected_indices for rule in selected_rules])

    logging.info(f"Total Coverage: {len(total_coverage)} out of {len(df)} ({len(total_coverage)/len(df)*100:.2f}%)")
    logging.info(f"Protected Coverage: {len(total_protected_coverage)} out of {len(protected_group)} ({len(total_protected_coverage)/len(protected_group)*100:.2f}%)")
    logging.info(f"Expected Utility: {expected_utility:.4f}")

    # Calculate protected expected utility
    protected_expected_utility = calculate_expected_utility([Rule(r.condition, r.treatment, r.covered_protected_indices, r.covered_protected_indices, r.protected_utility, r.protected_utility) for r in selected_rules])
    logging.info(f"Expected Protected Utility: {protected_expected_utility:.4f}")

    # Check if the fairness constraint is satisfied
    fairness_measure = abs(protected_expected_utility - expected_utility)

    if fairness_measure <= fairness_threshold:
        logging.info(f"Fairness constraint satisfied: {fairness_measure:.4f} <= {fairness_threshold}")
    else:
        logging.warning(f"Fairness constraint violated: {fairness_measure:.4f} > {fairness_threshold}")

    # Print the total time of this whole program
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Starting time: {start_time}")
    logging.info(f"Ending time: {end_time}")
    logging.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
