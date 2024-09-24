import logging
from typing import List, Set
from prescription import Prescription

from utility_functions import CATE, expected_utility


def score_rule(rule: Prescription, solution: List[Prescription], covered: Set[int], covered_protected: Set[int],
               protected_group: Set[int],
               unprotected_coverage_threshold: float, protec_cvrg_th: float) -> float:
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
    new_covered_protec = rule.covered_protected_indices - covered_protected
    logging.debug(
        f"Scoring rule: new_covered={len(new_covered)}, new_covered_protected={len(new_covered_protec)}")

    if len(rule.covered_indices) == 0:
        logging.warning("Rule covers no individuals, returning -inf score")
        return float('-inf')

    # Calculate expected utility with the new rule added to the solution
    new_solution = solution + [rule]
    exp_util = expected_utility(new_solution)

    # Calculate coverage factors for both protected and unprotected groups
    protec_cvrg_factor = (len(new_covered_protec) / len(protected_group)) / \
        protec_cvrg_th if protec_cvrg_th > 0 else 1
    unprotec_cvrg_factor = \
        (len(new_covered - new_covered_protec) / \
          (len(rule.covered_indices - protected_group))) /\
            unprotected_coverage_threshold if unprotected_coverage_threshold > 0 else 1

    # Use the minimum of the two coverage factors
    coverage_factor = min(protec_cvrg_factor,
                          unprotec_cvrg_factor)

    score = rule.utility * coverage_factor

    logging.debug(f"Rule score: {score:.4f} (expected_utility: {exp_util:.4f}, "
                  f"utility: {rule.utility:.4f}, coverage_factor: {coverage_factor:.4f}")

    return score



def benefit(cate_all, cate_protec, cate_unprotec, fair_constr={'variant': 'SP'}):
    """
    Calculate the fairness score for a given treatment.

    Args:

    Returns:
        float: The calculated fairness score.
    """
    if fair_constr == None:
        fair_constr = {'variant': 'SP'}
        
    if fair_constr['variant'] == 'SP':
        if cate_protec - cate_unprotec >= -0.001:
            return cate_all
        else:
            return cate_all / abs(cate_unprotec - cate_protec)
    elif fair_constr['variant'] == 'BGL': 
        tau = fair_constr['tau']
        if tau >= cate_protec:
            return cate_all / (tau - cate_protec)
        else:
            return cate_all 

# TODO overload for prescription type
# def group_fairness(prescription: Prescription, df_g, DAG, attrOrdinal, tgtO, protected_group, variant='SP'):
