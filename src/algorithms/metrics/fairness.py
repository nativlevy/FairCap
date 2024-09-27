import logging
from typing import List, Set
from prescription import Prescription



def score_rule(rule: Prescription, solution: List[Prescription], covered: Set[int], covered_protected: Set[int],
               idx_protec,
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
    new_covered = rule.covered_idx - covered
    new_covered_protec = rule.covered_protected_indices - covered_protected
    logging.debug(
        f"Scoring rule: new_covered={len(new_covered)}, new_covered_protected={len(new_covered_protec)}")

    if len(rule.covered_idx) == 0:
        logging.warning("Rule covers no individuals, returning -inf score")
        return float('-inf')

    # Calculate expected utility with the new rule added to the solution
    new_solution = solution + [rule]
    exp_util, protec_exp_util = expected_utilities(new_solution, idx_protec)

    # Calculate coverage factors for both protected and unprotected groups
    protec_cvrg_factor = (len(new_covered_protec) / len(idx_protec)) / \
        protec_cvrg_th if protec_cvrg_th > 0 else 1
    unprotec_cvrg_factor = \
        (len(new_covered - new_covered_protec) / \
          (len(rule.covered_idx - idx_protec))) /\
            unprotected_coverage_threshold if unprotected_coverage_threshold > 0 else 1

    # Use the minimum of the two coverage factors
    coverage_factor = min(protec_cvrg_factor,
                          unprotec_cvrg_factor)

    score = rule.utility * coverage_factor

    logging.debug(f"Rule score: {score:.4f} (expected utility: {exp_util:.4f}, expected protected utility: {protec_exp_util:.4f}, "
                  f"utility: {rule.utility:.4f}, coverage_factor: {coverage_factor:.4f}")

    return score



def benefit(cate_all, cate_protec, cate_unprotec, fair_constr=None):
    """
    Calculate the fairness score for a given treatment.

    Args:

    Returns:
        float: The calculated fairness score.
    """
 
    # TODO confirm: if no fair_constr, do we perform greedy on CATE? 
    # TODO If it's individual SP, do we use group SP for greedy anyway?
    if fair_constr == None:
        return cate_all 
    if fair_constr['variant'] == 'group_bgl': 
        threshold = fair_constr['threshold']
        if threshold >= cate_protec:
            return cate_all / (threshold - cate_protec)
        else:
            return cate_all 
    elif fair_constr['variant'] != 'group_sp':
        if cate_protec - cate_unprotec >= fair_constr['threshold']:
            return cate_all
        else:
            return cate_all / abs(cate_unprotec - cate_protec)
    else:
        # TODO confirm benefit definition under other constaints
        return cate_all


# TODO overload for prescription type
# def group_fairness(prescription: Prescription, df_g, DAG, attrOrdinal, tgtO, protected_group, variant='SP'):
