
from typing import Dict, List, Set
import pandas as pd
import os, sys
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent))
from prescription import Prescription

# TODO
def group_coverage(rules: List[Prescription]) -> int:
    covered = set()
    for p in rules: 
        covered.update(p.covered_idx)
    return len(covered)

def protected_group_coverage(rules: List[Prescription]) -> int:
    covered = set()
    for p in rules: 
        covered.update(p.covered_protected_indices)
    return len(covered)


# coverage function in case there is only grouping
def group_coverage(df: pd.DataFrame, groups: List[Dict]) -> int:
    mask = (df[groups.keys()] == groups.values()).all(axis=1)
    covered = (df[mask]).index 
    return len(covered)

def protected_group_coverage(df: pd.DataFrame, groups: List[Dict], protected_index: Set[int]) -> int:
    mask = (df[groups.keys()] == groups.values()).all(axis=1)
    covered = (df[mask]).index 
    covered = covered.intersection(protected_index)
    return len(covered)  

def rule_coverage(rule: Prescription) -> int:
    return len(rule.covered_idx)

# Grouping only TODO: confirm with Brit
def rule_coverage(df: pd.DataFrame, rule: Dict) -> int:
    mask = (df[rule.keys()] == rule.values()).all(axis=1)
    covered = (df[mask]).index 
    return len(covered)

def apply_rule_coverage(rule: Dict, ):
    """
    Input: A collection of candidate rules
    Output: a collection of rules that statisfies
        1. Every rule covers at least some fraction theta of the tuples
        2. Every rule covers at least some fraction theta_p of the tuples

    """
    

