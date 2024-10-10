
from typing import Dict, List, MutableSequence, Set
import pandas as pd
import os, sys
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent))
from prescription import Prescription

# TODO
def group_coverage(rules: MutableSequence[Prescription]) -> int:
    covered = set()
    for p in rules: 
        covered.update(p.covered_idx)
    return len(covered)

def protected_group_coverage(rules: MutableSequence[Prescription]) -> int:
    covered = set()
    for p in rules: 
        covered.update(p.covered_protected_indices)
    return len(covered)


# coverage function in case there is only grouping
def group_coverage(df: pd.DataFrame, groups: MutableSequence[Dict]) -> int:
    mask = (df[groups.keys()] == groups.values()).all(axis=1)
    covered = (df[mask]).index 
    return len(covered)

def protected_group_coverage(df: pd.DataFrame, groups: MutableSequence[Dict], protected_index: Set[int]) -> int:
    mask = (df[groups.keys()] == groups.values()).all(axis=1)
    covered = (df[mask]).index 
    covered = covered.intersection(protected_index)
    return len(covered)  

def rule_coverage(rule: Prescription) -> int:
    return len(rule.covered_idx)

# Grouping only TODO: confirm with Brit
def rule_coverage(df: pd.DataFrame, idx_p, rule: Dict) -> int:
    mask = (df[rule.keys()] == rule.values()).all(axis=1)
    covered = set(df[mask].index)
    return [len(covered), len(covered & idx_p)]


