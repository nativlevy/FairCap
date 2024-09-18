from typing import Dict, Set

class Prescription:
    """
    Represents a prescription with associated metrics.
    Developer note:
        should only instantiate a Prescription object when condition and treatment are clear 
    Attributes:
        condition (Dict): The condition part of the rule.
        treatment (Dict): The treatment part of the rule.
        covered_indices (Set[int]): Indices of individuals covered by this rule.
        covered_protected_indices (Set[int]): Indices of protected individuals covered by this rule.
        utility (float): The utility of this rule.
        protected_utility (float): The utility of this rule for the protected group.
    """
    def __init__(self, condition: Dict, treatment: Dict): 
        self.condition = condition
        self.treatment = treatment
    def __init__(self, condition: Dict, 
            treatment: Dict, 
            covered_indices: Set[int],
            covered_protected_indices: Set[int], 
            utility: float, 
            protected_utility: float):
        self.condition = condition
        self.treatment = treatment
        self.covered_indices = covered_indices
        self.covered_protected_indices = covered_protected_indices
        self.utility = utility
        self.protected_utility = protected_utility

    def condition(self):
        return self.condition
    def group(self):
        return self.condition
    
    def treatment(self):
        return self.treatment

    def utility(self) -> float:
        if self.utility == None:
            return -1
        return self.utility