from typing import Dict, Set

class Prescription:
    """
    Represents a prescription with associated metrics.
    Developer note:
        should only instantiate a Prescription object when condition and treatment are clear 
    Attributes:
        condition (Dict): The condition part of the rule.
        treatment (Dict): The treatment part of the rule.
        covered_idx (Set[int]): Indices of individuals covered by this rule.
        covered_protected_indices (Set[int]): Indices of protected individuals covered by this rule.
        utility (float): The utility of this rule.
        protected_utility (float): The utility of this rule for the protected group.
    """

    def __init__(self, condition: Dict, 
            treatment: Dict, 
            covered_idx: Set[int],
            covered_idx_protected: Set[int], 
            utility: float, 
            protected_utility: float):
        self.condition = condition
        self.treatment = treatment
        self.covered_idx = covered_idx
        self.covered_idx_protected = covered_idx_protected
        self.utility = utility
        self.protected_utility = protected_utility

    def update_covered(self, df, idx_protec):
        assert(self.treatment != None)
        
        # self.covered_idx = covered_idx
        


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
    



# def treatments_to_Rx(group, treatments) -> List[Prescription]:
#     return list(map())

           