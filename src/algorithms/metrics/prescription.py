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
    def __init__(self): 
        self.condition = None 
        self.treatment = None
        self.covered_idx = None
        self.covered_idx_protec = None
        self.utility = -1
        self.utility_protec = -1
    def __init__(self, condition: Dict, 
            treatment: Dict, 
            covered_idx: Set[int],
            covered_idx_protec: Set[int], 
            utility: float, 
            utility_protec: float):
        self.condition = condition
        self.treatment = treatment
        self.covered_idx = covered_idx
        self.covered_idx_protec = covered_idx_protec
        self.utility = utility
        self.utility_protec = utility_protec
    def __init__(self, treatment): 
        self.condition = None 
        self.treatment = None
        self.covered_idx = None
        self.covered_idx_protec = None
        self.utility = -1
        self.utility_protec = -1 
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
    

def merge(rx1: Prescription, rx2: Prescription) -> Prescription:
    # Ensure conditions are the same
    assert(rx1.condition() == (rx2.condition()))
    condition = rx1.condition
    treatment = {**rx1.treatment, **rx2.treatment}
    covered_idx = rx1.covered_idx.intersection(rx2.covered_idx)
    covered_idx_protec = rx1.covered_idx_protec.intersection(rx2.covered_idx_protec)
    # Note, utilities are not realized yet
    utility = 0
    utility_protec = 0
    return Prescription(condition, treatment, covered_idx, covered_idx_protec, covered_idx_protec, 0, 0)

# def treatments_to_Rx(group, treatments) -> List[Prescription]:
#     return list(map())

           