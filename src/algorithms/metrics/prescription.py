from re import L
from typing import Dict, List, Set

from pydantic import InstanceOf

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

    def getCondition(self):
        return self.condition
    def getGroup(self):
        return self.condition
    
    def getTreatment(self):
        return self.treatment

    def getUtility(self) -> float:
        if self.utility == None:
            return -1.0
        return self.utility
    
    def getProtectedUtility(self) -> float:
        if self.protected_utility == None:
            return -1.0
        return self.protected_utility
    def getCoveredIdx(self):
        return self.covered_idx
    
    def getCoverage(self):
        return len(self.covered_idx)
    def getProtectedCoverage(self):
        return len(self.covered_idx_protected)
    def compare(self, obj):
        if not isinstance(obj, Prescription):
            return -1
        else:
            return self.utility - obj.utility
    
 

class PrescriptionSet:
    def __init__(self, rules: List[Prescription], idx_protec):
        self.rules = rules
        self.idx_protec = set(idx_protec)
        self.covered_idx = set().union(*[rule.covered_idx for rule in self.rules])
        self.covered_idx_protected = self.covered_idx & self.idx_protec 
        expected_utilities = self.expected_utilities()
        self.expected_utility, self.protected_expected_utility = expected_utilities 
    def getRules(self) -> List[Prescription]: 
        return self.rules
    def expected_utilities(self) -> float:
        """
        Calculate the expected utility of a set of rules.

        Args:
            rules (List[Rule]): List of rules to calculate the expected utility for.

        Returns:
            float: The expected utility and expected protected utility.
        """
        # TODO double check old implementation
        if len(self.covered_idx) == 0:
            return 0.0, 0.0
        total_utility = 0.0
        for t in self.covered_idx:
            rules_covering_t = [r for r in self.getRules() if t in r.covered_idx]
            min_utility = min(r.utility for r in rules_covering_t)
            total_utility += min_utility
        
        if len(self.covered_idx_protected) == 0:
            return total_utility / len(self.covered_idx), 0.0
        total_protected_utility = 0.0
        for t in self.covered_idx_protected:
            rules_covering_t = [r for r in self.getRules() if t in r.covered_idx_protected]
            min_utility = min(r.utility for r in rules_covering_t)
            total_utility += min_utility
        return total_utility, total_protected_utility 
    def getExpectedUtility(self):
        return self.expected_utility
    def getProtectedExpectedUtility(self):
        return self.protected_expected_utility 
    def getCoveredIdx(self):
        return self.covered_idx
    def getCoveredIdxProtected(self):
        return self.covered_idx_protected 
    def getCoverage(self):
        return len(self.covered_idx)
    def getProtectedCoverage(self):
        return len(self.covered_idx_protected) 
    def toDict(self):
        rxDict = {}
        for r in self.getRules():
            rxDict[str(r.condition)] = r.treatment
        return rxDict 
        

           