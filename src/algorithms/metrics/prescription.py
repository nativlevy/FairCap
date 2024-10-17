import copy
from re import L
from typing import Dict, List, Set


class Prescription:
    """
    Represents a prescription with associated metrics.
    Developer note:
        should only instantiate a Prescription object when condition and treatment are clear
    """

    def __init__(
        self,
        condition: Dict,
        treatment: Dict,
        covered_idx: Set[int],
        covered_idx_p: Set[int],
        utility: float,
        utility_p: float,
        utility_u: float,
        pvals: List[float],
    ):
        self.condition = condition
        self.treatment = treatment
        self.covered_idx = covered_idx
        self.covered_idx_p = covered_idx_p
        self.covered_idx_u = covered_idx - covered_idx_p

        self.utility = utility
        self.utility_p = utility_p
        self.utility_u = utility_u
        self.pvals = pvals
        self.name = self.make_name()

    def make_name(self):
        name = ""
        for k, v in self.condition.items():
            name += f"{k.replace(' ', '_')}:{v.replace(' ', '_')};"
        return name

    def getCondition(self):
        return self.condition

    def getGroup(self):
        return self.condition

    def getTreatment(self):
        return self.treatment

    def getUtility(self) -> float:
        if self.utility == None:
            return -1.0
        return round(self.utility, 2)

    def getProtectedUtility(self) -> float:
        if self.utility_p == None:
            return -1.0
        return round(self.utility_p, 2)

    def getUnprotectedUtility(self) -> float:
        if self.utility_u == None:
            return -1.0
        return round(self.utility_u, 2)

    def getCoveredIdx(self):
        return self.covered_idx

    def getCoverage(self):
        return len(self.covered_idx)

    def getProtectedCoverage(self):
        return len(self.covered_idx_p)

    def getPvals(self):
        pv_a, pv_p, pv_u = self.pvals
        return {"all": pv_a, "protected": pv_p, "non-protected": pv_u}

    def compare(self, obj):
        if not isinstance(obj, Prescription):
            return -1
        else:
            return self.utility - obj.utility


class PrescriptionList:
    def __init__(self, rules: List[Prescription], idx_all, idx_p):
        self.rules = rules
        self.idx_all = set(idx_all)
        self.idx_p = set(idx_p)
        self.idx_u = idx_all - self.idx_p

        if rules:
            self.covered_idx = set().union(*[rule.covered_idx for rule in self.rules])
            self.covered_idx_p = self.covered_idx & self.idx_p
            self.covered_idx_u = self.covered_idx - self.idx_p

            self.expected_utility, self.expected_utility_u, self.expected_utility_p = (
                self.calculate_expected_utilities()
            )

        else:
            self.covered_idx = set()
            self.covered_idx_p = set()
            self.covered_idx_u = set()
            self.expected_utility_u = 0
            self.expected_utility_p = 0
            self.expected_utility = 0

    def getRules(self) -> List[Prescription]:
        return self.rules

    def calculate_expected_utilities(self) -> float:
        """
        Calculate the expected utilities of a set of rules.


        Returns:
            float: The unprotected expected utility and protected expected utility.
        """
        if len(self.covered_idx) == 0:
            return 0.0, 0.0, 0.0
        exp_utility = 0.0
        exp_utility_u = 0.0
        exp_utility_p = 0.0

        for t in self.covered_idx_u:
            rules_covering_t = [r for r in self.getRules() if t in r.covered_idx_u]
            max_utility = max(r.utility for r in rules_covering_t)
            exp_utility_u += max_utility / len(self.idx_u)
            exp_utility += max_utility / len(self.idx_all)
            # covered_idx_u != [] => idx_u != []

        for t in self.covered_idx_p:
            rules_covering_t = [r for r in self.getRules() if t in r.covered_idx_p]
            min_utility = min(r.utility for r in rules_covering_t)
            max_utility = max(r.utility for r in rules_covering_t)
            exp_utility_p += min_utility / len(self.idx_p)
            exp_utility += max_utility / len(self.idx_all)

        return exp_utility, exp_utility_u, exp_utility_p

    def getExpectedUtility(self):
        return round(self.expected_utility, 2)

    def getProtectedExpectedUtility(self):
        return round(self.expected_utility_p, 2)

    def getUnrotectedExpectedUtility(self):
        return round(self.expected_utility_u, 2)

    def getCoveredIdx(self):
        return self.covered_idx

    def getCoveredIdxProtected(self):
        return self.covered_idx_p

    def getCoverage(self):
        return len(self.covered_idx)

    def getProtectedCoverage(self):
        return len(self.covered_idx_p)

    def getCoverageRate(self):
        return round(len(self.covered_idx) / len(self.idx_all), 4)

    def getProtectedCoverageRate(self):
        return round(len(self.covered_idx_p) / len(self.idx_p), 4)

    def toDict(self):
        rxDict = {}
        for r in self.getRules():
            rxDict[str(r.condition)] = r.treatment
        return rxDict

    def addRx(self, newRx: Prescription):
        return PrescriptionList(self.rules + [newRx], self.idx_all, self.idx_p)

    def isFairnessMet(self, fair_constr) -> bool:
        if fair_constr == None:
            return True

        if "group" not in fair_constr["variant"]:
            return True
        if not self.rules:
            return False
        if "group_sp" in fair_constr["variant"]:
            threshold = fair_constr["threshold"]
            if abs(self.expected_utility_u - self.expected_utility_p) > threshold:
                return False

        if "group_bgl" in fair_constr["variant"]:
            threshold = fair_constr["threshold"]
            if self.expected_utility_p < threshold:
                return False
        return True

    def isCoverageMet(self, cvrg_constr) -> bool:
        if cvrg_constr == None:
            return True
        if "group" in cvrg_constr["variant"]:

            threshold = cvrg_constr["threshold"]
            threshold_p = cvrg_constr["threshold_p"]
            return (
                self.getCoverageRate() >= threshold
                and self.getProtectedCoverage() >= threshold_p
            )

        return True
