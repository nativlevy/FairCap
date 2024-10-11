import os
from pathlib import Path

GROUP_COVERAGE = "group_coverage"
RULE_COVERAGE = "rule_coverage"

GROUP_FAIRNESS_SP = "group_fairness_sp"
INDIVIDUAL_FAIRNESS_SP = "individual_fairness_sp" 
GROUP_FAIRNESS_BGL = "group_fairness_bgl"
INDIVIDUAL_FAIRNESS_BGL = "individual_fairness_bgl" 

APRIORI = 0.05
MIX_K = 2
MAX_K = 4
unprotected_coverage_threshold = 0.5
protected_coverage_threshold = 0.5
fairness_threshold = 0.05

# Location of config.py
CONST_PATH = os.path.abspath(__file__)

# Path of the project
PROJECT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
WORKER_OUTPUT_PATH = 'FairPrescriptionRules/output'
