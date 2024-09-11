import os
from pathlib import Path


APRIORI = 0.1
MIX_K = 2
MAX_K = 4
unprotected_coverage_threshold = 0.5
protected_coverage_threshold = 0.5
fairness_threshold = 0.05

# Location of config.py
CONST_PATH = os.path.abspath(__file__)

# Path of the project
PROJECT_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
WORKER_OUTPUT_PATH = 'FairPrescriptionRules/output'
