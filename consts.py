import os
from pathlib import Path


APRIORI = 0.1
MIX_K = 4
MAX_K = 7
unprotected_coverage_threshold = 0.5
protected_coverage_threshold = 0.5
fairness_threshold = 0.05

# Location of config.py
CONST_PATH = os.path.abspath(__file__)

# Path of the project
PROJECT_PATH = Path(__file__).parent