import os
from pathlib import Path


APRIORI = 0.1

# Location of config.py
CONST_PATH = os.path.abspath(__file__)

# Path of the project
PROJECT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
WORKER_OUTPUT_PATH = 'FairPrescriptionRules/output'
