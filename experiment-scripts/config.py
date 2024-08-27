import logging
import os
from pathlib import Path


REMOTE_USER = "bcyl2"
REMOTE_HOSTNAMES = ["node0.remote.fair-prescrip-pg0.utah.cloudlab.us"]

# Location of config.py
CONFIG_PATH = os.path.abspath(__file__)

# Path of the project
PROJECT_PATH = Path(__file__).parent.parent

# Desired path for experiment output
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
