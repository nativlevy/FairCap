import logging
import os
from pathlib import Path
import sys

# Location of config.py
CONFIG_PATH = os.path.abspath(__file__)
PROJECT_PATH = Path(__file__).parent.parent
ALL_OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')


REMOTE_USER = "bcyl2"
CLOUDLAB_EXPERIMENT_NAME = "remote"
CLOUDLAB_PROJECT_NAME = "fair-prescrip"

CLOUDLAB_NODENAMES = ["node0"]

# e.g. node1.remote.fair-prescrip-pg0.utah.cloudlab.us
REMOTE_HOSTNAMES = ["%s.%s.%s-pg0.utah.cloudlab.us" %
                    (node, CLOUDLAB_EXPERIMENT_NAME, CLOUDLAB_PROJECT_NAME) for node in CLOUDLAB_NODENAMES]
