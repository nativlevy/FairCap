import logging
import os
from pathlib import Path
import sys

# Location of config.py
CONFIG_PATH = os.path.abspath(__file__)
PROJECT_PATH = Path(__file__).parent.parent
ALL_OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

# cloud lab configs
REMOTE_USER = "bcyl2"
CLOUDLAB_EXPERIMENT_NAME = "remote"
CLOUDLAB_PROJECT_NAME = "fair-prescrip"
CLOUDLAB_NODES = ['node0', 'node1', 'node2']
CLOUDLAB_HOST_SUFFIX = 'pg0.utah.cloudlab.us'


class Config:
    def __init__(self, algo_name, start_script, node_name):
        self.algo_name = algo_name
        self.start_script = 'python3 FairPrescriptionRules/%s' % (start_script)
        self.remote_hostname = "%s.%s.%s-%s" % (
            node_name, CLOUDLAB_EXPERIMENT_NAME, CLOUDLAB_PROJECT_NAME, CLOUDLAB_HOST_SUFFIX)
        
    


# SO Config
SO_CONFIG = [Config('Greedy', 'greedy.py', CLOUDLAB_NODES[0]),
             Config('CauSumX', 'causumx.py', CLOUDLAB_NODES[1])]
