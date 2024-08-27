"""_summary_
This script launch an experiment, runs both Greedy and CauSumX where X will be the first argument (default: 4; max: 7). 
@author: Benton Li
@email: cl2597@cornell.edu
"""

# import utils
import logging
from expmt_util import run_single_remote_exmpt, ts_prefix
from remote_util import synch_repo_at_remote, run_algorithm
import concurrent.futures
import os
import subprocess
import sys
from exmpt_config import PROJECT_PATH, ALL_OUTPUT_PATH


# TODO better Experiment config spec
expmt_configs = [['python3 FairPrescriptionRules/greedy.py',
                  "", "node0.remote.fair-prescrip-pg0.utah.cloudlab.us"]]

logging.basicConfig(level=logging.DEBUG)

def main():
    """

    for each model in the model list and a provided dataset path, submit a request 
    """
    print("start")

    # Every experiment performed will have a time stamped output directory.
    # This directory contains the remote output directory
    # Each remote output directory has the following:
    #   1. stdout.log
    #   2. stderr.log
    #   3. experiment results

    # Prepare a output directory
    os.makedirs(os.path.join(PROJECT_PATH, 'output', ts_prefix()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        f = []
        for expmt_config in expmt_configs:
            # Synch codebase -> run algo -> pull result
            f.append(executor.submit(
                run_single_remote_exmpt, expmt_config, executor))

        for i in f:
            print(i.result())

    print("done")


if __name__ == "__main__":
    main()
