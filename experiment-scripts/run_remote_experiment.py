"""_summary_
This script launch an experiment, runs both Greedy and CauSumX where X will be the first argument (default: 4; max: 7). 
@author: Benton Li
@email: cl2597@cornell.edu
"""

# import utils
import logging
import subprocess
import sys

import concurrent.futures
from remote_util import synch_repo_at_remote, run_algorithm

# TODO better Experiment config spec
expmt_configs = [['python3 FairPrescriptionRules/greedy.py',
                  "", "node0.remote.fair-prescrip-pg0.utah.cloudlab.us"]]
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


def run_single_remote_exmpt(expmt_config, executor):
    # Return 0 if success; 1 otherwise
    # Attempt to synch codebase; future will be done EVENTUALLY as rsynch always returns a status code.
    synch_future = executor.submit(synch_repo_at_remote, expmt_config[2])
    if synch_future.result() != 0:
        return 1

    run_algo_future = executor.submit(
        run_algorithm, expmt_config[0], expmt_config[2])
    if run_algo_future.result() != 0:
        return 1
    return 0


def main():
    """
    for each model in the model list and a provided dataset path, submit a request 
    """
    print("start")

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
