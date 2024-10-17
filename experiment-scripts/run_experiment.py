"""_summary_
This script launch an experiment, runs both Greedy and CauSumX where X will be the first argument (default: 4; max: 7).
@author: Benton Li
@email: cl2597@cornell.edu
"""

# import utils
import json
import logging

from expmt_util import run_single_local_expmt, run_single_remote_expmt
import concurrent.futures
import os
import subprocess
import sys
from expmt_config import (
    PROJECT_PATH,
    DATA_PATH,
    REPO_NAME,
    CONFIG_PATH,
    WORKER_OUTPUT_PATH,
)
import argparse

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(data_config_path, expmt_config_path):
    """
    Input:
        argv[1]: data configuration path
        argv[2]: experiment configuration path
    """

    # --------------------------- LOAD CONFIGS --------------------------------
    # Expmt config
    if not os.path.isfile(expmt_config_path):
        expmt_config_path = os.path.join(CONFIG_PATH, expmt_config_path)
    with open(os.path.join(CONFIG_PATH, expmt_config_path)) as json_file:
        expmt_config = json.load(json_file)

    is_remote = expmt_config["_is_remote"]

    # Data config
    if is_remote:
        data_config_path = os.path.join(REPO_NAME, "data", data_config_path)
    elif not os.path.isfile(data_config_path):
        data_config_path = os.path.join(DATA_PATH, data_config_path)

    models = expmt_config["_models"]
    # Prepare a output directory, prefixed with time stamp
    expmt_title = expmt_config["_expmt_title"]
    print("BEGIN")
    # TODO add me back
    # os.makedirs(os.path.join(CONTROLLER_OUTPUT_PATH, tempore))
    if is_remote:
        remote_nodes = expmt_config["_cloudlab_nodes"]
        remote_postfix = expmt_config["_cloudlab_postfix"]
        remote_username = expmt_config["_cloudlab_user"]

        if len(remote_nodes) < len(models):
            raise AssertionError(
                "%d nodes required; %d provided" % (len(models), len(remote_nodes))
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Every experiment performed will have a time stamped output directory.
            f = []

            for i in range(len(models)):
                model = models[i]
                remote_host = "{}.{}".format(remote_nodes[i], remote_postfix)
                config = {
                    "_model": model,
                    "_data_config_path": data_config_path,
                    "_output_path": os.path.join(WORKER_OUTPUT_PATH, model["_name"]),
                    "_control_output_path": os.path.join(
                        PROJECT_PATH, "output", expmt_title, str(i) + model["_name"]
                    ),
                    "_remote_host": remote_host,
                    "_remote_user": remote_username,
                }
                f.append(executor.submit(run_single_remote_expmt, config))
                # run_single_remote_expmt(config)
        for i in f:
            print(i.result())
        print("DONE")
        return
    else:
        with open(data_config_path) as json_file:
            data_config = json.load(json_file)
        for model in models:
            run_single_local_expmt(
                model,
                data_config,
                os.path.join(PROJECT_PATH, "output", expmt_title, model["_name"]),
            )
    print("FINISHED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main(sys.argv[1], sys.argv[2])
