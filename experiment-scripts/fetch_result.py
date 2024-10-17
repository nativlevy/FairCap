import os

# import utils
import json
import logging
import time

from expmt_util import ts_prefix

from remote_util import fetch_logs_from_remote
import concurrent.futures
import os
import subprocess
import sys
from expmt_config import (
    PROJECT_PATH,
    CONFIG_PATH,
    WORKER_OUTPUT_PATH,
)
import argparse

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(expmt_config_path):
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
    models = expmt_config["_models"]
    k = expmt_config["_k"]
    # Prepare a output directory, prefixed with time stamp
    tempore = ts_prefix()
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
                    **model,
                    **{
                        "_output_path": os.path.join(
                            WORKER_OUTPUT_PATH, model["_name"]
                        ),
                        "_control_output_path": os.path.join(
                            PROJECT_PATH, "output", tempore, model["_name"]
                        ),
                        "_k": k,
                        "_remote_host": remote_host,
                        "_remote_user": remote_username,
                    },
                }
                f.append(executor.submit(fetch_logs_from_remote, config))
                # run_single_remote_expmt(config)


if __name__ == "__main__":
    main(sys.argv[1])
