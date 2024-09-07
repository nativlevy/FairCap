"""_summary_
This script launch an experiment, runs both Greedy and CauSumX where X will be the first argument (default: 4; max: 7).
@author: Benton Li
@email: cl2597@cornell.edu
"""

# import utils
import json
import logging

from expmt_util import run_single_local_exmpt, run_single_remote_exmpt, ts_prefix
from remote_util import synch_repo_at_remote, run_algorithm
import concurrent.futures
import os
import subprocess
import sys
from exmpt_config import PROJECT_PATH, DATA_PATH, CONTROLLER_OUTPUT_PATH, CONFIG_PATH, WORKER_OUTPUT_PATH

logging.basicConfig(level=logging.DEBUG)


def main(data_config_path, expmt_config_path):
    """
    Input:
        argv[1]: data configuration path
        argv[2]: experiment configuration path
    """

    # --------------------------- LOAD CONFIGS --------------------------------
    # Data config
    if not os.path.isfile(data_config_path):
        data_config_path = os.path.join(DATA_PATH, data_config_path)
    with open(data_config_path) as json_file:
        data_config = json.load(json_file)

    # Expmt config
    if not os.path.isfile(expmt_config_path):
        expmt_config_path = os.path.join(CONFIG_PATH, expmt_config_path)
    with open(os.path.join(CONFIG_PATH, expmt_config_path)) as json_file:
        exmpt_config = json.load(json_file)

    is_remote = exmpt_config['_is_remote']
    models = exmpt_config['_models']
    k = exmpt_config['_k']
    # Prepare a output directory, prefixed with time stamp
    tempore = ts_prefix()
    # TODO add me back
    # os.makedirs(os.path.join(CONTROLLER_OUTPUT_PATH, tempore))
    if is_remote:
        remote_nodes = exmpt_config['_cloudlab_nodes'] 
        remote_postfix = exmpt_config['_cloudlab_postfix'] 
        remote_username = exmpt_config['_cloudlab_user'] 

        if len(remote_nodes) < len(models):
            raise AssertionError("%d nodes required; %d provided" % (len(models), len(remote_nodes)))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Every experiment performed will have a time stamped output directory.
            f = []
            
            for i in range(len(models)):
                model = models[i]
                remote_host = "{}.{}".format(remote_nodes[i], remote_postfix)
                config = {**data_config, **model, **
                          {'_output_path':  os.path.join(WORKER_OUTPUT_PATH, model['_name']), '_control_output_path': os.path.join(PROJECT_PATH, 'output', tempore, model['_name']), '_k': k, '_remote_host': remote_host, '_remote_user': remote_username
                           }}
                f.append(executor.submit(
                    run_single_remote_exmpt, config))
                # run_single_remote_exmpt(config)

        
        print("done")
        return
    else:
        for model in models:
            config = {**data_config, **model, **
                      {'_output_path': os.path.join(
                          PROJECT_PATH, 'output', tempore, model['_name']), '_k': k}}
            run_single_local_exmpt(config)
    print("start")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
