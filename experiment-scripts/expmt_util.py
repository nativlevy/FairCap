
from concurrent import futures
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from remote_util import clean_up, fetch_logs_from_remote, run_algorithm, synch_repo_at_remote
import cProfile


from exmpt_config import PROJECT_PATH, Config
sys.path.append(os.path.join(PROJECT_PATH, 'src/baselines'))
sys.path.append(os.path.join(PROJECT_PATH, 'src/baselines/common'))
import greedy  # NOQA
import causumx  # NOQA




def summarize_exp_data(remote_exp_dir, local_out_dir, executor, is_exp_remote=False):
    # Gather csv outputs from remote servers

    if is_exp_remote:
        # TODO download data from remote
        # TODO build stderr & stdout pipeline
        pass
    # TODO make some charts
    # TODO make some plots
    # TODO make some slides
    return local_out_dir


# def run_single_exmpt(config):
#     """
#     If run remotely (mainly for production purpose), we run this very
#     experiment on a designated remote server.
#     Local machine will schedule a chain of callables to do the following
#     1. Synchronize codebase at remote machine. If the remote machine does not
#     have the codebase, the whole codebase will be copied over ssh
#     2. Upon successful synchronization, local machine will run the experiment
#     script on the remote machine via ssh, i.e, at this point, the entire model
#     logics will be executed on the remote server. N.b. That server spends all
#     its resources just to run the experiment.
#     3. Upon successful execution of the model/algorithm, the thread will be
#     flagged as successful. Then we copy the results to our local machine via
#     scp.
#     4. Finally, we mark this future as completed. The ThreadPool will carry out
#     the next step: data analysis, creating tables, plotting charts, etc.
#     """
#     is_remote = config['_is_remote']
#     if is_remote:
#         return (config)
#     else:
#         return run_single_local_exmpt(config)


def run_single_local_exmpt(config):
    algo_name = config["_name"]
    Path(config["_output_path"]).mkdir(parents=True, exist_ok=True)
    # TODO hoist
    greedy.main(config)

    # elif algo_name == 'causumx':
    #     causumx.main(config)

    return 0


def run_single_remote_exmpt(config):
    """
    Remote experiments are mainly for production purpose, we run this very
    experiment on a designated remote server.
    Local machine will schedule a chain of callables to do the following
    1. Synchronize codebase at remote machine. If the remote machine does not
    have the codebase, the whole codebase will be copied over ssh
    2. Upon successful synchronization, local machine will run the experiment
    script on the remote machine via ssh, i.e, at this point, the entire model
    logics will be executed on the remote server. N.b. That server spends all
    its resources just to run the experiment.
    3. Upon successful execution of the model/algorithm, the thread will be
    flagged as successful. Then we copy the results to our local machine via
    scp.
    4. Finally, we mark this future as completed. The ThreadPool will carry out
    the next step: data analysis, creating tables, plotting charts, etc.
    """
    remote_host = config['_remote_host']
    algo_name = config['_name']
    # Step 0: Kill previous experiment
    clean_up(config)

    # Step1: Attempt to synch codebase; future will be done EVENTUALLY as rsynch always returns a status code. rsynch returns 0 if success, 255 otherwise
    synch_status = synch_repo_at_remote(config)
    if synch_status != 0:
        logging.error("Failed to synch codebase at node %s. rsynch returns  status code %d" % (
            remote_host, synch_status))
        # If fail to synch, abort this experiment. No logs fetched
        return 1
    else:
        logging.info("Finished synching the codebase at node %s " % (
            remote_host))
    # Step 2. Attempt to run code logic
    run_algo_status = run_algorithm(config)
    # Per implementation, `run_algorithm`s return 0 if success, 1 otherwise
    if run_algo_status != 0:
        logging.error("Error(%s) occurred at %s. See more details at %s's stderr.log" % (
            run_algo_status, remote_host, remote_host))
    else:
        logging.info("Algo %s completed running on %s" % (
            algo_name, remote_host))

    # Future 3. Fetch all the outputs, including logs, experiment results
    # Even if the model failed on remote machines, we still fetch the logs
    fetch_status = fetch_logs_from_remote(
        config)
    if fetch_status != 0:
        logging.error("Failed to fetch outputs from  %s" % (
            remote_host))
    else:
        logging.info("Algo %s output fetched from %s" %
                     (algo_name, remote_host))

    return run_algo_status


def ts_prefix():
    tempore = time.strftime('%m-%d/%H:%M',
                            time.localtime())
    return tempore
