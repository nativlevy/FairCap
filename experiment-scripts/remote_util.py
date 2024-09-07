"""
    Author: Benton Li (cl2597@cornell.edu)

    This file contains helper functions for actions related to remote machine,
    including:
    - Copy local files to remote machine
    - Execute scripts at remote machine
    - Download data from remote machine
    N.b. All functions in this file are expected to run on local machine
    N.b. We suppress stderr and stdout at our local machine. Errors and outputs will be logged in files at remote machine.
"""
from ast import Dict
import json
import logging
from logging import config
import subprocess
import os
import sys
from exmpt_config import CONTROLLER_OUTPUT_PATH, REMOTE_USER, PROJECT_PATH, DATA_PATH




def prep_remote_cmd(command: str, config: Dict) -> str:
    """_summary_
    Prepare a bash script to be run command on remote_usr@remote_host
    Example:
    >>> prep_remote_cmd("echo helloworld", "ben", "ms.com")
    ouput: 'ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p ben@ms.com echo helloworld'

    Args:
        command (str): bash script to run at the remote server
        remote_usr (str): remote user name
        remote_host (str): remote host name

    Returns:
        str: a full ssh script that that runs command on remote sever

    """
    remote_cmd_tokens = ['ssh', '-o', 'StrictHostKeyChecking=no',
                         '-o', 'ControlMaster=auto',
                         '-o', 'ControlPersist=2m',
                         '-o', 'ControlPath=~/.ssh/cm-%r@%h:%p',
                         '%s@%s' % (config["_remote_user"], config["_remote_host"]), command]
    return remote_cmd_tokens


def run_remote_cmd_sync(command: str, config: Dict) -> str:
    """_summary_
    Execute command on remote_usr@remote_host right away
    Command example:
        - Experiment script that should be run one by one

    Args:
        command (str): bash script to run at the remote server
        config (Dict): configuration

    Returns:
        stdout (str)
    """

    logging.info("Execute remote command: `%s` synchronously " % command)

    # return subprocess.run(prep_remote_cmd(command, config),
    #                       stderr=subprocess.DEVNULL, universal_newlines=True).returncode
    return subprocess.run(prep_remote_cmd(command, config),universal_newlines=True).returncode


def run_remote_cmd_async(command: str, config: Dict):
    """_summary_
    Execute command on remote_usr@remote_host right away. Execution can finish at anytime. Use this function when remote commands can be executed in parallel

    Command example:
        - Download data from remote server

    Args:
        command (str): bash script to run at the remote server
        config (Dict): configuration

    Returns:
        subprocess.Popen[bytes]: _description_
    """
    command = '(%s) >& /dev/null & exit' % command

    logging.info("Execute remote command: (%s) asynchronously " % command)

    return subprocess.Popen(prep_remote_cmd(command, config))


def cp_local_to_remote(config, local_path, remote_path = '~', exclude_paths=[]):
    remote_user, remote_host = config['_remote_user'], config['_remote_host']
    logging.info("Synching {%s} to {%s@%s:%s}" %
                 (local_path, remote_user, remote_host, remote_path))
    args = ["rsync", "-ar", "-e", "ssh", local_path,
            '%s@%s:%s' % (remote_user, remote_host, remote_path)]
    if exclude_paths is not None:
        for i in range(len(exclude_paths)):
            args.append('--exclude')
            args.append(exclude_paths[i])

    return subprocess.call(args, stderr=subprocess.DEVNULL)


def fetch_logs_from_remote(config):
    """_summary_
        Copy remote:output to PROJECT_PATH/output/algo_name
    """
    # Make a directory for local, replace if already exists
    local_output_path = config['_control_output_path']
    os.makedirs(local_output_path, exist_ok=True)
    tar_file = 'logs.tar'
    tar_file_path = os.path.join(config['_output_path'], tar_file)
    # tarball remote file
    run_remote_cmd_async('tar -C %s -cf %s .' % (config['_output_path'],
                                                 tar_file_path), config)
    # Copy to local
    subprocess.call(["scp", "-r", "-p", '%s@%s:%s' %
                    (config['_remote_user'], config['_remote_host'], tar_file_path), local_output_path])

    # Extract at local
    subprocess.call(['tar', '-xf', os.path.join(local_output_path, tar_file),
                     '-C', local_output_path])
    # Remove tar
    subprocess.call(['rm', '-rf', os.path.join(local_output_path, tar_file)])
    return 0


def synch_repo_at_remote(config):
    # Returns a status code: 0 means success; non-0 mean failure
    return cp_local_to_remote(config, local_path=PROJECT_PATH, exclude_paths=['*/__pycache__', 'output', 'venv', 'lib', 'include', 'bin'])


def clean_up(config):
    commands = [
        "pkill -9 python;", "rm -fr %s;" % config['_output_path'], "mkdir -p %s" %  config['_output_path']
    ]
    return run_remote_cmd_sync(config=config, command=' '.join(commands))



def run_algorithm(config):
    arg_flag =  "\'{}\' {} ".format(json.dumps(config), config['_output_path'])
    # Cat all stdout and stderr on remote machine
    output_flag = ' &> {}/stdout.log 2> {}/stderr.log'.format(config['_output_path'], config['_output_path'])

    # TODO embed env into cloudlab
    algorithm_cmd = "python3 FairPrescriptionRules/src/models/{} {} {}".format(config["_start"], arg_flag, output_flag)

    # Returns a status code: 0 means success; non-0 mean failure
    return run_remote_cmd_sync(command=algorithm_cmd, config=config)
