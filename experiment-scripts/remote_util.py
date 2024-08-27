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
import logging
import subprocess
import os
import sys
from exmpt_config import REMOTE_USER, REMOTE_HOSTNAMES, PROJECT_PATH


def prep_remote_cmd(command: str, remote_usr: str = REMOTE_USER, remote_host: str = REMOTE_HOSTNAMES[0]) -> str:
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
                         '%s@%s' % (remote_usr, remote_host), command, ]
    return remote_cmd_tokens


def run_remote_cmd_sync(command: str, remote_usr: str = REMOTE_USER, remote_host: str = REMOTE_HOSTNAMES[0]) -> str:
    """_summary_
    Execute command on remote_usr@remote_host right away
    Command example:
        - Experiment script that should be run one by one

    Args:
        command (str): bash script to run at the remote server
        remote_usr (str): remote user name
        remote_host (str): remote host name

    Returns:
        stdout (str)
    """

    logging.info("Execute remote command: `%s` synchronously " % command)

    return subprocess.run(prep_remote_cmd(command, remote_usr, remote_host),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, universal_newlines=True).stdout


def run_remote_cmd_async(command: str, remote_usr: str = REMOTE_USER, remote_host: str = REMOTE_HOSTNAMES[0]):
    """_summary_
    Execute command on remote_usr@remote_host right away. Execution can finish at anytime. Use this function when remote commands can be executed in parallel

    Command example:
        - Download data from remote server

    Args:
        command (str): bash script to run at the remote server
        remote_usr (str): remote user name
        remote_host (str): remote host name

    Returns:
        subprocess.Popen[bytes]: _description_
    """
    command = '(%s) >& /dev/null & exit' % command

    logging.info("Execute remote command: (%s) asynchronously " % command)

    return subprocess.Popen(prep_remote_cmd(command, remote_usr, remote_host))


def cp_local_to_remote(local_path='', remote_path='~', remote_usr: str = REMOTE_USER, remote_host: str = REMOTE_HOSTNAMES[0], exclude_paths=[]):
    logging.info("Synching {%s} to {%s@%s:%s}" %
                 (local_path, remote_usr, remote_host, remote_path))
    args = ["rsync", "-ar", "-e", "ssh", local_path,
            '%s@%s:%s' % (remote_usr, remote_host, remote_path)]
    if exclude_paths is not None:
        for i in range(len(exclude_paths)):
            args.append('--exclude')
            args.append(exclude_paths[i])

    return subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cp_remote_to_local(remote_path, local_path='./', remote_usr: str = REMOTE_USER, remote_host: str = REMOTE_HOSTNAMES[0]):
    # Make a directory for local, replace if already exists
    os.makedirs(local_path, exist_ok=True)
    tar_file = 'logs.tar'
    tar_file_path = os.path.join(remote_path, tar_file)
    run_remote_cmd_async('tar -C %s -cf %s .' % (remote_path,
                                                 tar_file_path), remote_usr, remote_host)
    subprocess.call(["scp", "-r", "-p", '%s@%s:%s' %
                    (remote_usr, remote_host, tar_file_path), local_path])
    subprocess.call(['tar', '-xf', os.path.join(local_path, tar_file),
                     '-C', local_path])
    subprocess.call(['rm', '-rf', os.path.join(local_path, tar_file)])


def synch_repo_at_remote(remote_host: str):
    # Returns a status code: 0 means success; non-0 mean failure
    return cp_local_to_remote(local_path=PROJECT_PATH,
                              remote_host=remote_host, exclude_paths=['*/__pycache__', 'output'])


def clean_up(remote_host):
    commands = [
        "pkill -9 python;", "rm -r output;", "mkdir output"
    ]
    return run_remote_cmd_sync(command=' '.join(commands), remote_host=remote_host)


def run_algorithm(algorithm_cmd: str, remote_host: str):
    # Cat all stdout and stderr on remote machine
    flags = ' &> output/stdout.log 2> output/stderr.log'

    # Returns a status code: 0 means success; non-0 mean failure
    return run_remote_cmd_sync(command=algorithm_cmd + flags, remote_host=remote_host)
