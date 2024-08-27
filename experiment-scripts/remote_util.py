""" 
    Author: Benton Li (cl2597@cornell.edu)

    This file contains helper functions for actions related to remote machine, 
    including:
    - Copy local files to remote machine
    - Execute scripts at remote machine
    - Download data from remote machine
"""
import subprocess
import os
import logging
import config

logging.basicConfig(level=logging.WARNING)
def prep_remote_cmd(command: str, remote_usr:str = config.REMOTE_USER, remote_host:str = config.REMOTE_HOSTNAMES[0]) -> str:
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
        '%s@%s' % (remote_usr, remote_host), command]
    return remote_cmd_tokens

def run_remote_cmd_sync(command: str, remote_usr:str = config.REMOTE_USER, remote_host:str = config.REMOTE_HOSTNAMES[0] ) -> str:
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
        stdout=subprocess.PIPE, universal_newlines=True).stdout

def run_remote_cmd_async(command: str, remote_usr:str = config.REMOTE_USER, remote_host:str = config.REMOTE_HOSTNAMES[0]):
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


def cp_local_to_remote(local_path='', remote_path='~', remote_usr:str = config.REMOTE_USER, remote_host:str = config.REMOTE_HOSTNAMES[0], exclude_paths=[]):
    logging.info("Synching {%s} to {%s@%s:%s}" % (local_path, remote_usr, remote_host, remote_path))
    args = ["rsync", "-aPr", "-e", "ssh", local_path,
        '%s@%s:%s' % (remote_usr, remote_host, remote_path)]
    if exclude_paths is not None:
        for i in range(len(exclude_paths)):
            args.append('--exclude')
            args.append(exclude_paths[i])
    status:int = subprocess.call(args)
    # call returns a status code: 0 means successful


    if status != 0:
        logging.warning("Failed to synch {%s} to {%s@%s:%s}. rsynch returns  status code %d" % (local_path, remote_usr, remote_host, remote_path, status))
    return status

def cp_remote_to_local(remote_path, local_path='./', remote_usr:str = config.REMOTE_USER, remote_host:str = config.REMOTE_HOSTNAMES[0]):
    # Make a directory for local, replace if already exists
    os.makedirs(local_path, exist_ok=True)
    logging.info(local_path)
    tar_file = 'logs.tar'
    tar_file_path = os.path.join(remote_path, tar_file)
    run_remote_cmd_async('tar -C %s -cf %s .' % (remote_path,
        tar_file_path), remote_usr, remote_host)
    subprocess.call(["scp", "-r", "-p", '%s@%s:%s' % (remote_usr, remote_host, tar_file_path), local_path])
    subprocess.call(['tar', '-xf', os.path.join(local_path, tar_file),
        '-C', local_path])
    subprocess.call(['rm', '-rf', os.path.join(local_path, tar_file)])

def synch_repo_at_remote(remote_host: str):
    # Returns a status code: 0 means success; non-0 mean failure
    return cp_local_to_remote(local_path=config.PROJECT_PATH, 
                       remote_host=remote_host, exclude_paths=['*/__pycache__']) 

def run_algorithm(algorithm_cmd: str, remote_host: str):
    # Returns a status code: 0 means success; non-0 mean failure
    return run_remote_cmd_sync(command=algorithm_cmd, remote_host=remote_host)

