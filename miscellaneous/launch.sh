#!/bin/bash

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p username@node0.remote.fair-prescrip-pg0.utah.cloudlab.us git clone <repo_url>

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p username@node0.remote.fair-prescrip-pg0.utah.cloudlab.us git config --global user.name <username>

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p username@node0.remote.fair-prescrip-pg0.utah.cloudlab.us git config --global user.email <email>

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p username@node0.remote.fair-prescrip-pg0.utah.cloudlab.us sudo apt-get install htop