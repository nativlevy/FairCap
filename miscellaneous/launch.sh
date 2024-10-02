#!/bin/bash

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p bcyl2@node0.remote.fair-prescrip-pg0.utah.cloudlab.us git clone https://github.com/bentondecusin/FairPrescriptionRules 

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p bcyl2@node0.remote.fair-prescrip-pg0.utah.cloudlab.us git config --global user.name bcyl2

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p bcyl2@node0.remote.fair-prescrip-pg0.utah.cloudlab.us git config --global user.email cl2597@cornell.edu

ssh -o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPersist=2m -o ControlPath=~/.ssh/cm-%r@%h:%p bcyl2@node0.remote.fair-prescrip-pg0.utah.cloudlab.us sudo apt-get install htop