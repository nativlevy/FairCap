#!/usr/bin/bash
pkill -f ssh
python ../run_experiment.py german_credit/config_lo_attrI.json gc/remote_lo_attrI.json &
python ../run_experiment.py german_credit/config_lo_attrM.json gc/remote_lo_attrM.json &
python ../run_experiment.py german_credit/config_lo_attrIM.json gc/remote_lo_attrIM.json




