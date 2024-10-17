#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/german_credit/config_lo_attrI.json gc/remote_lo_attrI.json &
python ../experiment-scripts/run_experiment.py ../data/german_credit/config_lo_attrM.json gc/remote_lo_attrM.json &
python ../experiment-scripts/run_experiment.py ../data/german_credit/config_lo_attrIM.json gc/remote_lo_attrIM.json




