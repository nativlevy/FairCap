#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/stackoverflow/config_lo_attrI.json so/remote_lo_attrI.json &
python ../experiment-scripts/run_experiment.py ../data/stackoverflow/config_lo_attrM.json so/remote_lo_attrM.json &
python ../experiment-scripts/run_experiment.py ../data/stackoverflow/config_lo_attrIM.json so/remote_lo_attrIM.json




