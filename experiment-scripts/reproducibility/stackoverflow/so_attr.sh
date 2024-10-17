#!/usr/bin/bash
pkill -f ssh
python ../run_experiment.py stackoverflow/config_lo_attrI.json so/remote_lo_attrI.json &
python ../run_experiment.py stackoverflow/config_lo_attrM.json so/remote_lo_attrM.json &
python ../run_experiment.py stackoverflow/config_lo_attrIM.json so/remote_lo_attrIM.json




