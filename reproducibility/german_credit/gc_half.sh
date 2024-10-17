#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/german_credit/config_half.json gc/remote_all.json




