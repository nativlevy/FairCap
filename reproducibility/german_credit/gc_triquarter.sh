#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/german_credit/config_triquarter.json gc/remote_all.json




