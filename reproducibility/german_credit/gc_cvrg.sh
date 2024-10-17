#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/german_credit/config.json gc/remote_cvrg.json




