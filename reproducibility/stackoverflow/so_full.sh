#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/stackoverflow/config.json so/remote_full.json




