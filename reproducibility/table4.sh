#!/usr/bin/bash
pkill -f ssh
python ../experiment-scripts/run_experiment.py ../data/stackoverflow/config.json remote_full.json




