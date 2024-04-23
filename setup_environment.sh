#!/bin/bash
# This script creates the python environment for running the scripts
# on the cluster.

mkdir env

# Create environment with latest version of VIPRS:
module load python/3.8
python3 -m venv env/viprs/
python3 -m pip install -r requirements.txt

# Create environment with old version of VIPRS:
python3 -m venv env/viprs-old/
python3 -m pip install magenpy==0.0.12
python3 -m pip install viprs==0.0.4
