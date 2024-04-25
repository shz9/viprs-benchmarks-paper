#!/bin/bash
# This script creates the python environment for running the scripts
# on the cluster.

mkdir -p env

module load StdEnv/2020
module load python/3.8

# Create environment with latest version of VIPRS:
python -m venv env/viprs/
source env/viprs/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Deactivate the environment:
deactivate

# Create environment with old version of VIPRS:
python -m venv env/viprs-old/
source env/viprs-old/bin/activate
python -m pip install --upgrade pip
python -m pip install magenpy==0.0.12
python -m pip install viprs==0.0.4
