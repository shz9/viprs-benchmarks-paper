#!/bin/bash
# This script creates the python environment for running the scripts
# on the cluster.
# NOTE: On the cluster were this work was done, there were issues installing
# both versions of viprs using the same python version, due to incompatability
# between dependencies and available python versions on the cluster. To work around this,
# the two versions of viprs were installed in separate environments, each with a different
# python version. This script sets up the two environments.

mkdir -p env

echo "========================================================"
echo "Setting up environment for newer version of viprs (v0.1)"

module load StdEnv/2020
module load python/3.10
python --version

# Create environment with latest version of VIPRS:
rm -rf env/viprs/
python -m venv env/viprs/
source env/viprs/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "========================================================"

# Deactivate the environment:
deactivate

echo "========================================================"
echo "Setting environment for older version of viprs (v0.0.4)"

module load StdEnv/2020
module load python/3.7

python --version

# Create environment with old version of VIPRS:
rm -rf env/viprs-old/
python -m venv env/viprs-old/
source env/viprs-old/bin/activate
python -m pip install --upgrade pip
python -m pip install magenpy==0.0.12
python -m pip install viprs==0.0.4

echo "========================================================"
echo "Done!"
