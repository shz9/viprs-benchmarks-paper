#!/bin/bash

echo "Setting up the environment for PRScs..."

PRSCS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

module load python/3.10

python -m venv $PRSCS_PATH/PRScs_env/
source $PRSCS_PATH/PRScs_env/bin/activate
python -m pip install --upgrade pip
python -m pip install scipy h5py numpy pandas

# Clone the PRScs repository:
mkdir -p "$PRSCS_PATH/bin/PRScs/"
git clone https://github.com/getian107/PRScs.git "$PRSCS_PATH/bin/PRScs/"

# Test that it works:

prscs_bin="$PRSCS_PATH/bin/PRScs/PRScs.py"
python $prscs_bin --help

# ==============================================================================

# Download the LD data for European samples:

echo "Downloading the LD data for European samples..."

mkdir -p "$PRSCS_PATH/data/ld" || true
wget https://www.dropbox.com/s/t9opx2ty6ucrpib/ldblk_ukbb_eur.tar.gz -O "$PRSCS_PATH/data/ld/ldblk_ukbb_eur.tar.gz"

# untar/unzip the file:
tar -xvf "$PRSCS_PATH/data/ld/ldblk_ukbb_eur.tar.gz" -C "$PRSCS_PATH/data/ld/"
