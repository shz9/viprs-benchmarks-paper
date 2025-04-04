#!/bin/bash
# This is a setup script to prepare the R environment
# for running LDpred2.

LDPRED2_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading required input data for LDpred2..."

# Create the data directory:
mkdir -p "$LDPRED2_PATH/data" || true

# Install gdown to facilitate downloading the LD data
pip install --user gdown
gdown 17dyKGA2PZjMsivlYb_AjDmuZjM1RIGvs -O "$LDPRED2_PATH/data/ldref_hm3_plus.zip"

# Unzip the LD data:
mkdir -p "$LDPRED2_PATH/data/ld/" || true
unzip "$LDPRED2_PATH/data/ldref_hm3_plus.zip" -d "$LDPRED2_PATH/data/ld/"

# Download LD metadata:
wget "https://figshare.com/ndownloader/files/37802721" -O "$LDPRED2_PATH/data/ld/map_hm3_plus.rds"

# ---------------------------------------------------------------------

echo "Setting up the environment for LDpred2..."

# Setup the R environment:
module load gcc/12.3 r/4.3.1
mkdir -p "$LDPRED2_PATH/R_ldpred2_env" || true

export R_LIBS="$LDPRED2_PATH/R_ldpred2_env"

R -e 'install.packages("bigsnpr", repos="https://cloud.r-project.org/")'
R -e 'install.packages("dplyr", repos="https://cloud.r-project.org/")'
R -e 'install.packages("tidyr", repos="https://cloud.r-project.org/")'

echo "Done!"
