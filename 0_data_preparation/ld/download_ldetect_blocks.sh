#!/bin/bash

# Create directory to store the LD block files:

mkdir -p data/ldetect_data/

# Download the LD blocks for European samples:

wget -O data/ldetect_data/EUR_blocks.bed https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed

# Download LD blocks for African samples:

wget -O data/ldetect_data/AFR_blocks.bed https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/AFR/fourier_ls-all.bed

# Download LD blocks for East Asian samples:

wget -O data/ldetect_data/EAS_blocks.bed https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/ASN/fourier_ls-all.bed


# Copy EAS boundaries for SAS:
cp data/ldetect_data/EAS_blocks.bed data/ldetect_data/CSA_blocks.bed
# Copy EUR boundaries for MID:
cp data/ldetect_data/EUR_blocks.bed data/ldetect_data/MID_blocks.bed
# Copy AFR boundaries for AMR:
cp data/ldetect_data/AFR_blocks.bed data/ldetect_data/AMR_blocks.bed
