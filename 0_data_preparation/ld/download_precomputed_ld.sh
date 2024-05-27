#!/bin/bash

# Make the directory where the pre-computed LD matrices will
# be downloaded and stored:

mkdir -p data/ld/eur/old_format/

# Download the pre-computed LD matrices for the UK Biobank data:
# and Untar/extract the downloaded files:

for chrom in {1..22}
do

  # Download from Zenodo:
  wget -P data/ld/eur/old_format/ukbb_50k_windowed/ \
    "https://zenodo.org/records/7036625/files/chr_$chrom.tar.gz?download=1"

  # Remove the ?download=1 from the name:
  mv "data/ld/eur/old_format/ukbb_50k_windowed/chr_$chrom.tar.gz?download=1" \
    "data/ld/eur/old_format/ukbb_50k_windowed/chr_$chrom.tar.gz"

  # Extract the tar.gz file:
  tar -xzf "data/ld/eur/old_format/ukbb_50k_windowed/chr_$chrom.tar.gz" \
    -C "data/ld/eur/old_format/ukbb_50k_windowed/"

  # Remove the tar.gz file after extracting it:
  rm "data/ld/eur/old_format/ukbb_50k_windowed/chr_$chrom.tar.gz"

done
