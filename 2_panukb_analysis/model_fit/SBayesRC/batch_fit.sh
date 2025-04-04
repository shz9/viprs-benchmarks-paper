#!/bin/bash

ld_reference=${1:-"HapMap3"}

declare -A time_dict
time_dict["HapMap3"]="01:30:00"
time_dict["HapMap3-hq"]="01:30:00"
time_dict["HapMap3-hq-4cM"]="01:30:00"
time_dict["HapMap3-x-4cM"]="01:30:00"
time_dict["7m"]="10:00:00"

# Create the logging directory:
mkdir -p "./log/model_fit/panukb_sumstats/external/SBayesRC-${ld_reference}/EUR/"

for sumstats_file in data/sumstats/panukb_sumstats/EUR/*.sumstats.gz
do

  phenotype=$(basename "$sumstats_file" | sed 's/\.sumstats\.gz//g')

  sbatch -J "SBayesRC-${ld_reference}/EUR/$phenotype" --time "${time_dict[$ld_reference]}" \
                  2_panukb_analysis/model_fit/SBayesRC/sbayesrc_job.sh \
                  "$sumstats_file" \
                  "$ld_reference"

done