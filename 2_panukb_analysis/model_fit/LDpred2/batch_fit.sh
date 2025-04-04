#!/bin/bash

# Create the logging directory:
mkdir -p "./log/model_fit/panukb_sumstats/external/LDpred2-auto/EUR/"

for sumstats_file in data/sumstats/panukb_sumstats/EUR/*.sumstats.gz
do

  phenotype=$(basename "$sumstats_file" | sed 's/\.sumstats\.gz//g')

  sbatch -J "$phenotype" 2_panukb_analysis/model_fit/LDpred2/ldpred2_job.sh "$sumstats_file"

done

