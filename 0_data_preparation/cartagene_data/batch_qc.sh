#!/bin/bash

mkdir -p ./log/data_preparation/cartagene_genotypes/

for c in $(seq 1 22)
do
  sbatch -J "chr_$c" 0_data_preparation/cartagene_data/cartagene_qc_job.sh "$c"
done

