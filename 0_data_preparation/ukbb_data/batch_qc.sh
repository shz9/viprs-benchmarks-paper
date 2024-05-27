#!/bin/bash

mkdir -p ./log/data_preparation/genotypes/

for c in $(seq 1 22)
do
  sbatch -J "chr_$c" 0_data_preparation/ukbb_data/ukbb_qc_job.sh "$c"
done

