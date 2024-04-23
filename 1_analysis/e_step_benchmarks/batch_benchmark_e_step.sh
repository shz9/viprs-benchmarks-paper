#!/bin/bash

echo "> Launching per-chromosome jobs for benchmarking the E-Step..."
# Create the relevant logging directory:
mkdir -p ./log/analysis/e_step_benchmarks/

# Loop over the chromosomes and launch the benchmarking script for
# each one as a SLURM batch job:

for chrom in {1..22}
do
  sbatch -J "chr_$chrom" 1_analysis/e_step_benchmarks/benchmark_e_step_job.sh "$chrom"
done
