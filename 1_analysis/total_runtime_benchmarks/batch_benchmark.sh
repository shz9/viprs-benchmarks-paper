#!/bin/bash

# Loop over the 5 training folds and launch the benchmarking jobs for
# the new and old versions of VIPRS:

for fold in {1..5}
do
  sbatch -J "new_viprs_fold_$fold" 1_analysis/total_runtime_benchmarks/benchmark_new_viprs.sh "fold_$fold"
  sbatch -J "old_viprs_fold_$fold" 1_analysis/total_runtime_benchmarks/benchmark_old_viprs.sh "fold_$fold"
done
