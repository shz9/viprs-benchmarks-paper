#!/bin/bash

echo "> Launching jobs for benchmarking the total runtime..."
# Create logging directory:
mkdir -p ./log/analysis/total_runtime_benchmarks/

# Specify options for the new version of VIPRS:
ld_dtype=("int8" "int16" "float32" "float64")
threads=(1 2 4)
jobs=(1 2 4)

# Loop over the 5 training folds and launch the benchmarking jobs for
# the new and old versions of VIPRS:

for fold in {1..5}
do

  # Run the new version of VIPRS:
  for ld in "${ld_dtype[@]}"
  do
    for t in "${threads[@]}"
    do
      for j in "${jobs[@]}"
      do
        model_id="l${ld}_t${t}_j${j}"
        sbatch -J "new_viprs_fold_${fold}_${model_id}" 1_analysis/total_runtime_benchmarks/benchmark_new_viprs.sh "fold_$fold" "$ld" "$t" "$j"
      done
    done
  done

  # Run sad old VIPRS with its single default setting:
  sbatch -J "old_viprs_fold_$fold" 1_analysis/total_runtime_benchmarks/benchmark_old_viprs.sh "fold_$fold"

done
