#!/bin/bash

echo "> Launching jobs for benchmarking the total runtime..."
# Create logging directory:
mkdir -p ./log/analysis/total_runtime_benchmarks/

# Specify options for the new version of VIPRS:
ld_dtype=("int8")
threads=(2)
jobs=(1)
low_mem=("true")
dequantize=("false")

# Loop over the 5 training folds and launch the benchmarking jobs for
# the new and old versions of VIPRS:

for fold in {1..5}
do

  # Run the new version of VIPRS:
  for ld in "${ld_dtype[@]}"
  do
    for j in "${jobs[@]}"
    do

      # If ld type does not equal int8 and job > 1, then skip:
      if [ "$ld" != "int8" ] && [ "$j" -gt 1 ]; then
        continue
      fi

      for t in "${threads[@]}"
      do

        for m in "${low_mem[@]}"
        do

          # If low_mem is is true and threads = 1, loop over the dequantize value; otherwise, keep quantize as false:
          if [ "$m" = "true" ] && [ "$t" -eq 1 ]; then
            for q in "${dequantize[@]}"
            do
              model_id="l${ld}_m${m}_q${q}_t${t}_j${j}"
              sbatch -J "new_viprs_bma_fold_${fold}_${model_id}" 1_benchmarks/total_runtime_benchmarks/test_viprs_bma/benchmark_new_viprs_bma.sh "fold_$fold" "$ld" "$t" "$j" "$m" "$q"
            done
          else
            model_id="l${ld}_m${m}_qfalse_t${t}_j${j}"
            sbatch -J "new_viprs_bma_fold_${fold}_${model_id}" 1_benchmarks/total_runtime_benchmarks/test_viprs_bma/benchmark_new_viprs_bma.sh "fold_$fold" "$ld" "$t" "$j" "$m" "false"
          fi

        done
      done
    done
  done
done