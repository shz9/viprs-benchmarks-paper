#!/bin/bash

echo "> Launching jobs for benchmarking the total runtime..."
# Create logging directory:
mkdir -p ./log/analysis/total_runtime_benchmarks/

# Specify options for the new version of VIPRS:
ld_dtype=("int8" "int16" "float32" "float64")
threads=(1 2 4)
jobs=(1 2 4)
low_mem=("false" "true")
dequantize=("false" "true")
lambda_min_options=("set_zero") # ("infer" "infer_min" "infer_min_block")

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

        # If jobs > 1, then only run threads = 1:
        #if [ "$j" -gt 1 ] && [ "$t" -gt 1 ]; then
        #  continue
        #fi

        for m in "${low_mem[@]}"
        do

          # If low_mem is is true and threads = 1, loop over the dequantize value; otherwise, keep quantize as false:
          if [ "$m" = "true" ] && [ "$t" -eq 1 ]; then
            for q in "${dequantize[@]}"
            do
              model_id="l${ld}_m${m}_q${q}_t${t}_j${j}_lmo_set_zero"
              sbatch -J "new_viprs_fold_${fold}_${model_id}" 1_analysis/total_runtime_benchmarks/benchmark_new_viprs.sh "fold_$fold" "$ld" "$t" "$j" "$m" "$q"
            done
          else
            model_id="l${ld}_m${m}_qfalse_t${t}_j${j}_lmo_set_zero"
            sbatch -J "new_viprs_fold_${fold}_${model_id}" 1_analysis/total_runtime_benchmarks/benchmark_new_viprs.sh "fold_$fold" "$ld" "$t" "$j" "$m" "false"
          fi

        done
      done
    done
  done

  #for lmo in "${lambda_min_options[@]}"
  #do
  #  model_id="lint8_mfalse_qfalse_t1_j1_lmo_${lmo}"
  #  sbatch -J "new_viprs_fold_${fold}_${model_id}" 1_analysis/total_runtime_benchmarks/benchmark_new_viprs.sh "fold_$fold" "int8" 1 1 "true" "false" "$lmo"
  #done

  # Run sad old VIPRS with its single default setting:
  sbatch -J "old_viprs_fold_$fold" 1_analysis/total_runtime_benchmarks/benchmark_old_viprs.sh "fold_$fold"

done
