#!/bin/bash

pops=("AFR" "AMR" "CSA" "EAS" "EUR" "MID")

for var_set in data/keep_files/hq_imputed_variant*.txt
do
  VARIANT_SET=$(basename $var_set .txt)
  for pop in "${pops[@]}"
  do
    mkdir -p "./log/data_preparation/ld_mat/${VARIANT_SET}/${pop}/"
    for chrom in {1..22}
    do
      echo "${VARIANT_SET} | ${pop} | chr_${chrom}"
      sbatch -J "${VARIANT_SET}/${pop}/chr_${chrom}" 0_data_preparation/ld/compute_ld.sh "windowed" "data/ukbb_qc_genotypes/chr_${chrom}" "$pop" "data/keep_files/ukbb_qc_individuals_${pop}.keep" "$var_set"
    done
  done
done
