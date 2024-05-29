#!/bin/bash

pops=("AFR" "AMR" "CSA" "EAS" "EUR" "MID")

declare -A time_dict
time_dict["hq_imputed_variants_hm3"]="02:00:00"
time_dict["hq_imputed_variants_maf001"]="30:00:00"
time_dict["hq_imputed_variants"]="80:00:00"

for var_set in data/keep_files/hq_imputed_variant*.txt
do
  VARIANT_SET=$(basename $var_set .txt)
  for pop in "${pops[@]}"
  do

    # If variant set is not hm3 and population is not EUR, skip for now:
    if [[ "$VARIANT_SET" != "hq_imputed_variants_hm3" && "$pop" != "EUR" ]]
    then
      continue
    fi

    mkdir -p "./log/data_preparation/ld_mat/${VARIANT_SET}/${pop}/"
    for chrom in {1..22}
    do
      echo "${VARIANT_SET} | ${pop} | chr_${chrom}"
      # Set the time for SLURM depending on the variant set:
      sbatch --time "${time_dict[$VARIANT_SET]}" -J "${VARIANT_SET}/${pop}/chr_${chrom}" 0_data_preparation/ld/compute_ld.sh "windowed" "data/ukbb_qc_genotypes/chr_${chrom}" "$pop" "data/keep_files/ukbb_qc_individuals_${pop}.keep" "$var_set"
    done
  done
done
