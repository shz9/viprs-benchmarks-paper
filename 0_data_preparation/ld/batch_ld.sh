#!/bin/bash

VARIANT_SET=${1:-"hq_imputed_variants_hm3"}
LD_ESTIMATOR=${2:-"block"}
STORAGE_DTYPE=${3:-"int8"}
BACKEND=${4:-"xarray"}

pops=("EUR") #("AFR" "AMR" "CSA" "EAS" "EUR" "MID")

declare -A time_dict
time_dict["hq_imputed_variants_hm3"]="08:00:00"
time_dict["hq_imputed_variants_hm3_sbayes_intersect"]="08:00:00"
time_dict["hq_imputed_variants_maf001"]="60:00:00"
time_dict["hq_imputed_variants"]="80:00:00"

declare -A mem_dict
mem_dict["hq_imputed_variants_hm3"]="4GB"
mem_dict["hq_imputed_variants_hm3_sbayes_intersect"]="4GB"
mem_dict["hq_imputed_variants_maf001"]="8GB"
mem_dict["hq_imputed_variants"]="8GB"

var_set="data/keep_files/${VARIANT_SET}.txt"

for pop in "${pops[@]}"
do

  # If variant set is not hm3 and population is not EUR, skip for now:
  if [[ "$VARIANT_SET" != "hq_imputed_variants_hm3" && "$pop" != "EUR" ]]
  then
    continue
  fi

  mkdir -p "./log/data_preparation/ld_mat/${BACKEND}/${VARIANT_SET}/${pop}/${LD_ESTIMATOR}/${STORAGE_DTYPE}"
  for chrom in {1..22}
  do
    echo "${VARIANT_SET} | ${pop} | chr_${chrom}"
    # Set the time for SLURM depending on the variant set:
    sbatch --time "${time_dict[$VARIANT_SET]}" \
           --mem-per-cpu "${mem_dict[$VARIANT_SET]}" \
           -J "${BACKEND}/${VARIANT_SET}/${pop}/${LD_ESTIMATOR}/${STORAGE_DTYPE}/chr_${chrom}" \
           0_data_preparation/ld/compute_ld.sh "$LD_ESTIMATOR" "data/ukbb_qc_genotypes/chr_${chrom}" "$pop" "data/keep_files/ukbb_qc_individuals_${pop}.keep" "$var_set" "${STORAGE_DTYPE}" "${BACKEND}"
  done
done
