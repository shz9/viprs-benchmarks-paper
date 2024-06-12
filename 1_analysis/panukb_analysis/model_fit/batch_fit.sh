#!/bin/bash

declare -A time_dict
time_dict["hq_imputed_variants_hm3"]="00:20:00"
time_dict["hq_imputed_variants_maf001"]="12:00:00"
time_dict["hq_imputed_variants"]="48:00:00"

declare -A threads_dict
threads_dict["hq_imputed_variants_hm3"]=8
threads_dict["hq_imputed_variants_maf001"]=16
threads_dict["hq_imputed_variants"]=22


# Loop over the summary statistics files:
for sumstats_file in data/sumstats/panukb_sumstats/EUR/50.sumstats.gz
do
  # Extract the population name from the file path:
  pop=$(basename "$(dirname "$sumstats_file")")
  # Extract the phenotype code from the file name (exclude .sumstats.gz):
  pheno=$(basename $sumstats_file .sumstats.gz)

  # Loop over the available LD matrices for this population:
  for ld_dir in data/ld/hq_imputed_variants_maf001/$pop
  do
    # Extract the variant set name from the file path:
    variant_set=$(basename "$(dirname "$ld_dir")")

    echo "Submitting job for: $variant_set | $pop | $pheno"
    echo "Using ${threads_dict[$variant_set]} threads and time limit: ${time_dict[$variant_set]}"

    mkdir -p "./log/model_fit/${variant_set}/${pop}/${pheno}"

    sbatch --time "${time_dict[$variant_set]}" -J "${variant_set}/${pop}/${pheno}" \
            1_analysis/panukb_analysis/model_fit/viprs_fit_job.sh "$sumstats_file" "$ld_dir/chr_*" \
            "data/model_fit/panukb_sumstats/$variant_set/$pop/$pheno/" "${threads_dict[$variant_set]}"

  done

done