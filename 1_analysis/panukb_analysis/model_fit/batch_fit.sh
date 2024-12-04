#!/bin/bash

sumstats_to_fit=${1:-"panukb_sumstats"}
variantset_to_fit=${2:-"hq_imputed_variants_hm3"}
ld_est=${3:-"block"}
ld_dtype=${4:-"int16"}

declare -A time_dict
time_dict["hq_imputed_variants_hm3"]="00:20:00"
time_dict["hq_imputed_variants_maf001"]="03:00:00"
time_dict["hq_imputed_variants"]="06:00:00"

declare -A threads_dict
threads_dict["hq_imputed_variants_hm3"]=4
threads_dict["hq_imputed_variants_maf001"]=4
threads_dict["hq_imputed_variants"]=4


shopt -s nullglob

# Loop over the summary statistics files:
for sumstats_file in data/sumstats/"$sumstats_to_fit"/*/*.sumstats.gz
do
  # Extract the population name from the file path:
  pop=$(basename "$(dirname "$sumstats_file")")
  # Extract the phenotype code from the file name (exclude .sumstats.gz):
  pheno=$(basename "$sumstats_file" .sumstats.gz)
  # Extract the sumstats origin from the file path:
  sumstats_origin=$(basename "$(dirname "$(dirname "$sumstats_file")")")

  # Loop over the available LD matrices for this population:
  for ld_dir in data/l*/"$variantset_to_fit"/"$pop"/"${ld_est}"/"${ld_dtype}"
  do
    # Extract the variant set name from the file path (hq_imputed_variants...):
    variant_set=$(basename "$(dirname "$(dirname "$(dirname "$ld_dir")")")")

    echo "Submitting job for: $sumstats_origin | $variant_set | $pop | $pheno"
    echo "Using ${threads_dict[$variant_set]} threads and time limit: ${time_dict[$variant_set]}"

    # Extract LD matrix characteristics:
    # From the parent directory, extract the LD matrix type (windowed, block):
    ld_dtype=$(basename "$ld_dir")

    # Extract the LD estimator type (windowed, block):
    ld_estimator=$(basename "$(dirname "$ld_dir")")

    # If ld_xarray is present in ld_dir, add _mi, otherwise, add nothing:
    ld_est_data=$(echo "$ld_dir" | grep -q "ld_xarray" && echo "_mi" || echo "")

    ld_info="${ld_estimator}_${ld_dtype}${ld_est_data}"

    mkdir -p "./log/model_fit/${sumstats_origin}/${variant_set}/${ld_info}/${pop}/"

    sbatch --time "${time_dict[$variant_set]}" -J "${sumstats_origin}/${variant_set}/${ld_info}/${pop}/${pheno}" \
            1_analysis/panukb_analysis/model_fit/viprs_fit_job.sh "$sumstats_file" "$ld_dir/chr_*" \
            "data/model_fit/${sumstats_origin}/${variant_set}/${ld_info}/${pop}/${pheno}/" "${threads_dict[$variant_set]}"

  done

done